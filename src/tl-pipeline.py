# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

from pathlib import Path
from typing import Callable
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
from mstpp.model import MST_Plus_Plus
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from utils import AverageMeter, Loss_MRAE, Loss_PSNR, Loss_RMSE


class TransferLearning:
    def __init__(self, args):
        # 1. Setup Distributed Environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        if self.local_rank != -1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_distributed = True
            self.is_master = (self.local_rank == 0) # Only Rank 0 logs/saves
        else:
            # Fallback for single GPU/CPU debugging
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.is_distributed = False
            self.is_master = True
        
        # Args
        self.stage1_data_path = Path(args.stage1_data_path)
        self.stage1_data_type = args.stage1_data_type
        self.stage1_non_resize_picture = args.stage1_non_resize
        self.stage_1_epochs = args.stage1_epochs
        self.stage_1_lr = args.stage1_lr

        self.stage2_data_path = Path(args.stage2_data_path)
        self.stage2_data_type = args.stage2_data_type
        self.stage2_non_resize_picture = args.stage2_non_resize
        self.stage2_model = args.stage2_model
        self.stage_2_epochs = args.stage2_epochs
        self.stage_2_lr = args.stage2_lr
                
        self.stage3_data_path = Path(args.stage3_data_path)
        self.stage3_data_type = args.stage3_data_type
        self.stage3_non_resize_picture = args.stage3_non_resize
        self.stage3_model = args.stage3_model
        self.stage_3_epochs = args.stage3_epochs
        self.stage_3_lr = args.stage3_lr

        # Model details
        self.model = None
        self.dataset: DataCarrier = None
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = None

        # Loss functions
        self.criterion_mrae: Loss_MRAE = None
        self.criterion_rmse: Loss_RMSE = None
        self.criterion_psnr: Loss_PSNR = None
        
        # Logging
        if self.is_master:
            self.logWriter = SummaryWriter(log_dir="logs/transfer_learning/")
        else:
            self.logWriter = None


    def _load_pretrained(self, checkpoint_path, learning_rate):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        self.setup_optimizer(learning_rate)
        # self.setup_scheduler(...) # TODO consider to add a scheduler
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []

        for k, v in pretrained_dict.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]

            if key in model_state:
                if model_state[key].shape == v.shape:
                    filtered[key] = v
                else:
                    print(f"[Shape mismatch] {key}: model={model_state[key].shape}, pretrained={v.shape}")
                    skipped.append(key)
            else:
                skipped.append(key)

        # Update and load
        model_state.update(filtered)
        self.model.load_state_dict(model_state)

        print(
            f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params (incompatible shapes).")
        if skipped:
            print("Skipped keys:", skipped[:10], "..." if len(skipped) > 10 else "")
        print("DONE!")

    def _get_loader_function(self, data_type: str) -> Callable[[Path], tuple[list[Path], list[Path]]]:
        match data_type:
            case "Sri-Lanka":
                return load_sri_lanka
            case "Kazakhstan":
                return load_east_kaz
            case "Weedy-Rice":
                return load_weedy_rice
            case _:
                print("Unknown dataset type. Defaulting to Sri-Lanka patches.")
                breakpoint() #Dummefejl

    def load_mstpp(self, learning_rate, total_steps):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)

        self.setup_optimizer(learning_rate)
        self.setup_scheduler(total_steps, eta_min=1e-6)

        if self.is_distributed:
            # Wrap the model
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def set_requires_grad(self, module, requires_grad: bool):
        """Recursively set requires_grad for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad = requires_grad

    def save_model(self, save_path, stage_name, epoch=None):
        os.makedirs(save_path, exist_ok=True)

        if epoch is not None:
            filename = f"{stage_name}_epoch_{epoch}.pth"
        else:
            filename = f"{stage_name}_final.pth"

        full_path = os.path.join(save_path, filename)

        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'stage': stage_name,
            'epoch': epoch
        }

        torch.save(checkpoint, full_path)
        print(f"[Saved] Model saved to {full_path}")
        return full_path

    def freeze_all_except_decoder(self):
        """
        Freeze all layers except the decoder (conv_out layer).
        This is used in Stage 2 of transfer learning.
        """
        # Freeze conv_in and body
        self.set_requires_grad(self.model.conv_in, False)
        self.set_requires_grad(self.model.body, False)

        # Unfreeze conv_out (decoder)
        self.set_requires_grad(self.model.conv_out, True)

        # Count trainable parameters
        trainable_params = len(list(p.numel() for p in self.model.parameters() if p.requires_grad))
        total_params = len(list(p.numel() for p in self.model.parameters()))

        print(f"[Freeze] Decoder only: {trainable_params}/{total_params} parameters trainable")

    def unfreeze_all(self):
        self.set_requires_grad(self.model, True)
        trainable_params = len(list(p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f"[Unfreeze] All layers: {trainable_params} parameters trainable")

    def setup_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        print(f"[Optimizer] Adam optimizer set with lr={learning_rate}")

    def setup_scheduler(self, total_steps, eta_min):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=total_steps,
            eta_min=eta_min
        )
        print(f"[Scheduler] CosineAnnealinLR schedular set with eta_min={eta_min}")

    def setup_criterion(self):
        self.criterion_mrae = Loss_MRAE()
        self.criterion_rmse = Loss_RMSE()
        self.criterion_psnr = Loss_PSNR()

        if torch.cuda.is_available():
            self.criterion_mrae.cuda()
            self.criterion_rmse.cuda()
            self.criterion_psnr.cuda()

    def stage_reset(self):
        self.scheduler = None
        self.dataset = None

    def train_epoch(self, dataloader):
        total_loss = 0.0
        num_batches = 0

        scaler = torch.amp.GradScaler()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            inputs = batch["rgb"].to(self.device)
            targets = batch["ms"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion_mrae(outputs, targets)

            # Backward pass
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            #self.optimiser.step()

            total_loss += loss.item()
            num_batches += 1

            scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            if (batch_idx) % 10 == 0:
                 print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate_epoch(self, dataloader):
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                for batch in dataloader:
                    inputs = batch["rgb"].to(self.device, non_blocking=True)
                    targets = batch["ms"].to(self.device, non_blocking=True)

                    # Forward pass only
                    outputs = self.model(inputs)
                    loss_mrae = self.criterion_mrae(outputs, targets)
                    loss_rmse = self.criterion_rmse(outputs, targets)
                    loss_psnr = self.criterion_psnr(outputs, targets)
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)

        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

    def load_dataset(self, root_dir: Path, loader: Callable[[Path], tuple[list[Path], list[Path]]], non_resize_picture=False):
        self.dataset = DataCarrier(root_dir, loader, resize=(not non_resize_picture)) # non_resize_picture is default False
        print(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")

    def train_from_scratch(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        print("\n" + "="*60)
        print("STAGE 1: Train from scratch")
        print("="*60)

        # Set model mode train
        self.model.train(mode=True)

        # Unfreeze all layers
        self.unfreeze_all()

        # Track best validation loss
        best_val_loss = float('inf')
        best_model_path = None

        # Training loop
        for epoch in range(epochs):
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            print(f"\n[Stage 1] Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"[Stage 1] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")
            print(f"[Stage 1] Scheduler LR: {self.scheduler.get_last_lr()}")

            # Validation
            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)
                print(f"[Stage 1] Epoch {epoch + 1} - MRAE loss: {mrae_loss:.6f}, RMSE loss: {rmse_loss}, PSNR: {psnr_loss}")

                # Log to tensorboard
                if self.is_master:
                    self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)
                    self.logWriter.add_scalar("Stage1/MRAE_Loss", mrae_loss, epoch)
                    self.logWriter.add_scalar("Stage1/RMSE_Loss", rmse_loss, epoch)
                    self.logWriter.add_scalar("Stage1/PSNR_Loss", psnr_loss, epoch)

                # Save best model when validation loss improves
                if mrae_loss < best_val_loss:
                    best_val_loss = mrae_loss
                    if not self.is_master:
                        best_model_path = self.save_model(save_dir, "stage1_best")
                    print(f"[Stage 1] New best model saved! Val Loss: {mrae_loss:.6f}")
            else:
                # Log to tensorboard (training only)
                print("[Stage 1] There is no validate dataloader")
                self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                if not self.is_master:
                    return
                self.save_model(save_dir, "stage1", epoch + 1)

        # Save final model
        if not self.is_master:
            final_path = self.save_model(save_dir, "stage1")

        if val_dataloader is not None:
            print(f"\n[Stage 1] Training completed. Best MRAE Loss: {best_val_loss:.6f}")
            print(f"[Stage 1] Best model: {best_model_path}")
        else:
            print(f"\n[Stage 1] Training completed. (No validate dataloader)")
        print(f"[Stage 1] Final model: {final_path}")

        return best_model_path if best_model_path else final_path 

    def run_stage_2(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        print("\n" + "="*60)
        print("STAGE 2: Decoder Training (Frozen Encoder)")
        print("="*60)

        # Set model mode train
        self.model.train(mode=True)

        # Freeze all except decoder
        self.freeze_all_except_decoder()


        # Track best validation loss
        best_val_loss = float('inf')
        best_model_path = None

        # Training loop
        for epoch in range(epochs):
            print(f"\n[Stage 2] Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"[Stage 2] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")

            # Validation
            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)
                print(f"[Stage 2] Epoch {epoch + 1} - MRAE loss: {mrae_loss:.6f}, RMSE loss: {rmse_loss}, PSNR: {psnr_loss}")

                # Log to tensorboard
                self.logWriter.add_scalar("Stage2/Train_Loss", train_loss, epoch)
                self.logWriter.add_scalar("Stage2/MRAE_Loss", mrae_loss, epoch)
                self.logWriter.add_scalar("Stage2/RMSE_Loss", rmse_loss, epoch)
                self.logWriter.add_scalar("Stage2/PSNR_Loss", psnr_loss, epoch)

                # Save best model when validation loss improves
                if mrae_loss < best_val_loss:
                    best_val_loss = mrae_loss
                    best_model_path = self.save_model(save_dir, "stage2_best")
                    print(f"[Stage 2] New best model saved! MRAE Loss: {mrae_loss:.6f}")
            else:
                # Log to tensorboard (training only)
                print("[Stage 2] There is no validate dataloader")
                self.logWriter.add_scalar("Stage2/Train_Loss", train_loss, epoch)

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_model(save_dir, "stage2", epoch + 1)

        # Save final model
        final_path = self.save_model(save_dir, "stage2")

        if val_dataloader is not None:
            print(f"\n[Stage 2] Training completed. Best Val Loss: {best_val_loss:.6f}")
            print(f"[Stage 2] Best model: {best_model_path}")
        else:
            print(f"\n[Stage 2] Training completed. (No validate dataloader)")
        print(f"[Stage 2] Final model: {final_path}")

        return best_model_path if best_model_path else final_path

    def run_stage_3(self, train_dataloader, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        print("\n" + "="*60)
        print("STAGE 3: Full Model Fine-tuning (All Layers Unfrozen)")
        print("="*60)

        # Set model mode train
        self.model.train(mode=True)

        # Unfreeze all layers
        self.unfreeze_all()

        # Track best validation loss
        best_val_loss = float('inf')
        best_model_path = None

        # Training loop
        for epoch in range(epochs):
            print(f"\n[Stage 3] Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"[Stage 3] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")

            # Validation
            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)
                print(f"[Stage 3] Epoch {epoch + 1} - MRAE loss: {mrae_loss:.6f}, RMSE loss: {rmse_loss}, PSNR: {psnr_loss}")

                # Log to tensorboard
                self.logWriter.add_scalar("Stage3/Train_Loss", train_loss, epoch)
                self.logWriter.add_scalar("Stage3/MRAE_Loss", mrae_loss, epoch)
                self.logWriter.add_scalar("Stage3/RMSE_Loss", rmse_loss, epoch)
                self.logWriter.add_scalar("Stage3/PSNR_Loss", psnr_loss, epoch)

                # Save best model when validation loss improves
                if mrae_loss < best_val_loss:
                    best_val_loss = mrae_loss
                    best_model_path = self.save_model(save_dir, "stage3_best")
                    print(f"[Stage 3] New best model saved! Val Loss: {mrae_loss:.6f}")
            else:
                # Log to tensorboard (training only)
                print("[Stage 3] There is no validate dataloader")
                self.logWriter.add_scalar("Stage3/Train_Loss", train_loss, epoch)

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_model(save_dir, "stage3", epoch + 1)

        # Save final model
        final_path = self.save_model(save_dir, "stage3")

        if val_dataloader is not None:
            print(f"\n[Stage 3] Training completed. Best Val Loss: {best_val_loss:.6f}")
            print(f"[Stage 3] Best model: {best_model_path}")
        else:
            print(f"\n[Stage 3] Training completed. (No validate dataloader)")
        print(f"[Stage 3] Final model: {final_path}")

        return best_model_path if best_model_path else final_path

    def run_full_pipeline(self,
                          stage1_epochs=100,
                          stage1_lr=1e-5,
                          stage2_epochs=100,
                          stage2_lr=1e-5,
                          stage3_epochs=100,
                          stage3_lr=1e-7,
                          save_dir="checkpoints"):
        print("\n" + "="*70)
        print(" STAGED TRANSFER LEARNING PIPELINE")
        print("="*70)

        results = {}

        # --------------------------------
        # Stage 1 train model from scratch
        # --------------------------------
        loader = self._get_loader_function(self.stage1_data_type)
        self.load_dataset(root_dir=self.stage1_data_path, loader=loader, non_resize_picture=self.stage1_non_resize_picture)

        total_steps = stage1_epochs * len(self.dataset)
        self.load_mstpp(learning_rate=stage1_lr, total_steps=total_steps)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        # Prepare your dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

        results['stage1'] = self.train_from_scratch(
            train_dataloader=train_dataloader,
            epochs=stage1_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )

        # --------------------------------
        # Stage 2: Decoder training
        # --------------------------------
        self.stage_reset()
        loader = self._get_loader_function(self.stage2_data_type)
        self.load_dataset(root_dir=self.stage2_data_path, loader=loader, non_resize_picture=self.stage2_non_resize_picture)

        model_path = results["stage1"] if self.stage2_model is None else self.stage2_model
        self._load_pretrained(model_path, learning_rate=stage2_lr)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        # Prepare your dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=12, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

        results['stage2'] = self.run_stage_2(
            train_dataloader=train_dataloader,
            epochs=stage2_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )


        # --------------------------------
        # Stage 3: Full fine-tuning
        # --------------------------------
        self.stage_reset()
        loader = self._get_loader_function(self.stage3_data_type)
        self.load_dataset(root_dir=self.stage3_data_path, loader=loader, non_resize_picture=self.stage3_non_resize_picture)
        
        model_path = results["stage2"] if self.stage3_model is None else self.stage3_model
        self._load_pretrained(model_path, learning_rate=stage3_lr)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        # Prepare your dataloaders
        if self.is_distributed:
            sampler = DistributedSampler(train_dataset, shuffle=True)
            shuffle = False # Sampler handles shuffling
        else:
            sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=12, # Note: This is now "Batch Size Per GPU"
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True
        )
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

        results['stage3'] = self.run_stage_3(
            train_dataloader=train_dataloader,
            epochs=stage3_epochs,
            val_dataloader=val_dataloader,
            save_dir=save_dir
        )

        print("\n" + "="*70)
        print(" PIPELINE COMPLETED")
        print("="*70)
        print("\nSaved models:")
        for stage, path in results.items():
            print(f"  {stage}: {path}")

        return results


# Usage example
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get data paths.")
    parser.add_argument("--stage1_data_path", default="data/")
    parser.add_argument("--stage1_data_type", help="Which dataset", default="Kazakhstan")
    parser.add_argument("--stage1_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage1_epochs", type=int, help="Number of epochs for stage 1 (train from scratch). To skip set epochs to '0'", default=0)
    parser.add_argument("--stage1_lr", type=float, help="Learning rate for stage 1 (train from scratch)", default=4e-4)
    
    parser.add_argument("--stage2_data_path", default="data/")
    parser.add_argument("--stage2_data_type", help="Which dataset", default="Kazakhstan")
    parser.add_argument("--stage2_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage2_model", type=str, help="(Optional) Path to model for stage 2", default=None)
    parser.add_argument("--stage2_epochs", type=int, help="Number of epochs for stage 2. To skip set epochs to '0'", default=0)
    parser.add_argument("--stage2_lr", type=float, help="Learning rate for stage 2", default=1e-5)

    parser.add_argument("--stage3_data_path", default="data/")
    parser.add_argument("--stage3_data_type", help="Which dataset Sri-Lanka or Kazakhstan, default=Sri-Lanka", default="Weedy-Rice")
    parser.add_argument("--stage3_non_resize", type=bool, help="Use non-resized pictures, default=False", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage3_model", type=str, help="(Optional) Path to model for stage 3", default=None)
    parser.add_argument("--stage3_epochs", type=int, help="Number of epochs for stage 3. To skip set epochs to '0'", default=0)
    parser.add_argument("--stage3_lr", type=float, help="Learning rate for stage 3", default=1e-7)

    # Initialize the transfer learning pipeline
    tl = TransferLearning(parser.parse_args())

    # Setup criterion
    tl.setup_criterion()

    # Run the full 3-stage pipeline with validation
    results = tl.run_full_pipeline(
        stage1_epochs=tl.stage_1_epochs,
        stage1_lr=tl.stage_1_lr,
        stage2_epochs=tl.stage_2_epochs, # Train decoder for 50 epochs
        stage2_lr=tl.stage_2_lr,         # Medium-high learning rate for stage 2
        stage3_epochs=tl.stage_3_epochs, # Fine-tune all layers for 30 epochs
        stage3_lr=tl.stage_3_lr,         # Low learning rate for stage 3
        save_dir="checkpoints"
    )
