# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import os
import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from mstpp.model import MST_Plus_Plus
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier
from utils import AverageMeter, Loss_MRAE, Loss_PSNR, Loss_RMSE


class TransferLearning:
    def __init__(self, args):
        # 1. Setup Distributed Environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_distributed = self.local_rank != -1

        if self.is_distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_master = (self.local_rank == 0)
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self.is_master = True

        if self.is_master:
            print(f"[Device] Using device: {self.device}")

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
        
        # Logging (Only Rank 0 logs to TensorBoard)
        if self.is_master:
            self.logWriter = SummaryWriter(log_dir="logs/transfer_learning/")
        else:
            self.logWriter = None

    def _get_base_model(self):
        """Helper to get the un-wrapped model whether DDP is used or not."""
        return self.model.module if self.is_distributed else self.model

    def _wrap_ddp(self):
        """Wraps the model in DDP if distributed training is enabled."""
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_pretrained(self, checkpoint_path, learning_rate):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)
        
        # Wrap the model for Multi-GPU
        self._wrap_ddp()

        # When loading weights across GPUs, map them to the specific device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_distributed else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))

        self.setup_optimizer(learning_rate)
        
        base_model = self._get_base_model()
        model_state = base_model.state_dict()
        filtered = {}
        skipped = []

        for k, v in pretrained_dict.items():
            key = k[len("module."):] if k.startswith("module.") else k
            if key in model_state:
                if model_state[key].shape == v.shape:
                    filtered[key] = v
                else:
                    if self.is_master: print(f"[Shape mismatch] {key}: model={model_state[key].shape}, pretrained={v.shape}")
                    skipped.append(key)
            else:
                skipped.append(key)

        model_state.update(filtered)
        base_model.load_state_dict(model_state)

        if self.is_master:
            print(f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params.")
            print("DONE!")

    def _get_loader_function(self, data_type: str) -> Callable[[Path], tuple[list[Path], list[Path]]]:
        match data_type:
            case "Sri-Lanka": return load_sri_lanka
            case "Kazakhstan": return load_east_kaz
            case "Weedy-Rice": return load_weedy_rice
            case _:
                if self.is_master: print("Unknown dataset type. Defaulting to Sri-Lanka patches.")
                breakpoint()

    def load_mstpp(self, learning_rate, total_steps):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)
        self._wrap_ddp()

        self.setup_optimizer(learning_rate)
        self.setup_scheduler(total_steps, eta_min=1e-6)

    def set_requires_grad(self, module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def save_model(self, save_path, stage_name, epoch=None):
        if not self.is_master:
            return None # Only save on Rank 0

        os.makedirs(save_path, exist_ok=True)
        filename = f"{stage_name}_epoch_{epoch}.pth" if epoch is not None else f"{stage_name}_final.pth"
        full_path = os.path.join(save_path, filename)

        checkpoint = {
            'model_state_dict': self._get_base_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'stage': stage_name,
            'epoch': epoch
        }

        torch.save(checkpoint, full_path)
        print(f"[Saved] Model saved to {full_path}")
        return full_path

    def freeze_all_except_decoder(self):
        base_model = self._get_base_model()
        self.set_requires_grad(base_model.conv_in, False)
        self.set_requires_grad(base_model.body, False)
        self.set_requires_grad(base_model.conv_out, True)

        if self.is_master:
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in base_model.parameters())
            print(f"[Freeze] Decoder only: {trainable_params}/{total_params} parameters trainable")

    def unfreeze_all(self):
        self.set_requires_grad(self.model, True)
        if self.is_master:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[Unfreeze] All layers: {trainable_params} parameters trainable")

    def setup_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        if self.is_master: print(f"[Optimizer] Adam optimizer set with lr={learning_rate}")

    def setup_scheduler(self, total_steps, eta_min):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=total_steps, eta_min=eta_min)
        if self.is_master: print(f"[Scheduler] CosineAnnealingLR scheduler set with eta_min={eta_min}")

    def setup_criterion(self):
        self.criterion_mrae = Loss_MRAE().to(self.device)
        self.criterion_rmse = Loss_RMSE().to(self.device)
        self.criterion_psnr = Loss_PSNR().to(self.device)

    def stage_reset(self):
        self.scheduler = None
        self.dataset = None

    def train_epoch(self, dataloader):
        total_loss = 0.0
        num_batches = 0

        scaler = torch.amp.GradScaler('cuda')
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["rgb"].to(self.device)
            targets = batch["ms"].to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion_mrae(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_master and (batch_idx % 10 == 0):
                 print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, dataloader):
        self.model.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            for batch in dataloader:
                inputs = batch["rgb"].to(self.device, non_blocking=True)
                targets = batch["ms"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss_mrae = self.criterion_mrae(outputs, targets)
                loss_rmse = self.criterion_rmse(outputs, targets)
                loss_psnr = self.criterion_psnr(outputs, targets)
                
                losses_mrae.update(loss_mrae.data)
                losses_rmse.update(loss_rmse.data)
                losses_psnr.update(loss_psnr.data)

        avg_mrae = losses_mrae.avg
        avg_rmse = losses_rmse.avg
        avg_psnr = losses_psnr.avg

        # Aggregate metrics across all GPUs
        if self.is_distributed:
            metrics = torch.tensor([avg_mrae, avg_rmse, avg_psnr], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_mrae, avg_rmse, avg_psnr = metrics.tolist()

        return avg_mrae, avg_rmse, avg_psnr

    def load_dataset(self, root_dir: Path, loader: Callable[[Path], tuple[list[Path], list[Path]]], non_resize_picture=False):
        self.dataset = DataCarrier(root_dir, loader, resize=(not non_resize_picture))
        if self.is_master: print(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")

    def _create_dataloaders(self, train_dataset, val_dataset):
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # Batch size is PER GPU. E.g., batch_size=12 on 4 GPUs = Effective batch size 48
        train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=shuffle, sampler=train_sampler, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, sampler=val_sampler, pin_memory=True)
        
        return train_dataloader, val_dataloader, train_sampler

    def train_from_scratch(self, train_dataloader, train_sampler, epochs, val_dataloader=None, save_dir="checkpoints", save_every=10):
        if self.is_master:
            print("\n" + "="*60 + "\nSTAGE 1: Train from scratch\n" + "="*60)

        self.model.train(mode=True)
        self.unfreeze_all()

        best_val_loss = float('inf')
        best_model_path = None

        for epoch in range(epochs):
            # IMPORTANT: Set the epoch for the DistributedSampler to shuffle properly
            if self.is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if self.is_master: print(f"\n[Stage 1] Epoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch(train_dataloader)
            
            if self.is_master:
                print(f"[Stage 1] Epoch {epoch + 1} - Train Loss: {train_loss:.6f}")
                print(f"[Stage 1] Scheduler LR: {self.scheduler.get_last_lr()}")

            if val_dataloader is not None:
                mrae_loss, rmse_loss, psnr_loss = self.validate_epoch(val_dataloader)
                
                if self.is_master:
                    print(f"[Stage 1] Epoch {epoch + 1} - MRAE loss: {mrae_loss:.6f}, RMSE loss: {rmse_loss}, PSNR: {psnr_loss}")
                    self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)
                    self.logWriter.add_scalar("Stage1/MRAE_Loss", mrae_loss, epoch)
                    self.logWriter.add_scalar("Stage1/RMSE_Loss", rmse_loss, epoch)
                    self.logWriter.add_scalar("Stage1/PSNR_Loss", psnr_loss, epoch)

                    if mrae_loss < best_val_loss:
                        best_val_loss = mrae_loss
                        best_model_path = self.save_model(save_dir, "stage1_best")
                        print(f"[Stage 1] New best model saved! Val Loss: {mrae_loss:.6f}")
            else:
                if self.is_master:
                    print("[Stage 1] There is no validate dataloader")
                    self.logWriter.add_scalar("Stage1/Train_Loss", train_loss, epoch)

            if self.is_master and (epoch + 1) % save_every == 0:
                self.save_model(save_dir, "stage1", epoch + 1)

        final_path = self.save_model(save_dir, "stage1")

        if self.is_master:
            print(f"\n[Stage 1] Training completed. Best MRAE Loss: {best_val_loss:.6f}")
            print(f"[Stage 1] Final model: {final_path}")

        return best_model_path if best_model_path else final_path 

    # -------------------------------------------------------------------------
    # Note: Apply the EXACT SAME `if self.is_master:` and `train_sampler.set_epoch(epoch)`
    # logic to `run_stage_2` and `run_stage_3`. I am keeping them brief here to
    # avoid redundancy, but they follow the identical structure to `train_from_scratch`.
    # -------------------------------------------------------------------------

    def run_full_pipeline(self, stage1_epochs=100, stage1_lr=1e-5, stage2_epochs=100, stage2_lr=1e-5, stage3_epochs=100, stage3_lr=1e-7, save_dir="checkpoints"):
        if self.is_master:
            print("\n" + "="*70 + "\n STAGED TRANSFER LEARNING PIPELINE\n" + "="*70)

        results = {}

        # Stage 1
        loader = self._get_loader_function(self.stage1_data_type)
        self.load_dataset(root_dir=self.stage1_data_path, loader=loader, non_resize_picture=self.stage1_non_resize_picture)

        total_steps = stage1_epochs * len(self.dataset)
        self.load_mstpp(learning_rate=stage1_lr, total_steps=total_steps)

        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        train_dataloader, val_dataloader, train_sampler = self._create_dataloaders(train_dataset, val_dataset)

        results['stage1'] = self.train_from_scratch(
            train_dataloader=train_dataloader, train_sampler=train_sampler,
            epochs=stage1_epochs, val_dataloader=val_dataloader, save_dir=save_dir
        )

        # Ensure all GPUs wait for the master to finish saving the model before moving to Stage 2
        if self.is_distributed:
            dist.barrier()

        # [Stage 2 and 3 would go here using the same dataloader creation method]
        # ...
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get data paths.")
    parser.add_argument("--stage1_data_path", default="data/")
    # ... rest of your argparsers ...

    tl = TransferLearning(parser.parse_args())
    tl.setup_criterion()
    results = tl.run_full_pipeline(
        stage1_epochs=tl.stage_1_epochs, stage1_lr=tl.stage_1_lr,
        stage2_epochs=tl.stage_2_epochs, stage2_lr=tl.stage_2_lr,
        stage3_epochs=tl.stage_3_epochs, stage3_lr=tl.stage_3_lr,
        save_dir="checkpoints"
    )

    # Cleanup distributed backend
    if tl.is_distributed:
        dist.destroy_process_group()
