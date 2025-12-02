from torch import device
import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from mstpp.model import MST_Plus_Plus

from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

class Opt():
    def __init__(self):
        self.ckp_path = None
        self.epochs = 100
        self.lr = None
        self.batch_size = 1
        self.size = 256
        self.bands = 4
        # When True, instantiate a fresh MST_Plus_Plus and train from scratch
        self.train_from_scratch = False
        # Model architecture params for scratch training
        self.n_feat = 4
        self.stage = 3
        # Progressive unfreezing options
        # If None, will be set after model is loaded to freeze all but the last body module
        self.progressive_unfreeze = True
        self.freeze_body_initial = None
        # Unfreeze one additional body module every `unfreeze_every` epochs
        self.unfreeze_every = 5


class TransferLearning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dataset = None
        self.criterion = None
        self.optimiser = None
        self.options = Opt()

        # Logging
        self.logWriter = SummaryWriter(log_dir="logs/transfer_learning/")

    def load_model(self):
        # Load pretrained or instantiate from scratch depending on options
        if self.options.train_from_scratch:
            self.model = MST_Plus_Plus(in_channels=3, out_channels=self.options.bands, n_feat=self.options.n_feat,
                                       stage=self.options.stage).to(self.device)
            print(
                f"[Init] Created new MST++ model (n_feat={self.options.n_feat}, stage={self.options.stage}) for training from scratch.")
        else:
            # Load MST++ model checkpoint
            self._load_pretrained(self.options.ckp_path)
            # Apply initial freezing policy (if enabled). This will also rebuild the optimiser
            # to include only trainable parameters.
            try:
                self._initial_freeze()
            except Exception as e:
                print(f"[Warning] Failed to apply initial freeze: {e}")

            print(f"[Loaded] MST++ model loaded from {self.options.ckp_path}.")

        self.logWriter.add_hparams(
            {
                "lr": self.options.lr,
                "batch_size": self.options.batch_size,
                "epochs": self.options.epochs,
                "bands": self.options.bands,

            },
            {}
        )
        # self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        # checkpoint = torch.load(self.options.ckp_path, map_location=self.device, weights_only=False)
        # self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=False)

    def _load_pretrained(self, checkpoint_path):
        self.model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

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

        # NOTE: removed interactive breakpoint for automated runs

    def set_requires_grad(self, module, requires_grad: bool):
        """Recursively set requires_grad for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad = requires_grad

    def save_model(self, save_path, stage_name, epoch=None):
        """
        Save model checkpoint with stage information.

        Args:
            save_path: Directory to save the model
            stage_name: Name of the current stage (e.g., 'stage1', 'stage2', 'stage3')
            epoch: Optional epoch number to include in filename
        """
        os.makedirs(save_path, exist_ok=True)

        if epoch is not None:
            filename = f"{stage_name}_epoch_{epoch}.pth"
        else:
            filename = f"{stage_name}_final.pth"

        full_path = os.path.join(save_path, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict() if self.optimiser else None,
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
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"[Freeze] Decoder only: {trainable_params}/{total_params} parameters trainable")

    def unfreeze_all(self):
        """
        Unfreeze all layers in the model.
        This is used in Stage 3 of transfer learning.
        """
        self.set_requires_grad(self.model, True)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Unfreeze] All layers: {trainable_params} parameters trainable")

    def setup_optimizer(self, learning_rate):
        """
        Setup optimizer with the given learning rate.
        Only includes parameters that require gradients.

        Args:
            learning_rate: Learning rate for the optimizer
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimiser = torch.optim.Adam(trainable_params, lr=learning_rate)
        print(f"[Optimizer] Adam optimizer set with lr={learning_rate}")

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimiser.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate_epoch(self, dataloader):
        """
        Validate for one epoch.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass only
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def run_stage_1(self, save_dir="checkpoints"):
        """
        Stage 1: Load or train base model.
        The model is loaded via load_model() which is called before this.
        This stage just saves the initial model.

        Args:
            save_dir: Directory to save checkpoints
        """
        print("\n" + "="*60)
        print("STAGE 1: Base Model Loading/Training")
        print("="*60)

        # Model should already be loaded via load_model()
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Save the base model
        model_path = self.save_model(save_dir, "stage1")
        print(f"[Stage 1] Base model saved: {model_path}")

        return model_path

    def run_stage_2(self, train_dataloader, epochs, val_dataloader=None, learning_rate=1e-5, save_dir="checkpoints", save_every=10):
        """
        Stage 2: Freeze all layers except decoder, train with medium-high learning rate.

        Args:
            train_dataloader: DataLoader for training data
            epochs: Number of epochs to train
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Learning rate (default: 1e-5)
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Path to final stage 2 model
        """
        print("\n" + "="*60)
        print("STAGE 2: Decoder Training (Frozen Encoder)")
        print("="*60)

        # Freeze all except decoder
        self.freeze_all_except_decoder()

        # Setup optimizer with specified learning rate
        self.setup_optimizer(learning_rate)

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
                val_loss = self.validate_epoch(val_dataloader)
                print(f"[Stage 2] Epoch {epoch + 1} - Val Loss: {val_loss:.6f}")

                # Log to tensorboard
                self.logWriter.add_scalar("Stage2/Train_Loss", train_loss, epoch)
                self.logWriter.add_scalar("Stage2/Val_Loss", val_loss, epoch)

                # Save best model when validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = self.save_model(save_dir, "stage2_best")
                    print(f"[Stage 2] New best model saved! Val Loss: {val_loss:.6f}")
            else:
                # Log to tensorboard (training only)
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
            print(f"\n[Stage 2] Training completed.")
        print(f"[Stage 2] Final model: {final_path}")

        return best_model_path if best_model_path else final_path

    def run_stage_3(self, train_dataloader, epochs, val_dataloader=None, learning_rate=1e-7, save_dir="checkpoints", save_every=10):
        """
        Stage 3: Unfreeze all layers, fine-tune with low learning rate.

        Args:
            train_dataloader: DataLoader for training data
            epochs: Number of epochs to train
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Learning rate (default: 1e-7)
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Path to final stage 3 model
        """
        print("\n" + "="*60)
        print("STAGE 3: Full Model Fine-tuning (All Layers Unfrozen)")
        print("="*60)

        # Unfreeze all layers
        self.unfreeze_all()

        # Setup optimizer with lower learning rate
        self.setup_optimizer(learning_rate)

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
                val_loss = self.validate_epoch(val_dataloader)
                print(f"[Stage 3] Epoch {epoch + 1} - Val Loss: {val_loss:.6f}")

                # Log to tensorboard
                self.logWriter.add_scalar("Stage3/Train_Loss", train_loss, epoch)
                self.logWriter.add_scalar("Stage3/Val_Loss", val_loss, epoch)

                # Save best model when validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = self.save_model(save_dir, "stage3_best")
                    print(f"[Stage 3] New best model saved! Val Loss: {val_loss:.6f}")
            else:
                # Log to tensorboard (training only)
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
            print(f"\n[Stage 3] Training completed.")
        print(f"[Stage 3] Final model: {final_path}")

        return best_model_path if best_model_path else final_path

    def run_full_pipeline(self, train_dataloader, stage2_epochs, stage3_epochs,
                          val_dataloader=None, stage2_lr=1e-5, stage3_lr=1e-7, save_dir="checkpoints"):
        """
        Run the complete 3-stage transfer learning pipeline.

        Args:
            train_dataloader: DataLoader for training data
            stage2_epochs: Number of epochs for stage 2
            stage3_epochs: Number of epochs for stage 3
            val_dataloader: Optional DataLoader for validation data
            stage2_lr: Learning rate for stage 2 (default: 1e-5)
            stage3_lr: Learning rate for stage 3 (default: 1e-7)
            save_dir: Directory to save all checkpoints

        Returns:
            Dictionary with paths to all saved models
        """
        print("\n" + "="*70)
        print(" STAGED TRANSFER LEARNING PIPELINE")
        print("="*70)

        results = {}

        # Stage 1: Base model
        results['stage1'] = self.run_stage_1(save_dir)

        # Stage 2: Decoder training
        results['stage2'] = self.run_stage_2(
            train_dataloader, stage2_epochs, val_dataloader=val_dataloader,
            learning_rate=stage2_lr, save_dir=save_dir
        )

        # Stage 3: Full fine-tuning
        results['stage3'] = self.run_stage_3(
            train_dataloader, stage3_epochs, val_dataloader=val_dataloader,
            learning_rate=stage3_lr, save_dir=save_dir
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

    # Initialize the transfer learning pipeline
    tl = TransferLearning()

    # Configure options
    tl.options.ckp_path = "path/to/pretrained_model.pth"  # Or set train_from_scratch=True
    tl.options.train_from_scratch = False
    tl.options.bands = 4
    tl.options.n_feat = 4
    tl.options.stage = 3

    # Setup criterion
    tl.criterion = torch.nn.MSELoss()

    # Load the model (Stage 1)
    tl.load_model()

    # Prepare your dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Run the full 3-stage pipeline with validation
    results = tl.run_full_pipeline(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,  # Optional: will save best model when val_loss improves
        stage2_epochs=50,      # Train decoder for 50 epochs
        stage3_epochs=30,      # Fine-tune all layers for 30 epochs
        stage2_lr=1e-5,        # Medium-high learning rate for stage 2
        stage3_lr=1e-7,        # Low learning rate for stage 3
        save_dir="checkpoints"
    )

    # Or run stages individually for more control:
    # tl.run_stage_1(save_dir="checkpoints")
    # tl.run_stage_2(train_dataloader, epochs=50, val_dataloader=val_dataloader,
    #                learning_rate=1e-5, save_dir="checkpoints")
    # tl.run_stage_3(train_dataloader, epochs=30, val_dataloader=val_dataloader,
    #                learning_rate=1e-7, save_dir="checkpoints")
