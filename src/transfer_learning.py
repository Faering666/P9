from torch import device
import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from mstpp.model import MST_Plus_Plus

from torch.utils.tensorboard import SummaryWriter


class Opt():
    def __init__(self):
        self.ckp_path = "src/mstpp/mst_plus_plus.pth"
        self.epochs = 3
        self.lr = 1e-4
        self.batch_size = 1
        self.size = 208
        self.bands = 4
        # When True, instantiate a fresh MST_Plus_Plus and train from scratch
        self.train_from_scratch = False
        # Model architecture params for scratch training
        self.n_feat = 4
        self.stage = 3
        # Progressive unfreezing options
        # If None, will be set after model is loaded to freeze all but the last body module
        self.progressive_unfreeze = False
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
            self.model = MST_Plus_Plus(in_channels=3, out_channels=self.options.bands, n_feat=self.options.n_feat, stage=self.options.stage).to(self.device)
            print(f"[Init] Created new MST++ model (n_feat={self.options.n_feat}, stage={self.options.stage}) for training from scratch.")
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

        print(f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params (incompatible shapes).")
        if skipped:
            print("Skipped keys:", skipped[:10], "..." if len(skipped) > 10 else "")
        print("DONE!")

        # NOTE: removed interactive breakpoint for automated runs

    def set_requires_grad(self, module, requires_grad: bool):
        """Recursively set requires_grad for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _initial_freeze(self):
        """Apply initial freezing according to options. Freezes a prefix of body modules.

        If options.freeze_body_initial is None it will default to freezing all but the
        last body module (so at least one body module is trainable).
        """
        if not self.options.progressive_unfreeze:
            return

        total = len(self.model.body)
        if self.options.freeze_body_initial is None:
            # freeze all but last module by default
            k = max(0, total - 1)
            self.options.freeze_body_initial = k
        else:
            k = int(self.options.freeze_body_initial)

        # Freeze modules 0..k-1, leave k..end trainable
        for i, mod in enumerate(self.model.body):
            if i < k:
                self.set_requires_grad(mod, False)
            else:
                self.set_requires_grad(mod, True)


        self._frozen_body_count = k
        print(f"[Freeze] Initially froze {self._frozen_body_count} body modules out of {total}.")
        # Rebuild optimiser so it only includes trainable params
        self._rebuild_optimizer()


    def _unfreeze_step(self):
        """Unfreeze one additional body module from the frozen prefix (right-to-left).
        Returns True if something was unfrozen.
        """
        if not hasattr(self, "_frozen_body_count"):
            return False
        if self._frozen_body_count <= 0:
            return False

        # Unfreeze the last frozen module index
        idx = self._frozen_body_count - 1
        self.set_requires_grad(self.model.body[idx], True)
        self._frozen_body_count -= 1
        print(f"[Unfreeze] Unfroze body module {idx}. Remaining frozen: {self._frozen_body_count}")
        # Rebuild optimiser to include newly trainable params
        self._rebuild_optimizer()
        return True

    def _rebuild_optimizer(self):
        """Recreate the optimiser to include only parameters with requires_grad=True."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimiser = torch.optim.Adam(params, lr=self.options.lr)
        n_params = sum(1 for _ in params)
        print(f"[Optimiser] Rebuilt optimiser with {n_params} parameter tensors (trainable).")

    def load_dataset(self, root_dir):
        from data_carrier import DataCarrier
        self.dataset = DataCarrier(root_dir)
        print(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")

    def loss_function(self):
        self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.nn.MSELoss()

    def optimizer_function(self):
        # Only include parameters that require gradients (respecting any freezes)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimiser = torch.optim.Adam(params, lr=self.options.lr)

    def _create_dummy_mask(self, batch_size, H, W, extra_channels):
        """
        Create dummy Phi and PhiPhiT masks to match model input channels.
        """
        if extra_channels > 0:
            Phi = torch.ones(batch_size, extra_channels, H, W, device=self.device)
            PhiPhiT = torch.mean(Phi ** 2, dim=1, keepdim=True) + 1e-6
        else:
            Phi, PhiPhiT = None, None
        return Phi, PhiPhiT

    def train(self):
        print(f"[Training] Training started on {self.device} for {self.options.epochs} epochs...")
        self.model.train()

        # Split dataset into 90% train / 10% val
        total_len = len(self.dataset)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_dataset, val_dataset = random_split(self.dataset, [train_len, val_len])

        train_loader = DataLoader(train_dataset, batch_size=self.options.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # LR scheduler (reduce LR when val loss plateaus)
        scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=3)

        best_val_loss = float('inf')

        for epoch in range(self.options.epochs):
            # Progressive unfreezing schedule: unfreeze one body module every `unfreeze_every` epochs
            if self.options.progressive_unfreeze and epoch > 0 and (epoch % self.options.unfreeze_every == 0):
                changed = self._unfreeze_step()
                if changed:
                    # Optimiser was rebuilt; recreate scheduler to attach to the new optimiser
                    scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=3)

            # ======== Training Phase ========
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                rgb = data['rgb'].to(self.device)
                target = data['ms'].to(self.device)

                expected_in_channels = self.options.bands
                f0_channels = self.options.bands
                extra_channels = expected_in_channels - f0_channels

                Phi, PhiPhiT = self._create_dummy_mask(
                    batch_size=rgb.shape[0],
                    H=rgb.shape[2],
                    W=rgb.shape[3],
                    extra_channels=extra_channels
                )

                self.optimiser.zero_grad()
                out = self.model(rgb)
                loss = self.criterion(out, target)
                loss.backward()
                self.optimiser.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ======== Validation Phase ========
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    rgb = data['rgb'].to(self.device)
                    target = data['ms'].to(self.device)

                    Phi, PhiPhiT = self._create_dummy_mask(
                        batch_size=rgb.shape[0],
                        H=rgb.shape[2],
                        W=rgb.shape[3],
                        extra_channels=extra_channels
                    )

                    out = self.model(rgb)
                    loss = self.criterion(out, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            # step scheduler and print if LR changed
            old_lr = self.optimiser.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = self.optimiser.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[LR Scheduler] LR changed to {new_lr:.2e}")

            # ======== Logging ========
            print(f"Epoch [{epoch+1}/{self.options.epochs}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {self.optimiser.param_groups[0]['lr']:.2e}")

            # Tensorboard logging
            self.logWriter.add_scalar("Loss/Train", train_loss, epoch)
            self.logWriter.add_scalar("Loss/Val", val_loss, epoch)
            self.logWriter.add_scalar("LR", self.optimiser.param_groups[0]['lr'], epoch)

            # ======== Save best model ========
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("model_best.pkl")

        print(f"[Done] Best val loss: {best_val_loss:.6f}")

    def save(self, path="model_finetuned.pkl"):
        torch.save(self.model.state_dict(), path)
        print(f"[Saved] Model saved to {path}.")
        self.logWriter.close()


if __name__ == "__main__":
    transfer_learning = TransferLearning()
    transfer_learning.options.train_from_scratch = True
    transfer_learning.load_model()
    transfer_learning.load_dataset(root_dir="data/")
    transfer_learning.loss_function()
    transfer_learning.optimizer_function()
    transfer_learning.train()
    transfer_learning.save(path="model_final.pkl")
