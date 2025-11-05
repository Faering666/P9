from torch import device
import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from ssr.model import Net


class Opt():
    def __init__(self):
        self.stage = 3 # Number of stages
        self.bands = 4  # Number of output bands
        self.size = 128  # Resize to this

        self.epochs = 15 
        self.batch_size = 1
        self.lr = 1e-4
        self.step_size = 20
        self.gamma = 0.5


class TransferLearning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dataset = None
        self.criterion = None
        self.optimiser = None
        self.options = Opt()

    def load_model(self):
        from ssr.rgb_transfer import SSRNetRGBTransfer
        self.model = SSRNetRGBTransfer(
            self.options,
            load_pretrained_checkpoint="src/baseline_models/model_L.pkl",
            device=self.device
        )
        self.model.to(self.device)
        print(f"[Loaded] Model loaded on {self.device}.")

    def load_dataset(self, root_dir):
        from data_carrier import DataCarrier
        self.dataset = DataCarrier(root_dir, size=self.options.size)
        print(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")

    def loss_function(self):
        self.criterion = torch.nn.L1Loss()

    def optimizer_function(self, learning_rate=1e-4):
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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

    def train(self, epochs):
        print(f"[Training] Training started on {self.device} for {epochs} epochs...")
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

        for epoch in range(epochs):
            # ======== Training Phase ========
            self.model.train()
            train_loss = 0.0
            for data in train_loader:
                rgb = data['rgb'].to(self.device)
                target = data['ms'].to(self.device)

                expected_in_channels = self.model.ssr.initial.in_channels
                f0_channels = self.options.bands
                extra_channels = expected_in_channels - f0_channels

                Phi, PhiPhiT = self._create_dummy_mask(
                    batch_size=rgb.shape[0],
                    H=rgb.shape[2],
                    W=rgb.shape[3],
                    extra_channels=extra_channels
                )

                self.optimiser.zero_grad()
                out = self.model(rgb, input_mask=(Phi, PhiPhiT))[-1]
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

                    out = self.model(rgb, input_mask=(Phi, PhiPhiT))[-1]
                    loss = self.criterion(out, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            new_lr = self.optimiser.param_groups[0]['lr']
            if new_lr != self.optimiser.param_groups[0]['lr']:
                print(f"[LR Scheduler] LR changed to {new_lr:.2e}")

            # ======== Logging ========
            print(f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {self.optimiser.param_groups[0]['lr']:.2e}")

            # ======== Save best model ========
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save("model_best.pkl")

        print(f"[Done] Best val loss: {best_val_loss:.6f}")

    def save(self, path="model_finetuned.pkl"):
        torch.save(self.model.state_dict(), path)
        print(f"[Saved] Model saved to {path}.")


if __name__ == "__main__":
    transfer_learning = TransferLearning()
    transfer_learning.load_model()
    transfer_learning.load_dataset(root_dir="data/")
    transfer_learning.loss_function()
    transfer_learning.optimizer_function(learning_rate=1e-4) # maybe try with lr 5e-5
    transfer_learning.train(epochs=25)
    # transfer_learning.save(path="model_finetuned.pkl")
    transfer_learning.save(path="model_final.pkl")
