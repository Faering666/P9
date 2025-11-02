from torch import device
import torch
from torch.utils.data import DataLoader

from ssr.model import Net


class Opt():
    def __init__(self):
        self.stage = 1  # Number of stages
        self.bands = 4  # Number of output bands
        self.size = 256  # Resize to this

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
            load_pretrained_checkpoint="src/ssr/model_S.pkl",
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
            PhiPhiT = torch.ones(batch_size, 1, H, W, device=self.device)
        else:
            Phi, PhiPhiT = None, None
        return Phi, PhiPhiT

    def train(self, epochs):
        print(f"[Training] Training started on {self.device} for {epochs} epochs...")
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.options.batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for _, data in enumerate(dataloader):
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

                # Forward pass
                self.optimiser.zero_grad()
                out = self.model(rgb, input_mask=(Phi, PhiPhiT))[-1]

                # Loss
                loss = self.criterion(out, target)
                loss.backward()
                self.optimiser.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss / len(dataloader):.6f}")

    def save(self, path="model_finetuned.pkl"):
        torch.save(self.model.state_dict(), path)
        print(f"[Saved] Model saved to {path}.")


if __name__ == "__main__":
    transfer_learning = TransferLearning()
    transfer_learning.load_model()
    transfer_learning.load_dataset(root_dir="data/")
    transfer_learning.loss_function()
    transfer_learning.optimizer_function(learning_rate=1e-4)
    transfer_learning.train(epochs=15)
    transfer_learning.save(path="model_finetuned.pkl")
