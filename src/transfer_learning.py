from torch import device
import torch

from ssr.model import Net


class Opt():
    def __init__(self):
        self.stage = 1 # Number of stages
        self.bands = 4 # Number of output bands
        self.size = 256 # Resize to this

        self.epochs = 5
        self.batch_size = 16
        self.lr = 1e-4
        self.step_size = 20
        self.gamma = 0.5


class TransferLearning:
    def __init__(self):
        """
        Initializes the TransferLearning class with default parameters.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None 
        self.dataset = None
        self.criterion = None
        self.optimiser = None
        self.options = Opt()


    def load_model(self):
        """
        Loads the pre-trained model onto the specified device.
        """
        self.model = Net(self.options).to(self.device)
        checkpoint = torch.load("src/ssr/model_L.pkl", map_location=self.device, weights_only=False)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(pretrained_dict, strict=False)
        print(f"[Loaded] Model loaded on {self.device}.")


    def load_dataset(self, root_dir):
        """
        Loads the dataset from the specified directory.
        Args:
            root_dir (str): Path to the root directory of the dataset.
        """
        from data_carrier import DataCarrier
        self.dataset = DataCarrier(root_dir, size=self.options.size)
        print(f"[Loaded] Dataset loaded with {len(self.dataset)} samples.")


    def loss_function(self):
        """
        Defines the loss function for training.
        """
        self.criterion = torch.nn.L1Loss()

    
    def optimizer_function(self, learning_rate=1e-4):
        """
        Defines the optimizer for training.
        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        
    def train(self, epochs):
        """
        Trains the model for a specified number of epochs.
        Args:
            epochs (int): Number of epochs to train the model.
        """
        print(f"[Training] Training started...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for rgb, target in self.dataset:
                rgb, target = rgb.to(self.device), target.to(self.device)

                if len(rgb.shape) == 3:
                    rgb = rgb.unsqueeze(0)  # Add batch dimension if missing
        
                # Convert RGB to single-channel grayscale
                g = torch.mean(rgb, dim=1, keepdim=True) # (B1,H,W)

                # Create dummy mask for each batch
                dummy_mask = _create_dummy_mask(g.shape[0], self.options.bands, g.shape[2], g.shape[3], self.device)

                # Forward pass with mask (model expects 4-channel input RGB+Mask)
                out = self.model(g, dummy_mask)
                preds = out[-1] # Get the output of the last stage
                
                loss = self.criterion(preds, target)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.dataset):.4f}")


    def save(self, path="model_finetuned.pkl"):
        """
        Saves the fine-tuned model to the specified path.
        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"[Saved] Model saved to {path}.")


def _create_dummy_mask(batch_size, bands, H, W, device):
    """Create a dummy sensing mask (Phi) and its self-product (PhiPhiT)"""
    Phi = torch.ones(batch_size, bands, H, W, device=device)
    PhiPhiT = torch.ones(batch_size, 1, H, W, device=device)
    return (Phi, PhiPhiT)

if __name__ == "__main__":
    transfer_learning = TransferLearning()
    transfer_learning.load_model()
    transfer_learning.load_dataset(root_dir="data/")
    transfer_learning.loss_function()
    transfer_learning.optimizer_function(learning_rate=1e-4)
    transfer_learning.train(epochs=5)
    transfer_learning.save(path="model_finetuned.pkl")