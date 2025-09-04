import torch
import torch.nn as nn


class TinyVGG(nn.Module):
    """
    TinyVGG-like CNN for MNIST and Fashion-MNIST with dynamic in_features
    and optional Dropout. Supports custom hidden units per block.

    Architecture
    ------------
    - Two convolutional blocks, each with:
        - Two Conv2d layers with kernel size 3x3
        - ReLU activations
        - MaxPool2d to downsample
        - Optional Dropout
    - Flatten layer
    - Fully connected Linear layer for classification
    - Optional Dropout after Linear

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for grayscale).
    hidden_units : list of int
        Number of filters per convolutional block.
        Example: [32, 64] → 32 filters in block 1, 64 in block 2.
    out_features : int
        Number of classes.
    dropout : float, optional
        Dropout probability between 0 and 1. Default is 0 (no dropout).
    """

    def __init__(self, in_channels: int, hidden_units=[32, 64], out_features: int = 10, dropout: float = 0.0):
        super().__init__()

        assert len(hidden_units) == 2, "hidden_units must have exactly 2 elements for the 2 conv blocks."

        # ------------------------
        # First convolutional block
        # ------------------------
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units[0], out_channels=hidden_units[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # ------------------------
        # Second convolutional block
        # ------------------------
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units[0], out_channels=hidden_units[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units[1], out_channels=hidden_units[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # ------------------------
        # Dynamic calculation of in_features for Linear
        # ------------------------
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 28, 28)
            x = self.conv_block_1(dummy_input)
            x = self.conv_block_2(x)
            in_features = x.numel() // x.shape[0]

        # ------------------------
        # Classifier
        # ------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
