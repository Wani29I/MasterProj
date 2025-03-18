import torch
import torch.nn as nn
import torch.nn.functional as F


class WheatEarModel(nn.Module):
    def __init__(self):
        super(WheatEarModel, self).__init__()

        # CNN for RGB (3 channels)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # CNN for DSM (1 channel)
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers to combine both
        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 64 + 64 * 32 * 64, 512),  # Adjust based on feature map size
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output one value (regression)
        )

    def forward(self, rgb, dsm):
        # Pass RGB through CNN
        rgb_features = self.rgb_conv(rgb)
        rgb_features = torch.flatten(rgb_features, start_dim=1)

        # Pass DSM through CNN
        dsm_features = self.dsm_conv(dsm)
        dsm_features = torch.flatten(dsm_features, start_dim=1)

        # Concatenate both
        combined_features = torch.cat((rgb_features, dsm_features), dim=1)

        # Fully connected layers
        output = self.fc(combined_features)

        return output