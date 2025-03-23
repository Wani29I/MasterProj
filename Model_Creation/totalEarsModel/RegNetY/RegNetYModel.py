import torch
import torch.nn as nn
from torchvision.models import regnet_y_8gf, RegNet_Y_8GF_Weights

class RegNetYWheatModel(nn.Module):
    def __init__(self):
        super(RegNetYWheatModel, self).__init__()

        # ✅ Pretrained RegNetY-8GF for RGB
        self.rgb_model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.IMAGENET1K_V1)
        self.rgb_model.fc = nn.Identity()  # Remove classification head

        # ✅ Small CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # ✅ Compute combined feature size
        self.feature_size = self._get_feature_size()

        # ✅ Fully connected head for regression
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: single regression value
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feat = self.rgb_model(dummy_rgb)           # [1, 3712]
            dsm_feat = self.dsm_conv(dummy_dsm)            # [1, N]

            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model(rgb)          # [B, 3712]
        dsm_feat = self.dsm_conv(dsm)           # [B, N]
        dsm_feat = torch.flatten(dsm_feat, 1)

        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        output = self.fc(combined)
        return output
