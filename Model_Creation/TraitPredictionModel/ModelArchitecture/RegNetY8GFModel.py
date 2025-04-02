import torch
import torch.nn as nn
from torchvision.models import regnet_y_8gf, RegNet_Y_8GF_Weights

class RegNetY8GFModel(nn.Module):
    def __init__(self):
        super(RegNetY8GFModel, self).__init__()

        # ✅ Load pretrained RegNetY-8GF
        self.rgb_model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.IMAGENET1K_V1)
        self.rgb_model.fc = nn.Identity()  # Remove classifier head

        # ✅ Lightweight DSM CNN
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # ✅ Compute feature size
        self.feature_size = self._get_feature_size()

        # ✅ Regression + log-variance output
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [prediction, log_variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feat = self.rgb_model(dummy_rgb)  # [1, 2016]
            dsm_feat = self.dsm_conv(dummy_dsm)   # [1, N]
            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)
