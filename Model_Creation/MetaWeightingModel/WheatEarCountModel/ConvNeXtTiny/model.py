import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F

class ConvNeXtTinyWheatModelWithConfidence(nn.Module):
    def __init__(self):
        super(ConvNeXtTinyWheatModelWithConfidence, self).__init__()

        # ✅ Load pretrained ConvNeXt-Tiny
        self.rgb_model = create_model("convnext_tiny", pretrained=True, features_only=True)
        
        # ✅ CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # ✅ Compute combined feature size
        self.feature_size = self._get_feature_size()

        # ✅ Regression + confidence output
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [mean_prediction, log_variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feats = self.rgb_model(dummy_rgb)[-1]  # Last stage
            rgb_feats = F.adaptive_avg_pool2d(rgb_feats, 1)
            rgb_feats = torch.flatten(rgb_feats, 1)

            dsm_feats = self.dsm_conv(dummy_dsm)
            return torch.cat((rgb_feats, dsm_feats), dim=1).shape[1]

    def forward(self, rgb, dsm):
        rgb_feats = self.rgb_model(rgb)[-1]
        rgb_feats = F.adaptive_avg_pool2d(rgb_feats, 1)
        rgb_feats = torch.flatten(rgb_feats, 1)

        dsm_feats = self.dsm_conv(dsm)
        dsm_feats = torch.flatten(dsm_feats, 1)

        combined = torch.cat((rgb_feats, dsm_feats), dim=1)
        return self.fc(combined)  # Output: [B, 2]
