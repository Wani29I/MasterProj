import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

class EfficientNetV2MWheatModel(nn.Module):
    def __init__(self):
        super(EfficientNetV2MWheatModel, self).__init__()

        # ✅ Pretrained EfficientNetV2-M for RGB
        self.rgb_model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove classification head

        # ✅ CNN for DSM input (enhanced)
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),  # makes output [B, 64, 1, 1]
            nn.Flatten()                  # → [B, 64]
        )

        # ✅ Dynamically compute fusion feature size
        self.feature_size = self._get_feature_size()

        # ✅ Final regression head (enhanced)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1)  # Final output
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feat = self.rgb_model.features(dummy_rgb)
            rgb_feat = nn.functional.adaptive_avg_pool2d(rgb_feat, 1)
            rgb_feat = torch.flatten(rgb_feat, 1)  # [B, 1280]

            dsm_feat = self.dsm_conv(dummy_dsm)   # [B, 64]
            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model.features(rgb)
        rgb_feat = nn.functional.adaptive_avg_pool2d(rgb_feat, 1)
        rgb_feat = torch.flatten(rgb_feat, 1)

        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        output = self.fc(combined)
        return output
