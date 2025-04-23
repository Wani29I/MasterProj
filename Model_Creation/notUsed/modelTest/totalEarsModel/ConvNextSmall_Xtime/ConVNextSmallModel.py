import torch
import torch.nn as nn
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

class ConvNeXtSmallWheatModel(nn.Module):
    def __init__(self):
        super(ConvNeXtSmallWheatModel, self).__init__()

        # ✅ Load pretrained ConvNeXt-Small for RGB
        self.rgb_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove final classifier

        # ✅ Small CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # ✅ Compute feature size for combined input
        self.feature_size = self._get_feature_size()

        # ✅ Fully connected regression head
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

            rgb_feat = self.rgb_model.features(dummy_rgb)  # [B, 768, H, W]
            rgb_feat = self.rgb_model.avgpool(rgb_feat)    # [B, 768, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)          # [B, 768]

            dsm_feat = self.dsm_conv(dummy_dsm)            # [B, N]

            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model.features(rgb)
        rgb_feat = self.rgb_model.avgpool(rgb_feat)
        rgb_feat = torch.flatten(rgb_feat, 1)

        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        output = self.fc(combined)
        return output
