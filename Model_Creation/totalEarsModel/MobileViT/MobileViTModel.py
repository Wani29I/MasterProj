import torch
import torch.nn as nn
from timm import create_model

class MobileViTWheatModel(nn.Module):
    def __init__(self):
        super(MobileViTWheatModel, self).__init__()

        # ✅ Load pretrained MobileViT-S
        self.rgb_model = create_model('mobilevit_s', pretrained=True)
        self.rgb_model.head = nn.Identity()  # Remove classifier head

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

        # ✅ Compute combined feature size
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

            rgb_feat = self.rgb_model.forward_features(dummy_rgb)       # [1, 640, H, W]
            rgb_feat = nn.functional.adaptive_avg_pool2d(rgb_feat, 1)   # [1, 640, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)                       # [1, 640]

            dsm_feat = self.dsm_conv(dummy_dsm)                         # [1, N]

            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]


    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model.forward_features(rgb)                # [B, 640, H, W]
        rgb_feat = nn.functional.adaptive_avg_pool2d(rgb_feat, 1)      # [B, 640, 1, 1]
        rgb_feat = torch.flatten(rgb_feat, 1)                          # [B, 640]

        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        output = self.fc(combined)
        return output

