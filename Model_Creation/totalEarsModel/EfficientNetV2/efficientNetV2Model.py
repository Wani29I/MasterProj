import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2SWheatModel(nn.Module):
    def __init__(self):
        super(EfficientNetV2SWheatModel, self).__init__()

        # ✅ Load pretrained EfficientNetV2-S for RGB
        self.rgb_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove the final classification head

        # ✅ Simple CNN for DSM input (1-channel)
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # ✅ Dynamically compute feature vector size
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

            rgb_feat = self.rgb_model.features(dummy_rgb)  # [B, 1280, H, W]
            rgb_feat = self.rgb_model.avgpool(rgb_feat)     # [B, 1280, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)           # [B, 1280]

            dsm_feat = self.dsm_conv(dummy_dsm)             # [B, N]

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
