import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtWheatModel(nn.Module):
    def __init__(self):
        super(ConvNeXtWheatModel, self).__init__()

        # ✅ Pretrained ConvNeXt-Tiny for RGB (3-channel)
        self.rgb_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove final classification head

        # ✅ Simple CNN for DSM (1-channel)
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # ✅ Calculate combined feature size dynamically
        self.feature_size = self._get_feature_size()

        # ✅ Fully Connected Layers for Regression
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: Wheat Ear Count
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            # Extract ConvNeXt RGB features
            rgb_feat = self.rgb_model.features(dummy_rgb)      # [1, 768, H, W]
            rgb_feat = self.rgb_model.avgpool(rgb_feat)        # [1, 768, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)              # [1, 768]

            # Extract DSM features
            dsm_feat = self.dsm_conv(dummy_dsm)                # [1, N]

            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]


    def forward(self, rgb, dsm):
        # RGB → ConvNeXt
        rgb_feat = self.rgb_model.features(rgb)
        rgb_feat = self.rgb_model.avgpool(rgb_feat)
        rgb_feat = torch.flatten(rgb_feat, 1)

        # DSM → small CNN
        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        output = self.fc(combined)
        return output

