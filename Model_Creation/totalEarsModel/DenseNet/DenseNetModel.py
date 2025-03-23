import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121WheatModel(nn.Module):
    def __init__(self):
        super(DenseNet121WheatModel, self).__init__()

        # ✅ Load pretrained DenseNet-121 for RGB input
        self.rgb_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove classifier head

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

        # ✅ Calculate combined feature size
        self.feature_size = self._get_feature_size()

        # ✅ Fully connected head
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

            rgb_feat = self.rgb_model.features(dummy_rgb)           # [1, 1024, H, W]
            rgb_feat = nn.functional.adaptive_avg_pool2d(rgb_feat, 1)  # [1, 1024, 1, 1]
            rgb_feat = torch.flatten(rgb_feat, 1)                   # [1, 1024]

            dsm_feat = self.dsm_conv(dummy_dsm)                     # [1, N]

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
