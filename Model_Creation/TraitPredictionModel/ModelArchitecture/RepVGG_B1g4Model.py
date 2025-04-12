import torch
import torch.nn as nn
from timm import create_model

class RepVGGB1g4Model(nn.Module):
    def __init__(self):
        super(RepVGGB1g4Model, self).__init__()

        # ✅ Load pretrained RepVGG-B1g4 from timm
        self.rgb_model = create_model("repvgg_b1g4", pretrained=True, num_classes=0, global_pool="avg")

        # ✅ Lightweight CNN for DSM
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # ✅ Compute total feature size
        self.feature_size = self._get_feature_size()

        # ✅ Regression + confidence head
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

            rgb_feat = self.rgb_model(dummy_rgb)             # [1, 2048]
            dsm_feat = self.dsm_conv(dummy_dsm)              # [1, N]

            combined = torch.cat((rgb_feat, dsm_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)
