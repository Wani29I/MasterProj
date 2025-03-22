import torch
import torch.nn as nn
import timm  # Ensure timm is installed

class ConvNeXtV2WheatModel(nn.Module):
    def __init__(self):
        super(ConvNeXtV2WheatModel, self).__init__()

        # ✅ Load ConvNeXtV2 and remove classification head
        self.rgb_model = timm.create_model("convnextv2_base.fcmae_ft_in1k", pretrained=True, features_only=False)
        self.rgb_model.head = nn.Identity()
        self.rgb_feature_dim = 1024

        # ✅ Small CNN for DSM
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256x512 → 128x256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x256 → 64x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x128 → 32x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → [B, 128, 1, 1]
            nn.Flatten()  # → [B, 128]
        )

        # ✅ Calculate DSM feature size and align it to RGB
        with torch.no_grad():
            dummy_dsm = torch.randn(1, 1, 256, 512)
            dsm_features = self.dsm_conv(dummy_dsm)
            dsm_output_size = dsm_features.shape[1]

        self.dsm_fc = nn.Linear(dsm_output_size, self.rgb_feature_dim)  # → [B, 1024]

        # ✅ Final FC for regression
        self.fc = nn.Sequential(
            nn.Linear(self.rgb_feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, rgb, dsm):
        rgb_features = self.rgb_model.forward_features(rgb)  # [B, 1024, H, W]
        rgb_features = nn.functional.adaptive_avg_pool2d(rgb_features, (1, 1))  # [B, 1024, 1, 1]
        rgb_features = torch.flatten(rgb_features, start_dim=1)  # [B, 1024]

        dsm_features = self.dsm_conv(dsm)  # [B, 128]
        dsm_features = self.dsm_fc(dsm_features)  # [B, 1024]

        combined = torch.cat((rgb_features, dsm_features), dim=1)  # [B, 2048]

        return self.fc(combined)
