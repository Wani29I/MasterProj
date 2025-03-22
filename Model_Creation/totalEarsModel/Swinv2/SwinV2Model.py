import torch
import torch.nn as nn
import timm

class SwinV2WheatModel(nn.Module):
    def __init__(self):
        super(SwinV2WheatModel, self).__init__()

        # ✅ Load SwinV2 with custom image size (512x256)
        self.rgb_model = timm.create_model(
            "swinv2_tiny_window16_256",
            pretrained=True,
            img_size=(256, 512),  # 👈 Important: match your image size
            features_only=False
        )
        self.rgb_model.head = nn.Identity()
        self.rgb_feature_dim = self.rgb_model.num_features  # 768

        # ✅ DSM branch: Simple CNN
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # → [B, 64]
        )
        self.dsm_fc = nn.Linear(64, self.rgb_feature_dim)

        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)  # Your actual input size
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feat = self.rgb_model.forward_features(dummy_rgb)
            rgb_feat = torch.flatten(rgb_feat, start_dim=1)  # Flatten to [1, ?]

            dsm_feat = self.dsm_conv(dummy_dsm)
            dsm_feat = self.dsm_fc(dsm_feat)

            combined_dim = rgb_feat.shape[1] + dsm_feat.shape[1]


        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, rgb, dsm):
        rgb_features = self.rgb_model.forward_features(rgb)  # → [B, C, H, W]
        rgb_features = torch.flatten(rgb_features, start_dim=1)  # → [B, C*H*W]

        dsm_features = self.dsm_conv(dsm)   # → [B, 64]
        dsm_features = self.dsm_fc(dsm_features)  # → [B, same as rgb_features shape]

        # 💡 Optional sanity print
        # print(rgb_features.shape, dsm_features.shape)

        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)
        return output

