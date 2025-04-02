import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2S_Confidence_Addoneextrainput(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ EfficientNetV2-S for RGB
        self.rgb_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()

        # ✅ CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # ✅ Linear layer for 1D trait input 
        self.trait_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        # ✅ Compute combined feature size
        self.feature_size = self._get_feature_size()

        # ✅ Final FC layers (output: [mean, log_variance])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)
            dummy_trait = torch.randn(1, 1)

            rgb_feat = self.rgb_model.features(dummy_rgb)
            rgb_feat = self.rgb_model.avgpool(rgb_feat)
            rgb_feat = torch.flatten(rgb_feat, 1)

            dsm_feat = self.dsm_conv(dummy_dsm)
            trait_feat = self.trait_fc(dummy_trait)

            combined = torch.cat((rgb_feat, dsm_feat, trait_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm, ear_weight):
        # ✅ EfficientNetV2S RGB features
        rgb_feat = self.rgb_model.features(rgb)            # [B, C, H, W]
        rgb_feat = self.rgb_model.avgpool(rgb_feat)        # [B, C, 1, 1]
        rgb_feat = torch.flatten(rgb_feat, 1)              # [B, C]

        # ✅ DSM features
        dsm_feat = self.dsm_conv(dsm)                      # [B, C, H, W] or [B, C]
        if dsm_feat.dim() > 2:
            dsm_feat = torch.flatten(dsm_feat, 1)          # Ensure [B, N]

        # ✅ Ear weight input
        ear_weight = ear_weight.view(-1, 1)                # [B] or [B, 1]
        trait_feat = self.trait_fc(ear_weight)             # [B, 16]

        # ✅ Combine all features
        combined = torch.cat((rgb_feat, dsm_feat, trait_feat), dim=1)  # [B, M]

        return self.fc(combined)  # [B, 2] → [mean, logvar]

class EfficientNetV2S_Confidence_Add2Inputs(nn.Module):
    def __init__(self):
        super(EfficientNetV2S_Confidence_Add2Inputs, self).__init__()

        # ✅ Pretrained EfficientNetV2-S for RGB
        self.rgb_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()

        # ✅ Simple CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # ✅ MLP for 2 extra inputs (e.g. earWeight + time)
        self.extra_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        # ✅ Dynamically compute feature size
        self.feature_size = self._get_feature_size()

        # ✅ Final prediction + confidence head
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
            dummy_extra = torch.randn(1, 2)

            rgb_feat = self.rgb_model.features(dummy_rgb)
            rgb_feat = self.rgb_model.avgpool(rgb_feat)
            rgb_feat = torch.flatten(rgb_feat, 1)

            dsm_feat = self.dsm_conv(dummy_dsm)
            extra_feat = self.extra_fc(dummy_extra)

            combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model.features(rgb)
        rgb_feat = self.rgb_model.avgpool(rgb_feat)
        rgb_feat = torch.flatten(rgb_feat, 1)

        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        extra_feat = self.extra_fc(extra_input)  # [B, 2] → [B, 16]

        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
