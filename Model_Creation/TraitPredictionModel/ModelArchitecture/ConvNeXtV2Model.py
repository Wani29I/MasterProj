import torch
import torch.nn as nn
import timm
from timm import create_model
import torch.nn.functional as F

class ConvNeXtV2WheatModelWithConfidence(nn.Module):
    def __init__(self):
        super(ConvNeXtV2WheatModelWithConfidence, self).__init__()

        # ✅ Load pretrained ConvNeXt-Tiny
        self.rgb_model = create_model("convnextv2_base", pretrained=True, features_only=True)
        
        # ✅ CNN for DSM input
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

        # ✅ Regression + confidence output
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [mean_prediction, log_variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)

            rgb_feats = self.rgb_model(dummy_rgb)[-1]  # Last stage
            rgb_feats = F.adaptive_avg_pool2d(rgb_feats, 1)
            rgb_feats = torch.flatten(rgb_feats, 1)

            dsm_feats = self.dsm_conv(dummy_dsm)
            return torch.cat((rgb_feats, dsm_feats), dim=1).shape[1]

    def forward(self, rgb, dsm):
        rgb_feats = self.rgb_model(rgb)[-1]
        rgb_feats = F.adaptive_avg_pool2d(rgb_feats, 1)
        rgb_feats = torch.flatten(rgb_feats, 1)

        dsm_feats = self.dsm_conv(dsm)
        dsm_feats = torch.flatten(dsm_feats, 1)

        combined = torch.cat((rgb_feats, dsm_feats), dim=1)
        return self.fc(combined)  # Output: [B, 2]


class ConvNeXtV2ConfidenceAddoneextrainput(nn.Module):
    def __init__(self):
        super(ConvNeXtV2ConfidenceAddoneextrainput, self).__init__()

        self.rgb_model = timm.create_model('convnextv2_base', pretrained=True, num_classes=0, global_pool='avg')

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.feature_size = self._get_feature_size()

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
            rgb_feat = self.rgb_model(torch.randn(1, 3, 256, 512))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, 256, 512))
            dsm_feat = torch.flatten(dsm_feat, 1)
            extra_feat = self.trait_fc(torch.randn(1, 1))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
    
class ConvNeXtV2ConfidenceAddtwoextrainput(nn.Module):
    def __init__(self):
        super(ConvNeXtV2ConfidenceAddtwoextrainput, self).__init__()

        # RGB Backbone
        self.rgb_model = timm.create_model('convnextv2_base', pretrained=True, num_classes=0, global_pool='avg')

        # DSM Backbone
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Trait input (2 values)
        self.trait_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # Total feature size
        self.feature_size = self._get_feature_size()

        # Final regression head
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
            rgb_feat = self.rgb_model(torch.randn(1, 3, 256, 512))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, 256, 512))
            dsm_feat = torch.flatten(dsm_feat, 1)
            extra_feat = self.trait_fc(torch.randn(1, 2))  # now 2 inputs
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)
        extra_feat = self.trait_fc(extra_input.view(-1, 2))  # view as 2D input
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
