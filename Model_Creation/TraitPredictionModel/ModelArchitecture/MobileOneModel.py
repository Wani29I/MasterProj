import torch
import torch.nn as nn
import timm
from torch.nn import functional as F

class MobileOneConfidence(nn.Module):
    def __init__(self):
        super(MobileOneConfidence, self).__init__()
        self.rgb_model = timm.create_model("mobileone_s2", pretrained=True, num_classes=0, global_pool="avg")

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)  # [prediction, log_variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb_feat = self.rgb_model(torch.randn(1, 3, 256, 512))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, 256, 512)).flatten(1)
            return rgb_feat.shape[1] + dsm_feat.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)

class MobileOneConfidenceAddoneextrainput(nn.Module):
    def __init__(self):
        super(MobileOneConfidenceAddoneextrainput, self).__init__()
        self.rgb_model = timm.create_model("mobileone_s2", pretrained=True, num_classes=0, global_pool="avg")

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(nn.Linear(1, 16), nn.ReLU())

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb_feat = self.rgb_model(torch.randn(1, 3, 256, 512))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, 256, 512)).flatten(1)
            extra_feat = self.trait_fc(torch.randn(1, 1))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)

class MobileOneConfidenceAddtwoextrainput(nn.Module):
    def __init__(self):
        super(MobileOneConfidenceAddtwoextrainput, self).__init__()
        self.rgb_model = timm.create_model("mobileone_s2", pretrained=True, num_classes=0, global_pool="avg")

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb_feat = self.rgb_model(torch.randn(1, 3, 256, 512))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, 256, 512)).flatten(1)
            extra_feat = self.trait_fc(torch.randn(1, 2))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 2))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
