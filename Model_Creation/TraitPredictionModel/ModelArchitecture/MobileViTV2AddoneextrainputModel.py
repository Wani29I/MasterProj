import torch
import torch.nn as nn
import timm

class MobileViTV2ConfidenceAddoneextrainput(nn.Module):
    def __init__(self):
        super(MobileViTV2ConfidenceAddoneextrainput, self).__init__()

        # RGB Backbone
        self.rgb_model = timm.create_model('mobilevitv2_100', pretrained=True, num_classes=0, global_pool='avg')

        # DSM Backbone
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Extra Input (Tabular Trait Value)
        self.trait_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        # Calculate Feature Size
        self.feature_size = self._get_feature_size()

        # Final Regression Head
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
            extra_feat = self.trait_fc(torch.randn(1, 1))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)


class MobileViTV2ConfidenceAddtwoextrainput(nn.Module):
    def __init__(self):
        super(MobileViTV2ConfidenceAddtwoextrainput, self).__init__()

        # RGB Backbone
        self.rgb_model = timm.create_model('mobilevitv2_100', pretrained=True, num_classes=0, global_pool='avg')

        # DSM Backbone
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Extra Inputs (2 tabular trait values)
        self.trait_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        # Feature size calculation
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
            extra_feat = self.trait_fc(torch.randn(1, 2))  # Now 2 inputs
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)
        extra_feat = self.trait_fc(extra_input.view(-1, 2))  # Accept 2D input
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)