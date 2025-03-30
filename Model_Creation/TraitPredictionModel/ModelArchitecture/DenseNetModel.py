import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNet121WheatModel(nn.Module):
    def __init__(self):
        super(DenseNet121WheatModel, self).__init__()

        self.rgb_model = densenet121(pretrained=True)
        self.rgb_model.classifier = nn.Identity()

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.feature_size = self._get_feature_size()

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
            rgb_feat = self.rgb_model.features(dummy_rgb)
            rgb_feat = torch.nn.functional.adaptive_avg_pool2d(rgb_feat, (1, 1))
            rgb_feat = torch.flatten(rgb_feat, 1)
            dsm_feat = self.dsm_conv(dummy_dsm)
            return torch.cat((rgb_feat, dsm_feat), dim=1).shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model.features(rgb)
        rgb_feat = torch.nn.functional.adaptive_avg_pool2d(rgb_feat, (1, 1))
        rgb_feat = torch.flatten(rgb_feat, 1)
        dsm_feat = self.dsm_conv(dsm)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)
