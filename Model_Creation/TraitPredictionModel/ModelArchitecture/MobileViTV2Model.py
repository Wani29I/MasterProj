import torch
import torch.nn as nn
import timm  # Huggingface PyTorch Image Models

class MobileViTV2WheatModel(nn.Module):
    def __init__(self):
        super(MobileViTV2WheatModel, self).__init__()

        # Load MobileViT-V2 backbone
        self.rgb_model = timm.create_model('mobilevitv2_100', pretrained=True, num_classes=0, global_pool='avg')

        # DSM feature extractor
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # Combine RGB + DSM feature size
        self.feature_size = self._get_feature_size()

        # Final Regression Head
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # [Prediction, Log Variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = torch.randn(1, 3, 256, 512)
            dummy_dsm = torch.randn(1, 1, 256, 512)
            rgb_feat = self.rgb_model(dummy_rgb)
            dsm_feat = self.dsm_conv(dummy_dsm)
            return rgb_feat.shape[1] + dsm_feat.shape[1]

    def forward(self, rgb, dsm):
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)
