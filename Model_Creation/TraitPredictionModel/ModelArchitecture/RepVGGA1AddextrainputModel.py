import torch
import torch.nn as nn
from timm import create_model

class RepVGGA1_Confidence_Addextrainput(nn.Module):
    def __init__(self):
        super(RepVGGA1_Confidence_Addextrainput, self).__init__()

        # ✅ Pretrained RepVGG-A1 for RGB input
        self.rgb_model = create_model("repvgg_a1", pretrained=True)
        self.rgb_model.head = nn.Identity()  # Remove classification head

        # ✅ CNN for DSM input
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # ✅ FC for extra trait input (e.g., ear weight)
        self.trait_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        # ✅ Compute feature vector size
        self.feature_size = self._get_feature_size()

        # ✅ Final regression + confidence head
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
            dummy_extra = torch.randn(1, 1)

            rgb_feat = self.rgb_model.forward_features(dummy_rgb)
            rgb_feat = torch.flatten(rgb_feat, 1)  # ✅

            dsm_feat = self.dsm_conv(dummy_dsm)
            dsm_feat = torch.flatten(dsm_feat, 1)

            extra_feat = self.trait_fc(dummy_extra)

            combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
            return combined.shape[1]


    def forward(self, rgb, dsm, extra_input):
        rgb_feat = self.rgb_model.forward_features(rgb)
        rgb_feat = torch.flatten(rgb_feat, 1)  # ✅

        dsm_feat = self.dsm_conv(dsm)
        dsm_feat = torch.flatten(dsm_feat, 1)

        extra_feat = self.trait_fc(extra_input.view(-1, 1))

        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)

