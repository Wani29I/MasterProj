import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MaxViTConfidenceModel(nn.Module):
    def __init__(self, resize_shape=(512, 512)):
        super(MaxViTConfidenceModel, self).__init__()
        self.resize_shape = resize_shape  # Must be divisible by 16

        self.rgb_model = timm.create_model(
            'maxvit_small_tf_512.in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            # Use correct resize_shape to prevent MaxViT assertion
            rgb = torch.randn(1, 3, *self.resize_shape)
            dsm = torch.randn(1, 1, *self.resize_shape)
            rgb_feat = self.rgb_model(rgb)
            dsm_feat = self.dsm_conv(dsm).flatten(1)
            return rgb_feat.shape[1] + dsm_feat.shape[1]

    def forward(self, rgb, dsm):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)


class MaxViTConfidenceAddoneextrainput(nn.Module):
    def __init__(self, resize_shape=(512, 512)):
        super(MaxViTConfidenceAddoneextrainput, self).__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model(
            'maxvit_small_tf_512.in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.feature_size = self._get_feature_size()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb = torch.randn(1, 3, *self.resize_shape)
            dsm = torch.randn(1, 1, *self.resize_shape)
            extra = torch.randn(1, 1)
            rgb_feat = self.rgb_model(rgb)
            dsm_feat = self.dsm_conv(dsm).flatten(1)
            extra_feat = self.trait_fc(extra)
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)

class MaxViTConfidenceAddtwoextrainput(nn.Module):
    def __init__(self, resize_shape=(512, 512)):
        super(MaxViTConfidenceAddtwoextrainput, self).__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model(
            'maxvit_small_tf_512.in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )

        self.feature_size = self._get_feature_size()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb = torch.randn(1, 3, *self.resize_shape)
            dsm = torch.randn(1, 1, *self.resize_shape)
            extra = torch.randn(1, 2)
            rgb_feat = self.rgb_model(rgb)
            dsm_feat = self.dsm_conv(dsm).flatten(1)
            extra_feat = self.trait_fc(extra)
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input)
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
