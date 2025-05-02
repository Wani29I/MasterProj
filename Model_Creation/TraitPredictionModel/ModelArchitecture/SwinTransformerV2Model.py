import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinV2ConfidenceModel(nn.Module):
    def __init__(self, resize_shape=(256, 256)):
        super().__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model('swinv2_small_window8_256', pretrained=True, num_classes=0, global_pool='avg')
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
            rgb_feat = self.rgb_model(torch.randn(1, 3, *self.resize_shape))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, *self.resize_shape)).flatten(1)
            return rgb_feat.shape[1] + dsm_feat.shape[1]

    def forward(self, rgb, dsm):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        return self.fc(torch.cat((rgb_feat, dsm_feat), dim=1))

class SwinV2ConfidenceAddOneExtraInput(nn.Module):
    def __init__(self, resize_shape=(256, 256)):
        super().__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model('swinv2_small_window8_256', pretrained=True, num_classes=0, global_pool='avg')
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
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
            rgb_feat = self.rgb_model(torch.randn(1, 3, *self.resize_shape))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, *self.resize_shape)).flatten(1)
            trait_feat = self.trait_fc(torch.randn(1, 1))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + trait_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        return self.fc(torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1))

class SwinV2ConfidenceAddTwoExtraInput(nn.Module):
    def __init__(self, resize_shape=(256, 256)):
        super().__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model('swinv2_small_window8_256', pretrained=True, num_classes=0, global_pool='avg')
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.trait_fc = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.feature_size = self._get_feature_size()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb_feat = self.rgb_model(torch.randn(1, 3, *self.resize_shape))
            dsm_feat = self.dsm_conv(torch.randn(1, 1, *self.resize_shape)).flatten(1)
            trait_feat = self.trait_fc(torch.randn(1, 2))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + trait_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input)
        return self.fc(torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1))
