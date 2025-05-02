import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PoolFormerConfidenceModel(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(PoolFormerConfidenceModel, self).__init__()
        self.resize_shape = resize_shape  # (height, width)

        self.rgb_model = timm.create_model(
            'poolformer_s36',  # Smallest version
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

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
            rgb = F.interpolate(torch.randn(1, 3, 256, 512), size=self.resize_shape, mode='bilinear', align_corners=False)
            dsm = F.interpolate(torch.randn(1, 1, 256, 512), size=self.resize_shape, mode='bilinear', align_corners=False)
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


class PoolFormerConfidenceAddOneExtraInput(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(PoolFormerConfidenceAddOneExtraInput, self).__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model('poolformer_s36', pretrained=True, num_classes=0, global_pool='avg')

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.extra_fc = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU()
        )

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb = F.interpolate(torch.randn(1, 3, 256, 512), size=self.resize_shape)
            dsm = F.interpolate(torch.randn(1, 1, 256, 512), size=self.resize_shape)
            extra = self.extra_fc(torch.randn(1, 1))
            return self.rgb_model(rgb).shape[1] + self.dsm_conv(dsm).flatten(1).shape[1] + extra.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape)
        dsm = F.interpolate(dsm, size=self.resize_shape)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.extra_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)


class PoolFormerConfidenceAddTwoExtraInput(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(PoolFormerConfidenceAddTwoExtraInput, self).__init__()
        self.resize_shape = resize_shape

        self.rgb_model = timm.create_model('poolformer_s36', pretrained=True, num_classes=0, global_pool='avg')

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.extra_fc = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU()
        )

        self.feature_size = self._get_feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _get_feature_size(self):
        with torch.no_grad():
            rgb = F.interpolate(torch.randn(1, 3, 256, 512), size=self.resize_shape)
            dsm = F.interpolate(torch.randn(1, 1, 256, 512), size=self.resize_shape)
            extra = self.extra_fc(torch.randn(1, 2))
            return self.rgb_model(rgb).shape[1] + self.dsm_conv(dsm).flatten(1).shape[1] + extra.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape)
        dsm = F.interpolate(dsm, size=self.resize_shape)
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.extra_fc(extra_input)
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
