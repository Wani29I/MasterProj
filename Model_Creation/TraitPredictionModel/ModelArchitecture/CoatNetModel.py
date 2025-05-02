import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CoAtNetConfidenceModel(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(CoAtNetConfidenceModel, self).__init__()
        self.resize_shape = resize_shape  # Resize to (224, 224) for coatnet_2_rw_224

        # RGB Backbone (CoAtNet)
        self.rgb_model = timm.create_model(
            'coatnet_2_rw_224',
            pretrained=False,  # No pretrained weights available
            num_classes=0,
            global_pool='avg'
        )

        # DSM feature extractor
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Compute feature size from RGB + DSM
        self.feature_size = self._get_feature_size()

        # Regression head with confidence output
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)  # Output: [mean, log_variance]
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy_rgb = F.interpolate(torch.randn(1, 3, 300, 500), size=self.resize_shape, mode='bilinear', align_corners=False)
            dummy_dsm = F.interpolate(torch.randn(1, 1, 300, 500), size=self.resize_shape, mode='bilinear', align_corners=False)
            rgb_feat = self.rgb_model(dummy_rgb)
            dsm_feat = self.dsm_conv(dummy_dsm).flatten(1)
            return rgb_feat.shape[1] + dsm_feat.shape[1]

    def forward(self, rgb, dsm):
        # Resize both inputs
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear', align_corners=False)
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear', align_corners=False)

        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        combined = torch.cat((rgb_feat, dsm_feat), dim=1)
        return self.fc(combined)

class CoAtNetConfidenceAddOneExtraInput(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(CoAtNetConfidenceAddOneExtraInput, self).__init__()
        self.resize_shape = resize_shape
        self.rgb_model = timm.create_model('coatnet_2_rw_224', pretrained=False, num_classes=0, global_pool='avg')

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(
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
            rgb = F.interpolate(torch.randn(1, 3, 300, 500), size=self.resize_shape, mode='bilinear')
            dsm = F.interpolate(torch.randn(1, 1, 300, 500), size=self.resize_shape, mode='bilinear')
            rgb_feat = self.rgb_model(rgb)
            dsm_feat = self.dsm_conv(dsm).flatten(1)
            extra_feat = self.trait_fc(torch.randn(1, 1))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear')
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear')
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 1))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)

class CoAtNetConfidenceAddTwoExtraInput(nn.Module):
    def __init__(self, resize_shape=(224, 224)):
        super(CoAtNetConfidenceAddTwoExtraInput, self).__init__()
        self.resize_shape = resize_shape
        self.rgb_model = timm.create_model('coatnet_2_rw_224', pretrained=False, num_classes=0, global_pool='avg')

        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.trait_fc = nn.Sequential(
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
            rgb = F.interpolate(torch.randn(1, 3, 300, 500), size=self.resize_shape, mode='bilinear')
            dsm = F.interpolate(torch.randn(1, 1, 300, 500), size=self.resize_shape, mode='bilinear')
            rgb_feat = self.rgb_model(rgb)
            dsm_feat = self.dsm_conv(dsm).flatten(1)
            extra_feat = self.trait_fc(torch.randn(1, 2))
            return rgb_feat.shape[1] + dsm_feat.shape[1] + extra_feat.shape[1]

    def forward(self, rgb, dsm, extra_input):
        rgb = F.interpolate(rgb, size=self.resize_shape, mode='bilinear')
        dsm = F.interpolate(dsm, size=self.resize_shape, mode='bilinear')
        rgb_feat = self.rgb_model(rgb)
        dsm_feat = self.dsm_conv(dsm).flatten(1)
        extra_feat = self.trait_fc(extra_input.view(-1, 2))
        combined = torch.cat((rgb_feat, dsm_feat, extra_feat), dim=1)
        return self.fc(combined)
