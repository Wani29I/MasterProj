import torchvision.models as models
import torch.nn as nn
import torch

class ResNetWheatModel(nn.Module):
    def __init__(self):
        super(ResNetWheatModel, self).__init__()

        # ✅ Load Pretrained ResNet50
        self.rgb_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.rgb_model.fc = nn.Identity()  # Remove classification layer

        # ✅ Small CNN for DSM Processing
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ✅ Ensure Flatten Is Defined Before `_get_feature_size()`
        self.flatten = nn.Flatten()  # 🔹 Make sure this is here!

        # ✅ Get Feature Size Automatically
        self.feature_size = self._get_feature_size()

        # ✅ Fully Connected Layer (Corrected Input Size)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_feature_size(self):
        """Run a dummy forward pass to calculate feature size."""
        with torch.no_grad():
            rgb_dummy = torch.randn(1, 3, 256, 512)  # Example input
            dsm_dummy = torch.randn(1, 1, 256, 512)
            
            rgb_features = self.rgb_model(rgb_dummy)
            dsm_features = self.dsm_conv(dsm_dummy)
            dsm_features = self.flatten(dsm_features)  # ✅ Flatten Here

            combined = torch.cat((rgb_features, dsm_features), dim=1)
            return combined.shape[1]  # Get feature dimension

    def forward(self, rgb, dsm):
        rgb_features = self.rgb_model(rgb)  # (Batch, 2048)
        dsm_features = self.dsm_conv(dsm)   # (Batch, 32, H, W)
        dsm_features = self.flatten(dsm_features)  # ✅ Now it exists!

        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)  # Fully Connected Layer
        return output
