import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetWheatModel(nn.Module):
    def __init__(self):
        super(EfficientNetWheatModel, self).__init__()

        # ✅ Load Pretrained EfficientNet-B0 for RGB
        self.rgb_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_model.classifier = nn.Identity()  # Remove final classification layer

        # ✅ Small CNN for DSM Processing
        self.dsm_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # ✅ Calculate Input Size for Fully Connected Layer
        self.feature_size = self._get_feature_size()

        # ✅ Fully Connected Layer for Final Prediction
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: Wheat Ear Count
        )

    def _get_feature_size(self):
        """Run a dummy forward pass to calculate feature size."""
        with torch.no_grad():
            rgb_dummy = torch.randn(1, 3, 256, 512)  # Example RGB input
            dsm_dummy = torch.randn(1, 1, 256, 512)  # Example DSM input

            rgb_features = self.rgb_model(rgb_dummy)  # EfficientNet output
            dsm_features = self.dsm_conv(dsm_dummy)  # CNN output
            dsm_features = torch.flatten(dsm_features, start_dim=1)  # Ensure flattening

            combined = torch.cat((rgb_features, dsm_features), dim=1)
            return combined.shape[1]  # Get feature size dynamically

    def forward(self, rgb, dsm):
        rgb_features = self.rgb_model(rgb)  # Extract features from EfficientNet
        dsm_features = self.dsm_conv(dsm)   # Extract features from CNN
        dsm_features = torch.flatten(dsm_features, start_dim=1)  # Flatten DSM features

        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)  # Pass through Fully Connected Layer
        return output
