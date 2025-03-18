import torch
import torch.nn as nn
from transformers import ViTModel

class ViTWheatModel(nn.Module):
    def __init__(self):
        super(ViTWheatModel, self).__init__()

        # ✅ Load Pretrained ViT for RGB
        self.rgb_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.rgb_fc = nn.Linear(768, 512)  # ViT outputs (Batch, 768), reduce to 512

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
        """Run a dummy forward pass to calculate feature size dynamically."""
        with torch.no_grad():
            rgb_dummy = torch.randn(1, 3, 224, 224)  # ViT requires 224x224 input
            dsm_dummy = torch.randn(1, 1, 256, 512)

            rgb_features = self.rgb_model(rgb_dummy).last_hidden_state[:, 0, :]  # Extract CLS token features
            rgb_features = self.rgb_fc(rgb_features)

            dsm_features = self.dsm_conv(dsm_dummy)
            dsm_features = torch.flatten(dsm_features, start_dim=1)

            combined = torch.cat((rgb_features, dsm_features), dim=1)
            return combined.shape[1]

    def forward(self, rgb, dsm):
        rgb_features = self.rgb_model(rgb).last_hidden_state[:, 0, :]
        rgb_features = self.rgb_fc(rgb_features)

        dsm_features = self.dsm_conv(dsm)
        dsm_features = torch.flatten(dsm_features, start_dim=1)

        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)
        return output
