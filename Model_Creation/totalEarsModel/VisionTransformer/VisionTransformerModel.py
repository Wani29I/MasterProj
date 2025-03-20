import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTModel

class ViTWheatModel(nn.Module):
    def __init__(self):
        super(ViTWheatModel, self).__init__()

        # ✅ Load Pretrained ViT for RGB
        self.rgb_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.rgb_fc = nn.Linear(768, 512)  # ViT outputs (Batch, 768), reduce to 512

        # ✅ Resize transformation for both RGB & DSM images
        self.resize_transform = transforms.Resize((224, 224))

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

        # ✅ Dynamically calculate DSM feature size
        self.dsm_feature_size = self._get_dsm_feature_size()
        self.dsm_fc = nn.Linear(self.dsm_feature_size, 512)  # ✅ Adjusted DSM feature size

        # ✅ Calculate Input Size for Fully Connected Layer
        self.feature_size = 512 + 512  # RGB features (512) + DSM features (512)

        # ✅ Fully Connected Layer for Final Prediction
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: Wheat Ear Count
        )

    def _get_dsm_feature_size(self):
        """Run a dummy forward pass to calculate DSM feature size dynamically."""
        with torch.no_grad():
            dsm_dummy = torch.randn(1, 1, 224, 224)  # ✅ Resized DSM to 224x224
            dsm_features = self.dsm_conv(dsm_dummy)
            return dsm_features.shape[1]  # ✅ Get flattened feature size

    def forward(self, rgb, dsm):
        # ✅ Resize both inputs to 224x224
        rgb = self.resize_transform(rgb)
        dsm = self.resize_transform(dsm)

        # ✅ Process RGB through ViT
        rgb_features = self.rgb_model(rgb).last_hidden_state[:, 0, :]
        rgb_features = self.rgb_fc(rgb_features)

        # ✅ Process DSM through CNN
        dsm_features = self.dsm_conv(dsm)
        dsm_features = self.dsm_fc(dsm_features)  # ✅ Ensure DSM features match expected size

        # ✅ Combine features and pass through final FC layer
        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)
        return output
