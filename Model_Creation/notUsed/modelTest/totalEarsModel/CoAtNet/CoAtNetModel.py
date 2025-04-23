import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm  # ✅ CoAtNet is available in `timm`

class CoAtNetWheatModel(nn.Module):
    def __init__(self):
        super(CoAtNetWheatModel, self).__init__()

        # ✅ Load Pretrained CoAtNet for RGB
        self.rgb_model = timm.create_model("coatnet_0_rw_224.sw_in1k", pretrained=True)
        self.rgb_model.head = nn.Identity()  # Remove classification head

        # ✅ Adaptive Pooling to reduce CoAtNet features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.rgb_fc = nn.Linear(768, 1024)  # ✅ Match RGB to DSM feature size

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

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # ✅ Global Pooling to fix feature size mismatch

            nn.Flatten()
        )

        # ✅ Dynamically calculate DSM feature size
        self.dsm_feature_size = self._get_dsm_feature_size()
        self.dsm_fc = nn.Linear(self.dsm_feature_size, 1024)  # ✅ Align DSM to RGB

        # ✅ Final feature size
        self.feature_size = 1024 + 1024  # RGB + DSM features

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

        # ✅ Extract features from CoAtNet
        rgb_features = self.rgb_model(rgb)  # CoAtNet output is (batch, 768, 7, 7)
        rgb_features = self.rgb_pool(rgb_features)  # Adaptive Pool to (batch, 768, 1, 1)
        rgb_features = torch.flatten(rgb_features, start_dim=1)  # Convert to (batch, 768)
        rgb_features = self.rgb_fc(rgb_features)  # Convert to (batch, 1024)

        # ✅ Process DSM through CNN
        dsm_features = self.dsm_conv(dsm)
        dsm_features = self.dsm_fc(dsm_features)  # Reduce DSM feature size to 1024

        # ✅ Ensure shapes match before concatenation
        assert rgb_features.shape[1] == 1024, f"RGB feature mismatch: {rgb_features.shape}"
        assert dsm_features.shape[1] == 1024, f"DSM feature mismatch: {dsm_features.shape}"

        # ✅ Concatenate Features
        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)
        return output
