import torch
import torch.nn as nn
import timm  # ✅ CoAtNet is available in `timm`

class CoAtNetWheatModel(nn.Module):
    def __init__(self):
        super(CoAtNetWheatModel, self).__init__()

        # ✅ Load Pretrained CoAtNet for RGB
        self.rgb_model = timm.create_model("coatnet_0_rw_224.sw_in1k", pretrained=True)
        self.rgb_model.head = nn.Identity()  # Remove final classification layer

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
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        # ✅ Reduce DSM Feature Size to Match RGB
        self.dsm_fc = nn.Linear(65536, 37632)  # Reduce DSM features to match RGB

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
            rgb_dummy = torch.randn(1, 3, 224, 224)  # CoAtNet requires 224x224 input
            dsm_dummy = torch.randn(1, 1, 256, 512)

            rgb_features = self.rgb_model(rgb_dummy)
            rgb_features = torch.flatten(rgb_features, start_dim=1)  # Flatten CoAtNet features

            dsm_features = self.dsm_conv(dsm_dummy)
            dsm_features = torch.flatten(dsm_features, start_dim=1)
            dsm_features = self.dsm_fc(dsm_features)  # Reduce DSM feature size

            # ✅ Print feature shapes BEFORE concatenation
            print("✅ RGB Features Shape (Dummy, After Flattening):", rgb_features.shape)  
            print("✅ DSM Features Shape (Dummy):", dsm_features.shape)  

            combined = torch.cat((rgb_features, dsm_features), dim=1)  # Concatenate features
            return combined.shape[1]  # Get feature size

    def forward(self, rgb, dsm):
        # ✅ Extract features from CoAtNet
        rgb_features = self.rgb_model(rgb)

        # ✅ Flatten RGB Features if Needed
        rgb_features = torch.flatten(rgb_features, start_dim=1)  # Convert to (Batch, Features)

        # ✅ Process DSM Features
        dsm_features = self.dsm_conv(dsm)
        dsm_features = torch.flatten(dsm_features, start_dim=1)
        dsm_features = self.dsm_fc(dsm_features)  # Reduce DSM feature size

        # ✅ Print feature shapes for debugging
        print("✅ RGB Features Shape (After Flattening):", rgb_features.shape)  
        print("✅ DSM Features Shape:", dsm_features.shape)  

        # ✅ Ensure Shapes Match Before Concatenation
        assert rgb_features.shape == dsm_features.shape, f"Shape mismatch: RGB {rgb_features.shape}, DSM {dsm_features.shape}"

        # ✅ Concatenate Features
        combined = torch.cat((rgb_features, dsm_features), dim=1)
        output = self.fc(combined)
        return output


