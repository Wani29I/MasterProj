import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaWeightingModel(nn.Module):
    def __init__(self, num_traits=4, d_model=16, num_heads=4, dropout=0.3):
        super(MetaWeightingModel, self).__init__()
        self.num_traits = num_traits
        self.d_model = d_model

        # 1. Trait Input Layer
        self.trait_embedding = nn.Linear(2, d_model)  # [pred, std] → d_model
        self.input_norm = nn.LayerNorm(d_model)

        # 2. Transformer Attention Blocks (2-layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Flatten after transformer
        self.flatten = nn.Flatten()

        # 3. MLP Weighting Layer
        self.mlp = nn.Sequential(
            nn.Linear(num_traits * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 4. Output + Calibration
        self.output_head = nn.Linear(32, 2)  # → [prediction, log_variance]
        self.confidence_activation = nn.Softplus()  # Optional: ensures positive std

        # Optional learnable temperature scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Input shape: [B, num_traits * 2] → reshape to [B, T, 2]
        B = x.size(0)
        x = x.view(B, self.num_traits, 2)

        # Trait Embedding + Norm
        x = self.trait_embedding(x)
        x = self.input_norm(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Flatten
        x = self.flatten(x)  # [B, T * d_model]

        # MLP Fusion
        x = self.mlp(x)  # [B, 32]

        # Output prediction + log variance
        out = self.output_head(x)  # [B, 2]
        mean = out[:, 0:1]
        log_var = out[:, 1:2] * self.temperature
        var = self.confidence_activation(log_var)
        log_var = torch.log(var + 1e-6)

        return torch.cat([mean, log_var], dim=1)  # [B, 2]


# Example usage:
# model = MetaWeightingModel(num_traits=6, d_model=16)  # if using 6 traits
