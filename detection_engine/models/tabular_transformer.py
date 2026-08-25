import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    """
    Transformer-based model for tabular network intrusion detection.

    Input:
        [batch_size, 147]

    Output:
        [batch_size, 2]
        Class 0 -> Normal
        Class 1 -> Attack
    """

    def __init__(
        self,
        num_features=147,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Convert each scalar feature into an embedding vector.
        self.feature_embedding = nn.Linear(1, embedding_dim)

        # Learnable feature-position embeddings.
        self.feature_position = nn.Parameter(
            torch.randn(1, num_features, embedding_dim)
        )

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embedding_dim)

        # Classification head.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x shape:
            [batch_size, num_features]
        """

        # [B, 147] -> [B, 147, 1]
        x = x.unsqueeze(-1)

        # [B, 147, 1] -> [B, 147, embedding_dim]
        x = self.feature_embedding(x)

        # Add feature-specific positional information.
        x = x + self.feature_position

        # Transformer encoder.
        x = self.transformer_encoder(x)

        # Aggregate feature representations.
        x = x.mean(dim=1)

        # Normalize.
        x = self.norm(x)

        # Classification.
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":

    print("=" * 70)
    print("NIDS TABULAR TRANSFORMER")
    print("=" * 70)

    # Select GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model.
    model = TabularTransformer(
        num_features=147,
        embedding_dim=64,
        num_heads=4,
        num_layers=3,
        feedforward_dim=128,
        dropout=0.1,
        num_classes=2
    ).to(device)

    print("\nModel created successfully.")

    # Test with a fake batch.
    test_input = torch.randn(
        8,
        147,
        device=device
    )

    with torch.no_grad():
        output = model(test_input)

    print(f"\nInput shape  : {test_input.shape}")
    print(f"Output shape : {output.shape}")

    print("\nExpected output shape: [8, 2]")

    # Parameter count.
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")

    print("\nTransformer model test completed successfully.")