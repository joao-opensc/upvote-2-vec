"""
PyTorch model definition for the CombinedScorePredictor.
"""
import torch
import torch.nn as nn

class CombinedScorePredictor(nn.Module):
    def __init__(self, n_domains, n_users, domain_emb_dim=32, user_emb_dim=64,
                 title_emb_dim=200, numerical_dim=5, hidden_dim=256, dropout=0.3):
        super(CombinedScorePredictor, self).__init__()

        # Learnable embeddings
        self.domain_embedding = nn.Embedding(n_domains, domain_emb_dim)
        self.user_embedding = nn.Embedding(n_users, user_emb_dim)

        # Calculate total input dimension
        total_input_dim = title_emb_dim + numerical_dim + domain_emb_dim + user_emb_dim

        # Main neural network
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, title_emb, numerical_features, domain_ids, user_ids):
        # Get embeddings
        domain_emb = self.domain_embedding(domain_ids)
        user_emb = self.user_embedding(user_ids)

        # Concatenate all features
        combined = torch.cat([
            title_emb,
            numerical_features,
            domain_emb,
            user_emb
        ], dim=1)

        # Forward pass
        return self.model(combined).squeeze(1) 