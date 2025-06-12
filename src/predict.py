"""
Prediction script to load the trained model and artifacts for inference.
"""
import pickle
from datetime import datetime

import numpy as np
import torch

import config as cfg
from data_processing import (clean_text, extract_domains,
                                 load_glove_embeddings, title_to_embedding)
from model import CombinedScorePredictor

class Scorer:
    def __init__(self, artifacts_dir=cfg.ARTIFACTS_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load artifacts
        print("--- Loading artifacts for prediction ---")
        self.word_to_idx, self.embeddings = load_glove_embeddings(cfg.GLOVE_FILE)
        
        with open(cfg.DOMAIN_ENCODER_PATH, 'rb') as f:
            self.domain_encoder = pickle.load(f)
        with open(cfg.USER_ENCODER_PATH, 'rb') as f:
            self.user_encoder = pickle.load(f)
        with open(cfg.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        n_domains = len(self.domain_encoder.classes_)
        n_users = len(self.user_encoder.classes_)
        
        self.model = CombinedScorePredictor(
            n_domains=n_domains, n_users=n_users,
            domain_emb_dim=cfg.DOMAIN_EMB_DIM, user_emb_dim=cfg.USER_EMB_DIM,
            title_emb_dim=cfg.TITLE_EMB_DIM, numerical_dim=cfg.NUMERICAL_DIM,
            hidden_dim=cfg.HIDDEN_DIM, dropout=cfg.DROPOUT_RATE
        ).to(self.device)
        self.model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=self.device))
        self.model.eval()
        print("✅ Artifacts and model loaded successfully.")

    def predict(self, title: str, url: str, user: str, submission_time: datetime):
        """Predicts the score for a new story."""
        
        # 1. Title features
        title_emb = title_to_embedding(title, self.word_to_idx, self.embeddings)
        word_count = len(title.strip().split())

        # 2. Time features
        hour = submission_time.hour
        day = submission_time.weekday() # Monday is 0 and Sunday is 6
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        # 3. Domain features
        domain = extract_domains(url)
        domain_mapped = domain if domain in self.domain_encoder.classes_ else 'OTHER'
        domain_id = self.domain_encoder.transform([domain_mapped])[0]

        # 4. User features
        user_mapped = user if user in self.user_encoder.classes_ else 'OTHER'
        user_id = self.user_encoder.transform([user_mapped])[0]
        
        # 5. Combine and scale numerical features
        numerical_features = np.array([[word_count, time_sin, time_cos, day_sin, day_cos]])
        numerical_scaled = self.scaler.transform(numerical_features)
        
        # 6. Create tensors
        title_tensor = torch.FloatTensor(title_emb).unsqueeze(0).to(self.device)
        numerical_tensor = torch.FloatTensor(numerical_scaled).to(self.device)
        domain_tensor = torch.LongTensor([domain_id]).to(self.device)
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        
        # 7. Predict
        with torch.no_grad():
            pred_log = self.model(title_tensor, numerical_tensor, domain_tensor, user_tensor)
            pred_orig = np.expm1(pred_log.cpu().item())
        
        return max(1, int(round(pred_orig)))

# Singleton instance to be used by the API
scorer = Scorer()

def predict_score(title: str, url: str, user: str, timestamp: int):
    """Convenience function to use the singleton scorer."""
    submission_time = datetime.fromtimestamp(timestamp)
    return scorer.predict(title, url, user, submission_time)

if __name__ == '__main__':
    # Example usage:
    print("\n--- Example Prediction ---")
    test_title = "Show HN: A new tool to predict Hacker News scores"
    test_url = "https://github.com/someuser/newtool"
    test_user = "someuser" # A user not in the top list
    test_timestamp = int(datetime.now().timestamp())
    
    predicted_score = predict_score(test_title, test_url, test_user, test_timestamp)
    print(f"Title: '{test_title}'")
    print(f"Predicted Score: {predicted_score}") 