"""
Main training script for the HackerNews Score Prediction model.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split

import src.config as cfg
from src.data_processing import create_data_loader, prepare_features
from src.model import CombinedScorePredictor


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch_idx, (title_emb, num, dom_ids, usr_ids, y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(title_emb, num, dom_ids, usr_ids)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            wandb.log({"batch_loss": loss.item()})
    avg_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    return avg_loss

def evaluate_model(model, data_loader, criterion, prefix="val", epoch=None):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for title_emb, num, dom_ids, usr_ids, y in data_loader:
            outputs = model(title_emb, num, dom_ids, usr_ids)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    # R² score on the original scale
    from sklearn.metrics import r2_score
    pred_orig = np.expm1(predictions)
    targ_orig = np.expm1(targets)
    r2 = r2_score(targ_orig, pred_orig)

    log_data = {f"{prefix}_loss": avg_loss, f"{prefix}_r2": r2}
    if epoch is not None:
        log_data["epoch"] = epoch
    wandb.log(log_data)
    
    return avg_loss, r2

def main():
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- W&B Initialization ---
    
    # Create a dictionary of config parameters to pass to wandb
    wandb_config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}

    wandb.init(
        project="hackernews-score-prediction",
        name="structured-run-v1",
        config=wandb_config
    )
    config = wandb.config

    # --- Data Preparation ---
    data = prepare_features(config)

    # --- Train/Val/Test Split ---
    indices = np.arange(len(data["y"]))
    train_val_idx, test_idx = train_test_split(indices, test_size=config.VAL_SIZE, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=config.TEST_SIZE, random_state=42)
    
    train_loader = create_data_loader(data, train_idx, config.BATCH_SIZE)
    val_loader = create_data_loader(data, val_idx, config.BATCH_SIZE, shuffle=False)
    test_loader = create_data_loader(data, test_idx, config.BATCH_SIZE, shuffle=False)

    print(f"Split sizes: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}")

    # --- Model Initialization ---
    model = CombinedScorePredictor(
        n_domains=data["n_domains"],
        n_users=data["n_users"],
        domain_emb_dim=config.DOMAIN_EMB_DIM,
        user_emb_dim=config.USER_EMB_DIM,
        title_emb_dim=config.TITLE_EMB_DIM,
        numerical_dim=config.NUMERICAL_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=config.FACTOR)
    wandb.watch(model, log="all", log_freq=200)

    # --- Training Loop ---
    best_val_r2 = -float('inf')
    patience_counter = 0
    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_r2 = evaluate_model(model, val_loader, criterion, "val", epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            os.makedirs(cfg.ARTIFACTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), cfg.MODEL_PATH)
            print(f"  🎯 New best model saved with R²: {best_val_r2:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

    # --- Final Evaluation and Artifact Saving ---
    print("\n--- Training Finished. Evaluating on test set... ---")
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    test_loss, test_r2 = evaluate_model(model, test_loader, criterion, "test")
    print(f"🧪 Test Results: Loss={test_loss:.4f}, R²={test_r2:.4f}")
    
    # Save artifacts
    with open(cfg.DOMAIN_ENCODER_PATH, 'wb') as f:
        pickle.dump(data["domain_encoder"], f)
    with open(cfg.USER_ENCODER_PATH, 'wb') as f:
        pickle.dump(data["user_encoder"], f)
    with open(cfg.SCALER_PATH, 'wb') as f:
        pickle.dump(data["scaler"], f)
    print(f"✅ Artifacts saved to '{cfg.ARTIFACTS_DIR}'")
    
    wandb.finish()

if __name__ == '__main__':
    main() 