"""
Main training script for the HackerNews Score Prediction model.
"""
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as cfg
from data_processing import create_data_loader, prepare_features
from model import CombinedScorePredictor


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch_idx, (title_emb, num, dom_ids, usr_ids, y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(title_emb, num, dom_ids, usr_ids)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    return avg_loss, r2, predictions, targets

def train():
    """Main function to run the training pipeline."""
    # --- W&B Initialization ---
    default_config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    wandb.init(
        project="hackernews-score-sweeps-final",
        config=default_config
    )
    config = wandb.config
    # Set a default run name if not set by sweep, which usually provides one
    if wandb.run.name == wandb.run.id:
        wandb.run.name = "manual-single-run"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    data = prepare_features(config)

    # --- Train/Val/Test Split ---
    indices = np.arange(len(data["y"]))
    train_val_idx, test_idx = train_test_split(indices, test_size=config.VAL_SIZE, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=config.TEST_SIZE, random_state=42)
    
    # --- Feature Scaling (No Data Leakage) ---
    scaler = StandardScaler()
    
    # Fit on training data ONLY and transform it
    X_numerical_train_scaled = scaler.fit_transform(data['X_numerical'][train_idx])

    # Create a new array to hold all scaled data, ensuring alignment with original indices
    X_numerical_scaled = np.zeros_like(data['X_numerical'], dtype=np.float32)
    X_numerical_scaled[train_idx] = X_numerical_train_scaled
    X_numerical_scaled[val_idx] = scaler.transform(data['X_numerical'][val_idx])
    X_numerical_scaled[test_idx] = scaler.transform(data['X_numerical'][test_idx])
    
    # Replace unscaled data with the properly scaled version for the data loaders
    data['X_numerical_scaled'] = X_numerical_scaled
    del data['X_numerical']

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
        val_loss, val_r2, _, _ = evaluate_model(model, val_loader, criterion, "val", epoch)
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
    test_loss, test_r2, test_preds, test_targets = evaluate_model(model, test_loader, criterion, "test")
    print(f"🧪 Test Results: Loss={test_loss:.4f}, R²={test_r2:.4f}")
    
    # ── PREDICTED vs ACTUAL SCATTER ─────────────────────
    y_pred_orig = np.expm1(test_preds)
    y_true_orig = np.expm1(test_targets)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.1, s=5)
    plt.plot([y_true_orig.min(), y_true_orig.max()],
             [y_true_orig.min(), y_true_orig.max()],
             'r--', lw=2, label='Ideal Fit')
    plt.xscale('log');  plt.yscale('log')
    plt.xlabel('Actual Score');     plt.ylabel('Predicted Score')
    plt.title('Predicted vs. Actual (log-log scale)')
    plt.grid(True);      plt.legend();
    wandb.log({"predicted_vs_actual": wandb.Image(plt)})
    plt.close()

    # Save artifacts
    with open(cfg.DOMAIN_ENCODER_PATH, 'wb') as f:
        pickle.dump(data["domain_encoder"], f)
    with open(cfg.USER_ENCODER_PATH, 'wb') as f:
        pickle.dump(data["user_encoder"], f)
    with open(cfg.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Artifacts saved to '{cfg.ARTIFACTS_DIR}'")
    
    wandb.finish()

    # --- Cleanup ---
    # The data object holds a reference to the memory-mapped file.
    # Deleting it allows the file to be unmapped so we can remove the directory.
    print("🧹 Cleaning up temporary embedding files...")
    temp_dir = "temp_embeddings"
    if hasattr(data["X_title_embeddings"], 'filename') and os.path.dirname(data["X_title_embeddings"].filename) == temp_dir:
        del data
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("✅ Cleanup complete.")
    else:
        print("⏩ No temporary files to clean up or path mismatch.")

if __name__ == '__main__':
    train() 