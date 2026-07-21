from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import optim


# =======================
# Loss function
# =======================
def bernoulli_gamma_loss(p_raw, a_raw, b_raw, y_true, eps=1e-6):
    p_raw = p_raw.cpu()
    a_raw = a_raw.cpu()
    b_raw = b_raw.cpu()
    y_true = y_true.cpu()

    a_raw = torch.clamp(a_raw, max=10.0)
    b_raw = torch.clamp(b_raw, max=10.0)

    p = torch.sigmoid(p_raw)
    alpha = F.softplus(a_raw) + eps
    beta = F.softplus(b_raw) + eps

    p_true = (y_true > 0).float()

    # Log-likelihood terms
    no_rain = (1 - p_true) * torch.log(torch.clamp(1 - p, min=eps))
    rain = p_true * (
            torch.log(torch.clamp(p, min=eps))
            + (alpha - 1) * torch.log(torch.clamp(y_true, min=eps))
            - y_true / beta
            + alpha * torch.log(beta)
            - torch.lgamma(alpha)
    )
    loss = -(no_rain + rain)
    return loss.mean()


# =======================
# 1. Load and clean data
# =======================
package_root = Path(__file__).resolve().parents[1]
train_path = package_root / "data" / "station_hourly_training_set.csv"
test_path = package_root / "data" / "station_hourly_test_set.csv"
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

features = [
    'TAIR', 'DEWP', 'PAIR', 'QAIR', 'RHUS', 'UWIN', 'VWIN', 'WIND', 'PRE',
    'PoDu', 'PoXiang_x', 'PoXiang_y',
    'LandOver_10.0', 'LandOver_20.0', 'LandOver_52.0', 'LandOver_62.0',
    'LandOver_71.0', 'LandOver_72.0', 'LandOver_190.0', 'LandOver_210.0',
    'lon_coding', 'lat_coding', 'Gem', 'gpm', 'VO_850', 'VO_500', 'feat_sin', 'feat_cos', 'ejfi_3', 'ejfi_5', 'ejfi_8',
    'Jet_Emb_1', 'Jet_Emb_2', 'Jet_Emb_3', 'Jet_Emb_4'
]

# Extract features and labels.
X_train = train[features].copy()
X_test = test[features].copy()
y_train = train[['PRE_1h']].copy()
y_test = test[['PRE_1h']].copy()

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

# =======================
# 2. Normalize data
# =======================
print("Normalizing data...")

feature_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Normalize labels.
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# =======================
# 3. Convert to tensors
# =======================
device = 'cuda'

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

print(f"Training-set shape: {X_train_t.shape}")

# =======================
# 4. Initialize model
# =======================
input_dim = len(features)
group_attention_matrix = torch.eye(input_dim).to(device)

model = TabNet(
    input_dim=input_dim,
    output_dim=3,  # Outputs: p_raw, a_raw, b_raw
    n_d=64, n_a=64, n_steps=5, gamma=1.5,
    n_independent=2, n_shared=2,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=[],
    group_attention_matrix=group_attention_matrix
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================
# 5. Training loop
# =======================
epochs = 70
batch_size = 64
best_mae = float('inf')
best_model_path = output_dir / "paje_net_best_state.pth"

print(f"Device: {device}")
print("Starting training...")

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train_t.size(0))
    X_train_shuffled = X_train_t[perm]
    y_train_shuffled = y_train_t[perm]

    total_loss = 0.0
    num_batches = 0

    for i in range(0, X_train_t.size(0), batch_size):
        xb = X_train_shuffled[i:i + batch_size]
        yb = y_train_shuffled[i:i + batch_size]

        preds, _ = model(xb)
        p_raw, a_raw, b_raw = torch.split(preds, 1, dim=1)

        loss = bernoulli_gamma_loss(p_raw, a_raw, b_raw, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    model.eval()
    with torch.no_grad():
        preds_test, _ = model(X_test_t)
        p_raw, a_raw, b_raw = torch.split(preds_test, 1, dim=1)

        a_raw = torch.clamp(a_raw, max=10.0)
        b_raw = torch.clamp(b_raw, max=10.0)

        p = torch.sigmoid(p_raw)
        alpha = F.softplus(a_raw) + 1e-6
        beta = F.softplus(b_raw) + 1e-6

        # Expected value: E = p * alpha * beta
        y_pred_scaled_val = (p * alpha * beta).cpu().numpy().flatten()

        # Inverse normalization.
        y_pred = y_scaler.inverse_transform(y_pred_scaled_val.reshape(-1, 1)).flatten()
        y_pred[y_pred < 0] = 0

        mae = mean_absolute_error(test['PRE_1h'].values, y_pred)

    print(f"Epoch {epoch + 1:03d} | Loss={avg_loss:.4f} | MAE={mae:.5f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), best_model_path)
        print(f"--> Saved best model, MAE={mae:.5f}")

# =======================
# 6. Load best model and predict
# =======================
print("Training finished. Loading the best model for final prediction...")

best_model = TabNet(input_dim=input_dim, output_dim=3,
                    n_d=64, n_a=64, n_steps=5, gamma=1.5,
                    n_independent=2, n_shared=2,
                    cat_idxs=[], cat_dims=[], cat_emb_dim=[],
                    group_attention_matrix=group_attention_matrix)
best_model = best_model.to(device)
best_model.load_state_dict(torch.load(best_model_path, map_location=device))
best_model.eval()

with torch.no_grad():
    preds_test, _ = best_model(X_test_t)
    p_raw, a_raw, b_raw = torch.split(preds_test, 1, dim=1)

    p = torch.sigmoid(p_raw)
    alpha = F.softplus(a_raw) + 1e-6
    beta = F.softplus(b_raw) + 1e-6

    y_pred_scaled_final = (p * alpha * beta).cpu().numpy().flatten()
    y_pred_final = y_scaler.inverse_transform(y_pred_scaled_final.reshape(-1, 1)).flatten()
    y_pred_final[y_pred_final < 0] = 0

# =======================
# 7. Save results
# =======================
df_res = pd.DataFrame({
    'lon': test["lon"],
    'lat': test["lat"],
    'gt': test['PRE_1h'].values.flatten(),
    'PRE': test['PRE'],
    'pred': y_pred_final,
    'gpm': test['gpm']
})

output_path = output_dir / "paje_net_test_predictions.csv"
df_res.to_csv(output_path, index=False)

print(f"Completed. Results saved to: {output_path}")
