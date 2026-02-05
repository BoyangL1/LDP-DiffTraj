import os
import numpy as np
import pandas as pd
from scipy.special import rel_entr

# ==========================================================
# 0. Params & Paths
# ==========================================================
FEATURE_FILE = "../data_privacy/trajectory_features.npy"
NOISE_SWEEP_DIR = "../data_privacy/noise_sweep"
GRID_PRIVACY_CSV = "./grid_privacy_scores.csv"

GRID_SIZE = 200
TOP_K_GRIDS = 10000
TOP_N_PATTERN = 100

# ==========================================================
# 1. Load privacy index from trajectory_features.npy
# ==========================================================
print("Loading privacy features and selecting privacy trajectories...")

features = np.load(FEATURE_FILE)
privacy_budget = features[:, -1]
priv_idx = np.where(privacy_budget > 1e-8)[0]

print(f"Privacy trajectories selected: {len(priv_idx)} / {len(features)}")

# ==========================================================
# 2. Load baseline (noise=0.00) trajectories and apply priv_idx
# ==========================================================
print("Loading baseline trajectories (noise=0.00)...")

BASE_DIR = os.path.join(NOISE_SWEEP_DIR, "noise_0.00")
real_traj_all = np.load(os.path.join(BASE_DIR, "traj.npy"))

# 安全对齐
max_n = min(len(real_traj_all), len(features))
priv_idx_safe = priv_idx[priv_idx < max_n]

real_traj = real_traj_all[priv_idx_safe]

print(f"Baseline (privacy-only) trajectory count: {len(real_traj)}")

# ==========================================================
# 3. Compute global coordinate bounds (SINGLE SOURCE)
# ==========================================================
all_x = real_traj[:, :, 0].reshape(-1)
all_y = real_traj[:, :, 1].reshape(-1)

X_MIN, X_MAX = all_x.min(), all_x.max()
Y_MIN, Y_MAX = all_y.min(), all_y.max()

print("Global grid bounds:")
print(f"X: [{X_MIN:.3f}, {X_MAX:.3f}]")
print(f"Y: [{Y_MIN:.3f}, {Y_MAX:.3f}]")

# ==========================================================
# 4. Load & process privacy grid scores
# ==========================================================
privacy_df = pd.read_csv(GRID_PRIVACY_CSV)

def xy_to_linear(gid):
    x, y = gid.split("_")
    x = int(x)
    y = int(y)

    # 1-based -> 0-based
    x -= 1
    y -= 1

    assert 0 <= x < GRID_SIZE
    assert 0 <= y < GRID_SIZE

    return y * GRID_SIZE + x

privacy_df["grid_linear_id"] = privacy_df["grid_id"].apply(xy_to_linear)
privacy_df = privacy_df.sort_values(
    "privacy_lb95", ascending=False
).reset_index(drop=True)

top_priv_grids = privacy_df.head(TOP_K_GRIDS)["grid_linear_id"].values
top_priv_set = set(top_priv_grids)

print(f"Loaded Top-{TOP_K_GRIDS} privacy grids.")

# ==========================================================
# 5. Utils
# ==========================================================
def jsd(p, q, eps=1e-12):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= (p.sum() + eps)
    q /= (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def coord_to_grid(points):
    nx = (points[:, 0] - X_MIN) / (X_MAX - X_MIN + 1e-12)
    ny = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN + 1e-12)
    gx = np.clip((nx * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    gy = np.clip((ny * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    return gx, gy


def traj_to_grid_ids(traj):
    pts = traj.reshape(-1, 2)
    gx, gy = coord_to_grid(pts)
    return gy * GRID_SIZE + gx

# ==========================================================
# 6. Privacy-only Density JSD
# ==========================================================
def compute_privacy_density_jsd(real, noisy):
    idx_map = {g: i for i, g in enumerate(top_priv_set)}

    def density(traj):
        gids = traj_to_grid_ids(traj)
        gids = gids[np.isin(gids, list(top_priv_set))]
        vec = np.zeros(len(top_priv_set))
        u, c = np.unique(gids, return_counts=True)
        for gi, ci in zip(u, c):
            vec[idx_map[gi]] = ci
        return vec

    return jsd(density(real), density(noisy))

# ==========================================================
# 7. Privacy-only OD JSD (start / end)
# ==========================================================
def compute_privacy_od_jsd(real, noisy):
    idx_map = {g: i for i, g in enumerate(top_priv_set)}

    def endpoint(traj, t):
        pts = traj[:, t, :]
        gx, gy = coord_to_grid(pts)
        gids = gy * GRID_SIZE + gx
        gids = gids[np.isin(gids, list(top_priv_set))]
        vec = np.zeros(len(top_priv_set))
        u, c = np.unique(gids, return_counts=True)
        for gi, ci in zip(u, c):
            vec[idx_map[gi]] = ci
        return vec

    return 0.5 * (
        jsd(endpoint(real, 0), endpoint(noisy, 0)) +
        jsd(endpoint(real, -1), endpoint(noisy, -1))
    )

# ==========================================================
# 8. Privacy-only Pattern F1
# ==========================================================
def extract_privacy_patterns(traj):
    gids = traj_to_grid_ids(traj)
    gids = gids[np.isin(gids, list(top_priv_set))]
    u, c = np.unique(gids, return_counts=True)
    return set(u[np.argsort(-c)[:TOP_N_PATTERN]])


def compute_privacy_pattern_f1(real, noisy):
    P = extract_privacy_patterns(real)
    Q = extract_privacy_patterns(noisy)

    tp = len(P & Q)
    p = tp / (len(Q) + 1e-12)
    r = tp / (len(P) + 1e-12)
    return 2 * p * r / (p + r + 1e-12)

# ==========================================================
# 9. Sanity check
# ==========================================================
gids = traj_to_grid_ids(real_traj)
coverage = np.mean(np.isin(gids, list(top_priv_set)))
print(f"Coverage of top-privacy grids on real traj: {coverage:.4f}")

# ==========================================================
# 10. Evaluate generated trajectories (from result/)
# ==========================================================
import re
import pickle

GEN_DIR = "./result"

pattern = re.compile(r"Gen_traj_noise_([0-9.]+)\.pkl")
records = []

print("\nEvaluating generated trajectories from result/ ...")

for fname in sorted(os.listdir(GEN_DIR)):
    m = pattern.match(fname)
    if m is None:
        continue

    noise = float(m.group(1))

    # ---------- load generated trajectories ----------
    with open(os.path.join(GEN_DIR, fname), "rb") as f:
        gen_traj_all = np.array(pickle.load(f))

    # ---------- apply same privacy index ----------
    n = min(len(real_traj), len(gen_traj_all))

    real_use = real_traj[:n]
    gen_traj = gen_traj_all[:n]

    # ---------- compute privacy metrics ----------
    records.append({
        "noise": noise,
        "privacy_density_jsd": compute_privacy_density_jsd(real_use, gen_traj),
        "privacy_od_jsd": compute_privacy_od_jsd(real_use, gen_traj),
        "privacy_pattern_f1": compute_privacy_pattern_f1(real_use, gen_traj),
    })

df_privacy = pd.DataFrame(records).sort_values("noise").reset_index(drop=True)

print("\n========== Privacy Metrics on Generated Trajectories ==========")
