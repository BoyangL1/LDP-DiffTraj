import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# ==========================================================
# 0. Paths & Params
# ==========================================================
FEATURE_FILE = "../data_privacy/trajectory_features.npy"
REAL_TRAJ_FILE = "../data_privacy/traj.npy"
GEN_DIR = "./result" 
OUT_CSV = "./result/ssim_occupancy_summary.csv"

GRID_SIZE = 200
SIGMA = 1.5

# ==========================================================
# 1. Load real trajectories (privacy only)
# ==========================================================
print("Loading real trajectories...")

features = np.load(FEATURE_FILE)
privacy_budget = features[:, -1]
priv_idx = np.where(privacy_budget > 1e-8)[0]

real_traj = np.load(REAL_TRAJ_FILE)
real_traj = real_traj[priv_idx]

# Global coordinate range (fixed!)
all_x = real_traj[:, :, 0].reshape(-1)
all_y = real_traj[:, :, 1].reshape(-1)

X_MIN, X_MAX = all_x.min(), all_x.max()
Y_MIN, Y_MAX = all_y.min(), all_y.max()

print(f"Real trajectory count: {len(real_traj)}")

# ==========================================================
# 2. Trajectory → occupancy heatmap (ALL visited points)
# ==========================================================
def generate_occupancy_heatmap(traj_array, grid_size, sigma=1.5):
    """
    traj_array: (N, T, 2)
    return: (grid_size, grid_size) density heatmap of visited points
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    points = traj_array.reshape(-1, 2)

    norm_x = (points[:, 0] - X_MIN) / (X_MAX - X_MIN + 1e-12)
    norm_y = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN + 1e-12)

    gx = np.clip((norm_x * grid_size).astype(int), 0, grid_size - 1)
    gy = np.clip((norm_y * grid_size).astype(int), 0, grid_size - 1)

    np.add.at(heatmap, (gy, gx), 1)

    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap

# ==========================================================
# 3. Compute Occupancy SSIM
# ==========================================================
def compute_occupancy_ssim(real_traj, gen_traj):
    img_real = generate_occupancy_heatmap(real_traj, GRID_SIZE, SIGMA)
    img_gen  = generate_occupancy_heatmap(gen_traj,  GRID_SIZE, SIGMA)

    # Normalize to [0, 1]
    img_real = img_real / (img_real.max() + 1e-12)
    img_gen  = img_gen  / (img_gen.max()  + 1e-12)

    return ssim(img_real, img_gen, data_range=1.0)

# ==========================================================
# 4. Iterate all generated trajectories
# ==========================================================
print("\nScanning generated trajectories...")

pattern = re.compile(r"Gen_traj_noise_([0-9.]+)\.pkl")
records = []

files = sorted(os.listdir(GEN_DIR))

for fname in tqdm(files):
    m = pattern.match(fname)
    if m is None:
        continue

    noise = float(m.group(1))
    path = os.path.join(GEN_DIR, fname)

    with open(path, "rb") as f:
        gen_traj = np.array(pickle.load(f))

    # Safety: match trajectory count
    n = min(len(real_traj), len(gen_traj))
    real_sub = real_traj[:n]
    gen_sub = gen_traj[:n]

    occ_ssim = compute_occupancy_ssim(real_sub, gen_sub)

    records.append({
        "noise": noise,
        "occupancy_ssim": occ_ssim
    })

# ==========================================================
# 5. Save results
# ==========================================================
df = pd.DataFrame(records).sort_values("noise").reset_index(drop=True)

print("\n========== Occupancy SSIM Summary ==========")
print(df)

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")