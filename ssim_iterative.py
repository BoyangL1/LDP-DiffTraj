import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# ==========================================================
# 0. Paths
# ==========================================================
FEATURE_FILE = "../data_privacy/trajectory_features.npy"
REAL_TRAJ_FILE = "../data_privacy/traj.npy"
GEN_DIR = "./result" 
OUT_CSV = "./result/ssim_summary.csv"

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

# global coordinate range (fixed!)
all_x = real_traj[:, :, 0].reshape(-1)
all_y = real_traj[:, :, 1].reshape(-1)

X_MIN, X_MAX = all_x.min(), all_x.max()
Y_MIN, Y_MAX = all_y.min(), all_y.max()

print(f"Real traj count: {len(real_traj)}")

# ==========================================================
# 2. Trajectory → density heatmap
# ==========================================================
def generate_density_heatmap(traj_array, grid_size, mode="start", sigma=1.5):
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    if mode == "start":
        points = traj_array[:, 0, :]
    elif mode == "end":
        points = traj_array[:, -1, :]
    else:
        raise ValueError("mode must be 'start' or 'end'")

    norm_x = (points[:, 0] - X_MIN) / (X_MAX - X_MIN + 1e-12)
    norm_y = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN + 1e-12)

    gx = (norm_x * grid_size).astype(int)
    gy = (norm_y * grid_size).astype(int)

    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)

    np.add.at(heatmap, (gy, gx), 1)

    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    return heatmap

# ==========================================================
# 3. Compute SSIM
# ==========================================================
def compute_ssim(real_traj, gen_traj, mode):
    img_real = generate_density_heatmap(real_traj, GRID_SIZE, mode, SIGMA)
    img_gen = generate_density_heatmap(gen_traj, GRID_SIZE, mode, SIGMA)

    img_real = img_real / (img_real.max() + 1e-12)
    img_gen = img_gen / (img_gen.max() + 1e-12)

    return ssim(img_real, img_gen, data_range=1.0)

# ==========================================================
# 4. Iterate all generated results
# ==========================================================
print("\nScanning generated trajectories...")

pattern = re.compile(r"Gen_traj_noise_([0-9.]+)\.pkl")
records = []

files = sorted(os.listdir(GEN_DIR))

for fname in tqdm(files):
    match = pattern.match(fname)
    if match is None:
        continue

    noise = float(match.group(1))
    path = os.path.join(GEN_DIR, fname)

    with open(path, "rb") as f:
        gen_traj = np.array(pickle.load(f))

    # safety: match length
    n = min(len(real_traj), len(gen_traj))
    gen_traj = gen_traj[:n]
    real_sub = real_traj[:n]

    ssim_start = compute_ssim(real_sub, gen_traj, "start")
    ssim_end = compute_ssim(real_sub, gen_traj, "end")

    records.append({
        "noise": noise,
        "ssim_start": ssim_start,
        "ssim_end": ssim_end,
        "ssim_mean": 0.5 * (ssim_start + ssim_end)
    })

# ==========================================================
# 5. Output table
# ==========================================================
df = pd.DataFrame(records).sort_values("noise").reset_index(drop=True)

print("\n========== SSIM Summary ==========")
print(df)

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved SSIM table to: {OUT_CSV}")