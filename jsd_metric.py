import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.special import rel_entr
from skimage.metrics import structural_similarity as ssim

# ==========================================================
# 0. Paths & Params
# ==========================================================
FEATURE_FILE = "../data_privacy/trajectory_features.npy"
REAL_TRAJ_FILE = "../data_privacy/traj.npy"
GEN_DIR = "./result"              # Gen_traj_noise_*.pkl
OUT_CSV = "./result/metrics_summary.csv"

GRID_SIZE = 200
SIGMA = 1.5
TOP_N_PATTERN = 4000
LENGTH_BINS = 50

# ==========================================================
# 1. Load real trajectories (privacy only)
# ==========================================================
print("Loading real trajectories...")

features = np.load(FEATURE_FILE)
privacy_budget = features[:, -1]
priv_idx = np.where(privacy_budget > 1e-8)[0]

real_traj = np.load(REAL_TRAJ_FILE)
real_traj = real_traj[priv_idx]

# Global coordinate range (fixed)
all_x = real_traj[:, :, 0].reshape(-1)
all_y = real_traj[:, :, 1].reshape(-1)

X_MIN, X_MAX = all_x.min(), all_x.max()
Y_MIN, Y_MAX = all_y.min(), all_y.max()

print(f"Real trajectory count: {len(real_traj)}")

# ==========================================================
# 2. Utils
# ==========================================================
def jsd(p, q, eps=1e-12):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def coord_to_grid(points):
    nx = (points[:, 0] - X_MIN) / (X_MAX - X_MIN + 1e-12)
    ny = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN + 1e-12)
    gx = np.clip((nx * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    gy = np.clip((ny * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    return gx, gy

# ==========================================================
# 3. SSIM (OD)
# ==========================================================
def generate_density_heatmap(traj_array, mode="start"):
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    if mode == "start":
        points = traj_array[:, 0, :]
    elif mode == "end":
        points = traj_array[:, -1, :]
    else:
        raise ValueError("mode must be start or end")

    gx, gy = coord_to_grid(points)
    np.add.at(heatmap, (gy, gx), 1)

    if SIGMA > 0:
        heatmap = gaussian_filter(heatmap, sigma=SIGMA)

    heatmap /= (heatmap.max() + 1e-12)
    return heatmap


def compute_ssim(real_traj, gen_traj, mode):
    img_r = generate_density_heatmap(real_traj, mode)
    img_g = generate_density_heatmap(gen_traj, mode)
    return ssim(img_r, img_g, data_range=1.0)

# ==========================================================
# 4. Density Error (Global JSD)
# ==========================================================
def compute_density_jsd(real_traj, gen_traj):
    def density(traj):
        heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        points = traj.reshape(-1, 2)
        gx, gy = coord_to_grid(points)
        np.add.at(heat, (gy, gx), 1)
        return heat.flatten()

    return jsd(density(real_traj), density(gen_traj))

# ==========================================================
# 5. Trip Error (OD JSD)
# ==========================================================
def compute_trip_jsd(real_traj, gen_traj):
    def od_dist(traj):
        od = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
        start = traj[:, 0, :]
        end = traj[:, -1, :]

        sx, sy = coord_to_grid(start)
        ex, ey = coord_to_grid(end)

        for i in range(len(traj)):
            od[sy[i], sx[i], ey[i], ex[i]] += 1
        return od.flatten()

    return jsd(od_dist(real_traj), od_dist(gen_traj))

# ==========================================================
# 6. Length Error (JSD)
# ==========================================================
def trajectory_lengths(traj):
    lengths = []
    for t in traj:
        diff = np.diff(t, axis=0)
        lengths.append(np.linalg.norm(diff, axis=1).sum())
    return np.array(lengths)


def compute_length_jsd(real_traj, gen_traj):
    r = trajectory_lengths(real_traj)
    g = trajectory_lengths(gen_traj)
    hr, bins = np.histogram(r, bins=LENGTH_BINS, density=True)
    hg, _ = np.histogram(g, bins=bins, density=True)
    return jsd(hr, hg)

# ==========================================================
# 7. Pattern Score (Top-N grids)
# ==========================================================
def extract_patterns(traj):
    points = traj.reshape(-1, 2)
    gx, gy = coord_to_grid(points)
    idx = gy * GRID_SIZE + gx
    u, c = np.unique(idx, return_counts=True)
    return set(u[np.argsort(-c)[:TOP_N_PATTERN]])


def compute_pattern_score(real_traj, gen_traj):
    P = extract_patterns(real_traj)
    Pg = extract_patterns(gen_traj)

    tp = len(P & Pg)
    precision = tp / (len(Pg) + 1e-12)
    recall = tp / (len(P) + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

# ==========================================================
# 8. Iterate generated trajectories
# ==========================================================
print("\nScanning generated trajectories...")

pattern = re.compile(r"Gen_traj_noise_([0-9.]+)\.pkl")
records = []

for fname in tqdm(sorted(os.listdir(GEN_DIR))):
    m = pattern.match(fname)
    if m is None:
        continue

    noise = float(m.group(1))
    with open(os.path.join(GEN_DIR, fname), "rb") as f:
        gen_traj = np.array(pickle.load(f))

    n = min(len(real_traj), len(gen_traj))
    real_sub = real_traj[:n]
    gen_traj = gen_traj[:n]

    records.append({
        "noise": noise,
        "ssim_start": compute_ssim(real_sub, gen_traj, "start"),
        "ssim_end": compute_ssim(real_sub, gen_traj, "end"),
        "density_jsd": compute_density_jsd(real_sub, gen_traj),
        "trip_jsd": compute_trip_jsd(real_sub, gen_traj),
        "length_jsd": compute_length_jsd(real_sub, gen_traj),
        "pattern_f1": compute_pattern_score(real_sub, gen_traj)[2]
    })

# ==========================================================
# 9. Save results
# ==========================================================
df = pd.DataFrame(records).sort_values("noise").reset_index(drop=True)

print("\n========== Evaluation Summary ==========")
print(df)

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")