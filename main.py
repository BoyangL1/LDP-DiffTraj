import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Sampler
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil

# ==============================================================================
# Environment Setup
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class TimeSortBatchSampler(Sampler):
    """
    Sorts data by privacy score and batches them together.
    Ensures that within a batch, privacy scores are very similar,
    allowing us to set a unified t_min for the whole batch.
    """
    def __init__(self, time_scores, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(time_scores)
        
        # Convert to numpy for argsort
        if isinstance(time_scores, torch.Tensor):
            time_scores = time_scores.cpu().numpy()
            
        self.sorted_indices = np.argsort(time_scores)
        
    def __iter__(self):
        batches = []
        # Slice sorted indices into batches
        for i in range(0, self.num_samples, self.batch_size):
            if i + self.batch_size > self.num_samples:
                if not self.drop_last:
                    batches.append(self.sorted_indices[i:])
            else:
                batches.append(self.sorted_indices[i : i + self.batch_size])
        
        # Shuffle batches to ensure training randomness (Global Shuffle, Local Sort)
        np.random.shuffle(batches)
        
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        
def compute_alpha_bar_torch(beta_start, beta_end, num_steps, device='cuda'):
    """
    Computes the alpha_bar schedule based on config parameters.
    Returns: Tensor of shape [num_steps]
    """
    betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

# ==============================================================================
# 2. Main Training Function
# ==============================================================================

def main(config, logger, exp_dir):

    # ----------------------------------------
    # A. Setup Diffusion Schedule
    # ----------------------------------------
    n_steps = config.diffusion.num_diffusion_timesteps
    beta_start = config.diffusion.beta_start
    beta_end = config.diffusion.beta_end
    
    # Compute Alpha Bar strictly following the config
    alpha_bar = compute_alpha_bar_torch(beta_start, beta_end, n_steps).cuda()

    # Forward Process: q(x_t | x_0)
    def q_xt_given_xtmin(xtmin, t, t_min):
        alpha_ratio = gather(alpha_bar, t) / (gather(alpha_bar, t_min) + 1e-12)
        alpha_ratio = torch.clamp(alpha_ratio, min=1e-12, max=1.0)

        mean = torch.sqrt(alpha_ratio) * xtmin
        var = 1.0 - alpha_ratio
        eps = torch.randn_like(xtmin)
        return mean + torch.sqrt(var) * eps, eps

    # ----------------------------------------
    # B. Model & Optimizer
    # ----------------------------------------
    unet = Guide_UNet(config).cuda()
    
    base_lr = 1e-4  
    weight_decay = 1e-4
    optim = torch.optim.AdamW(unet.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config.training.n_epochs, eta_min=1e-6
    )
    
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # ----------------------------------------
    # C. Data Loading with Privacy Sorting
    # ----------------------------------------
    traj = np.load(config.data.traj_path, allow_pickle=True)
    head_raw = np.load(config.data.head_path, allow_pickle=True) 
    
    # [Batch, 2, Len]
    traj = np.swapaxes(traj, 1, 2)
    traj = torch.from_numpy(traj).float()
    head_tensor = torch.from_numpy(head_raw).float()
    
    # Extract Privacy Column (Assuming last column index -1)
    time_all = head_tensor[:, -1].long()
    
    dataset = TensorDataset(traj, head_tensor)
    
    # Use Custom Sampler
    batch_sampler = TimeSortBatchSampler(time_all, batch_size=config.training.batch_size)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8)

    # ----------------------------------------
    # D. Training Loop
    # ----------------------------------------
    model_save_dir = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        epoch_losses = []
        
        for iter_idx, (trainx, head_batch) in enumerate(dataloader):
            x0 = trainx.cuda()
            head_batch = head_batch.cuda()

            # 3. Calculate Minimum Timestep (t_min) via Inverse Function
            t_i = head_batch[:, -1].long()            
            u = torch.rand(len(x0), device=x0.device)
            t_batch = (t_i + (u * (n_steps - t_i)).long()).clamp(max=n_steps - 1)
            
            # 5. Add Noise (Forward Diffusion)
            xtmin = x0                       # x0 is actually x_{t_i}
            tmin_batch = t_i                 # (B,)
            xt, noise = q_xt_given_xtmin(xtmin, t_batch, tmin_batch)
            
            # 6. Model Prediction
            # Slice the head attributes for conditional generation (exclude privacy col if needed)
            attr = head_batch[:, :config.data.head_dim]
            pred_noise = unet(xt.float(), t_batch, attr)
            
            # 7. Optimization
            loss = F.mse_loss(noise.float(), pred_noise)
            epoch_losses.append(loss.item())
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if config.model.ema:
                ema_helper.update(unet)
            
            # Logging
            if iter_idx % 1 == 0:
                logger.info(
                    f"Epoch {epoch}, Iter {iter_idx}, Loss: {loss.item():.6f}, "
                    f"t_i[min,max]=({t_i.min().item()},{t_i.max().item()})"
                )

        scheduler.step()
        # End of Epoch Saving
        if (epoch) % 50 == 0:
            m_path = model_save_dir / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            
            loss_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            np.save(loss_path, np.array(epoch_losses))

# ==============================================================================
# 3. Entry Point
# ==============================================================================

if __name__ == "__main__":
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    base_config = SimpleNamespace(**temp)

    root_dir = Path(__name__).resolve().parents[0]

    NOISE_LEVELS = [f"{i/10:.2f}" for i in range(0, 11)] 

    for nl in NOISE_LEVELS:
        print(f"\n========== Training on noise_{nl} ==========")

        config = SimpleNamespace(**vars(base_config))
        config.data = SimpleNamespace(**vars(base_config.data))

        config.data.traj_path = f"/xxx/ldp-traj/data_privacy/noise_sweep/noise_{nl}/traj.npy"
        config.data.head_path = f"/xxx/ldp-traj/data_privacy/noise_sweep/noise_{nl}/traj_features.npy"

        result_name = f"{config.data.dataset}_noise_{nl}_steps={config.diffusion.num_diffusion_timesteps}_len={config.data.traj_length}"
        exp_dir = root_dir / "LDP-DiffTraj" / result_name

        for d in ["results", "models", "logs", "Files"]:
            os.makedirs(exp_dir / d, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

        logger = Logger(
            __name__,
            log_path=exp_dir / "logs" / (timestamp + '.log'),
            colorize=True,
        )
        log_info(config, logger)

        main(config, logger, exp_dir)