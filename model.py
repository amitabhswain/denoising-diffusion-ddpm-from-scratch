"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    return 1.0 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    out = a.gather(0, t.long())
    return out.reshape(-1, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x0)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'T': T,
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return torch.mean((noise - noise_pred) ** 2)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half = dim // 2
    t = t.float()

    if half == 1:
        exponents = torch.zeros(1, dtype=torch.float32, device=t.device)
    else:
        i = torch.arange(half, dtype=torch.float32, device=t.device)
        exponents = i / (half - 1)

    freqs = 10000 ** exponents          # shape (half,)
    args = t[:, None] / freqs[None, :]  # shape (B, half)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    
    std = 0.02
    
    conv_in_w = torch.randn(hidden, in_ch, 3, 3) * std
    conv_in_b = torch.zeros(hidden)
    
    time_mlp_w = torch.randn(hidden, time_dim) * std
    time_mlp_b = torch.zeros(hidden)
    
    conv_mid_w = torch.randn(hidden, hidden, 3, 3) * std
    conv_mid_b = torch.zeros(hidden)
    
    conv_out_w = torch.randn(in_ch, hidden, 3, 3) * std
    conv_out_b = torch.zeros(in_ch)
    
    params = {
        'conv_in_w': conv_in_w,
        'conv_in_b': conv_in_b,
        'time_mlp_w': time_mlp_w,
        'time_mlp_b': time_mlp_b,
        'conv_mid_w': conv_mid_w,
        'conv_mid_b': conv_mid_b,
        'conv_out_w': conv_out_w,
        'conv_out_b': conv_out_b,
    }
    
    for p in params.values():
        p.requires_grad_(True)
    
    return params

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # Step 1: first convolution
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)
    
    # Step 2: timestep embedding, projected and added as a per-channel bias
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    h = h + temb[:, :, None, None]
    
    # Step 3: nonlinearity, then a second convolution
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'], padding=1))
    
    # Step 4: final convolution back to in_ch channels (the noise prediction)
    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'], padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    
    radius = size // 4
    
    images = torch.zeros(n, 1, size, size)
    
    # coordinate grids for computing distance from a center, reused for every sample
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij'
    )
    
    for i in range(n):
        center = torch.randint(radius, size - radius, (2,))
        cy, cx = center[0].float(), center[1].float()
        
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        disk_mask = dist_sq <= radius ** 2
        
        images[i, 0][disk_mask] = 1.0
    
    return images

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    torch.manual_seed(seed)
    
    B = x0.shape[0]
    T = schedule['T']
    alphas_cumprod = schedule['alphas_cumprod']
    
    # sample per-sample random timesteps and fresh Gaussian noise
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)
    
    # forward pass + loss
    model = lambda x, t: tiny_unet_forward(x, t, params)
    loss = diffusion_training_loss(model, x0, t, noise, alphas_cumprod)
    
    # backward pass: populate .grad on every param in `params`
    loss.backward()
    
    # manual SGD update, re-detached and re-attached for the next step
    new_params = {}
    for name, p in params.items():
        if p.grad is not None:
            new_params[name] = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            new_params[name] = p.detach().clone().requires_grad_(True)
    
    return new_params, float(loss)

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    n = dataset.shape[0]
    history = []
    
    for step in range(num_steps):
        # seed the batch-sampling RNG for this step, then draw a minibatch
        torch.manual_seed(seed + step)
        idx = torch.randint(0, n, (batch_size,))
        x0 = dataset[idx]
        
        # run one atomic training step, seeded identically for t/noise sampling
        params, loss = ddpm_train_step(params, x0, schedule, lr=lr, seed=seed + step)
        history.append(loss)
    
    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    betas = schedule['betas']
    
    # x0 estimate from the predicted noise, clamped for stability
    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)
    x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
    
    # Build a shifted alphas_cumprod array so index t gives bar_alpha_{t-1},
    # with bar_alpha_{-1} := 1 for t == 0
    alphas_cumprod_prev_full = torch.cat(
        [torch.ones(1, dtype=alphas_cumprod.dtype, device=alphas_cumprod.device),
         alphas_cumprod[:-1]]
    )
    
    # Per-sample lookups, each broadcastable to x_t's shape (B, C, H, W)
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)
    alpha_bar_prev = extract_into_batch(alphas_cumprod_prev_full, t, x_t)
    beta_t = extract_into_batch(betas, t, x_t)
    alpha_t = extract_into_batch(alphas, t, x_t)
    
    # Posterior mean, mixing the x0 estimate and the current noisy sample
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    mean = coef_x0 * x0_hat + coef_xt * x_t
    
    # Simple fixed-variance choice
    variance = beta_t
    
    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

