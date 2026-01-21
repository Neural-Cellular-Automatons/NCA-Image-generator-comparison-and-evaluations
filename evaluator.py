# ============================================================
# NCA EVALUATION - Run AFTER training completes
# Generates GIFs + Improved Metrics + Clean Dashboards
# ============================================================

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Dict

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR = "/kaggle/working/nca_comparison_extended/car"
EVAL_DIR = "/kaggle/working/nca_evaluation_new"
GROWTH_STEPS = 300
DRIFT_STEPS = 100
GIF_STEPS = 200
GIF_FPS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EVAL_DIR, exist_ok=True)

CONFIG = {"channels": 16, "hidden_size": 128, "fire_rate": 0.5}

COLORS = {
    "Canonical": "#1f77b4",
    "MeshNCA": "#ff7f0e",
    "DiffNCA": "#2ca02c",
    "AdaNCA": "#d62728",
    "LightAttnNCA": "#9467bd",
    "SpikingNCA": "#8c564b",
    "HierNCA": "#e377c2",
    "VarNCA":  "#7f7f7f",
    "GraphNCA": "#bcbd22",
}

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def to_rgb(x, bg="black"):
    if x.dim() == 4:
        rgb = x[: , :3]
        alpha = torch.clamp(x[:, 3:4], 0, 1)
    else:
        rgb = x[: 3]
        alpha = torch.clamp(x[3:4], 0, 1)
    if bg == "black":
        return rgb
    if bg == "white":
        return rgb + (1.0 - alpha)
    return rgb


def expand_target_to_channels(tgt, channels):
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(0)
    b, c, h, w = tgt.shape
    if c >= channels:
        return tgt[: , :channels]
    pad = torch.zeros(b, channels - c, h, w, device=tgt.device, dtype=tgt.dtype)
    return torch.cat([tgt, pad], dim=1)


class BaseCAModel(nn.Module):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.dx_clip = dx_clip
        self.perception = nn.Conv2d(channels, channels * 3, 3, padding=1, bias=False)
        self.dmodel = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, channels, 1)
        )
        nn.init.zeros_(self. dmodel[-1].weight)
        nn.init.zeros_(self.dmodel[-1]. bias)
        self._initialized = False

    def _init_weights(self):
        if self._initialized:
            return
        if getattr(self, "perception", None) is None:
            self._initialized = True
            return
        with torch.no_grad():
            dev = self.perception.weight.device
            ident = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=dev) / 6.0
            dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev) / 8.0
            dy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev) / 8.0
            self.perception.weight. zero_()
            for i in range(self.channels):
                self.perception.weight[i * 3 + 0, i] = ident
                self.perception.weight[i * 3 + 1, i] = dx
                self.perception.weight[i * 3 + 2, i] = dy
        self._initialized = True

    def perceive(self, x):
        return self.perception(x)

    def update(self, x, y, fire_rate=None):
        dx = self. dmodel(y)
        if self.dx_clip is not None:
            dx = dx.clamp(-self.dx_clip, self. dx_clip)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[: , 3:4], 3, 1, 1) > 0.01
        return x * alive. float()

    def pop_aux_loss(self):
        val = getattr(self, "_aux_loss", None)
        if val is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        self._aux_loss = None
        return val

    def forward(self, x, fire_rate=None, target=None):
        if not self._initialized:
            self._init_weights()
        y = self.perceive(x)
        return self.update(x, y, fire_rate)


class CAModel(BaseCAModel):
    pass


class MeshNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.perception = nn.Conv2d(channels + 2, (channels + 2) * 3, 3, padding=1, bias=False)
        self.dmodel = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, channels, 1)
        )
        nn.init.zeros_(self.dmodel[-1].weight)
        nn.init.zeros_(self.dmodel[-1].bias)
        self._initialized = False

    def _init_weights(self):
        if self._initialized:
            return
        with torch.no_grad():
            dev = self.perception.weight.device
            ident = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=dev) / 6.0
            dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev) / 8.0
            dy = torch. tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev) / 8.0
            self.perception.weight.zero_()
            for i in range(self.channels + 2):
                self.perception.weight[i * 3 + 0, i] = ident
                self.perception.weight[i * 3 + 1, i] = dx
                self.perception.weight[i * 3 + 2, i] = dy
        self._initialized = True

    def perceive(self, x):
        b, c, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        x_aug = torch.cat([x, xx, yy], 1)
        return self.perception(x_aug)[:, :c * 3]


class DiffNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None,
                 init_diffusion=0.02, diffusion_scale=0.2):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.laplacian = nn.Conv2d(channels, channels, 3, padding=1, bias=False, groups=channels)
        self._lap_init = False
        init_ratio = max(min(init_diffusion / diffusion_scale, 0.999), 0.001)
        self.diffusion_logit = nn.Parameter(torch. log(torch.tensor(init_ratio / (1 - init_ratio), dtype=torch.float32)))
        self._diff_scale = diffusion_scale

    def _init_laplacian(self):
        if self._lap_init:
            return
        with torch.no_grad():
            dev = self. laplacian.weight.device
            lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev)
            self.laplacian.weight.zero_()
            for i in range(self.channels):
                self.laplacian. weight[i, 0] = lap
        self._lap_init = True

    def update(self, x, y, fire_rate=None):
        if not self._lap_init:
            self._init_laplacian()
        diff = self._diff_scale * torch.sigmoid(self.diffusion_logit)
        dx = self. dmodel(y) + diff * self.laplacian(x)
        if self.dx_clip is not None:
            dx = dx.clamp(-self.dx_clip, self.dx_clip)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()


class AdaNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_size // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, fire_rate=None, target=None):
        if not self._initialized:
            self._init_weights()
        y = self.perceive(x)
        dx = self.dmodel(y)
        if self.dx_clip is not None:
            dx = dx.clamp(-self.dx_clip, self.dx_clip)
        if target is not None and self.training:
            tgt = target.unsqueeze(0) if target.dim() == 3 else target
            tgt_state = expand_target_to_channels(tgt, x.shape[1]).expand(x.shape[0], -1, -1, -1)
            guide = torch.cat([x, x - tgt_state], 1)
            dx = dx * self.gate(guide)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()


class LightAttnNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None, attn_mid=32):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, attn_mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_mid, channels * 3, 1),
            nn.Sigmoid()
        )

    def perceive(self, x):
        y = super().perceive(x)
        a = self.attn(x)
        return y * (0.5 + 0.5 * a)


class SpikingNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5,
                 mem_idx=4, thresh=0.2, leak=0.0, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.mem_idx = mem_idx
        self.thresh = thresh
        self. leak = leak
        self.mem_head = nn.Conv2d(channels * 3, 1, 1)
        self.spike_warmup_steps = 0
        self._outer_step = 0

    def update(self, x, y, fire_rate=None):
        dx = self.dmodel(y)
        if self.dx_clip is not None:
            dx = dx.clamp(-self.dx_clip, self.dx_clip)
        if self.training and self._outer_step < self.spike_warmup_steps:
            fr = self.fire_rate if fire_rate is None else fire_rate
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            dmem = torch.tanh(self.mem_head(y))
            p = x[: , self.mem_idx:self.mem_idx + 1] + dmem - self.leak
            p = torch.clamp(p, -2.0, 2.0)
            spiked = (p >= self.thresh).float()
            p = p - self.thresh * spiked
            if self.training:
                fr = self.fire_rate if fire_rate is None else fire_rate
                mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
                x = x + dx * (mask * spiked)
            else:
                x = x + dx * spiked
            x[: , self.mem_idx:self.mem_idx + 1] = p
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x * alive.float()


class HierNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, pool=2, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.low_proj = nn.Conv2d(channels, channels, 1)
        self.pool = pool

    def perceive(self, x):
        if not self._initialized:
            self._init_weights()
        y = super().perceive(x)
        x_low = F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)
        inj = F.interpolate(self.low_proj(x_low), size=x.shape[-2:], mode='nearest')
        b, c, _, _ = x.shape
        y[: , : c] = y[:, :c] + inj
        return y


class VarNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.mu_head = nn.Conv2d(hidden_size, channels, 1)
        self.logvar_head = nn.Conv2d(hidden_size, channels, 1)
        self. encoder = nn.Sequential(nn.Conv2d(channels * 3, hidden_size, 1), nn.ReLU(inplace=True))
        self._aux_loss = None

    def update(self, x, y, fire_rate=None):
        h = self.encoder(y)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        dx = mu + eps * std
        if self.dx_clip is not None:
            dx = dx.clamp(-self.dx_clip, self.dx_clip)
        kl = -0.5 * (1 + logvar - mu. pow(2) - torch.exp(logvar))
        self._aux_loss = kl. mean()
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()


class GraphNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.dir_attn = nn.Conv2d(channels, 4 * channels, 1)
        self.msg_proj = nn.Conv2d(channels, channels, 1)
        self.perception = None
        self. register_buffer("_sobel_dx", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0)

    def shift(self, x, direction):
        if direction == 'N':
            return F.pad(x, (0, 0, 1, 0))[: , :, :-1, :]
        if direction == 'S':
            return F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        if direction == 'W':
            return F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]
        if direction == 'E':
            return F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        return x

    def perceive(self, x):
        b, c, h, w = x.shape
        n = self.shift(x, 'N')
        s = self.shift(x, 'S')
        wv = self.shift(x, 'W')
        e = self.shift(x, 'E')
        msgs = torch.stack([n, s, wv, e], dim=1).view(b * 4, c, h, w)
        msgs = self.msg_proj(msgs).view(b, 4, c, h, w)
        logits = self.dir_attn(x).view(b, 4, c, h, w)
        attn = F.softmax(logits, dim=1)
        agg = (attn * msgs).sum(dim=1)
        dx_k = self._sobel_dx. to(x.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        sobel_x = F.conv2d(x, dx_k, padding=1, groups=c)
        return torch.cat([x, agg, sobel_x], dim=1)


def make_model(name, cfg):
    ch = cfg.get("channels", 16)
    hs = cfg.get("hidden_size", 128)
    fr = cfg.get("fire_rate", 0.5)
    dx_clip = cfg.get("dx_clip")

    if name == "DiffNCA":
        return DiffNCA(ch, hs, fr, dx_clip, 0.02, 0.2)
    if name == "SpikingNCA":
        m = SpikingNCA(ch, hs, fr, mem_idx=4, thresh=0.2, leak=0.0, dx_clip=dx_clip)
        m.spike_warmup_steps = 600
        return m

    mapping = {
        "Canonical": CAModel,
        "MeshNCA": MeshNCA,
        "AdaNCA": AdaNCA,
        "LightAttnNCA": LightAttnNCA,
        "HierNCA": HierNCA,
        "VarNCA": VarNCA,
        "GraphNCA": GraphNCA,
    }
    if name not in mapping:
        raise ValueError(f"Unknown model: {name}")
    return mapping[name](ch, hs, fr, dx_clip)


# ============================================================
# HELPERS
# ============================================================
def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse < 1e-12 else -10 * np.log10(mse)


def calc_ssim(pred, target):
    return F.cosine_similarity(pred. flatten().unsqueeze(0), target.flatten().unsqueeze(0)).item()


def make_seed(size, channels, device):
    seed = torch.zeros(1, channels, size, size, device=device)
    seed[: , 3:, size // 2, size // 2] = 1.0
    return seed


def save_gif(frames, path, bg="black", fps=20):
    if not frames: 
        return
    imgs = []
    for t in frames:
        np_img = to_rgb(t, bg=bg).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        imgs.append(Image.fromarray((np_img * 255).astype(np.uint8)))
    duration = int(1000 / fps)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


# ============================================================
# LOAD TARGET & SUMMARY
# ============================================================
print(f"Loading from:  {TRAIN_DIR}")

target_path = f"{TRAIN_DIR}/target.png"
target_img = Image.open(target_path).convert("RGBA")
target = torch.from_numpy(np.float32(target_img) / 255.0).permute(2, 0, 1).to(device)
size = target.shape[-1]
print(f"Target:  {size}x{size}")

with open(f"{TRAIN_DIR}/summary.json") as f:
    summary = json.load(f)

model_names = [r["model"] for r in summary["results"]]
print(f"Models: {model_names}")

# ============================================================
# EVALUATE ALL MODELS
# ============================================================
results = {}

for name in model_names:
    model_dir = f"{TRAIN_DIR}/{name. lower()}"
    model_path = f"{model_dir}/model.pth"
    if not os.path.exists(model_path):
        print(f"Skip {name} (no model found)")
        continue

    print(f"\n{'=' * 60}\nEvaluating:  {name}\n{'=' * 60}")

    model = make_model(name, CONFIG).to(device)
    model.load_state_dict(torch. load(model_path, map_location=device))
    model.eval()

    with open(f"{model_dir}/metrics.json") as f:
        tm = json.load(f)

    m = {
        "name": name,
        "params": tm. get("parameters", 0),
        "train_time": tm.get("train_time_min", 0) * 60,
    }

    # ===== GROWTH + GIF =====
    seed = make_seed(size, CONFIG.get("channels", 16), device)
    x = seed. clone()
    psnr_hist = []
    hidden_energy = []
    rgb_energy = []
    best_psnr = -999
    best_step = 0
    best_state = None
    gif_frames = []

    with torch.no_grad():
        for step in range(GROWTH_STEPS):
            if hasattr(model, "_outer_step"):
                model._outer_step = 10000
            x = model(x, fire_rate=1.0, target=target if name == "AdaNCA" else None)

            p = calc_psnr(x[0, : 4], target[: 4])
            psnr_hist.append(p)

            if x.shape[1] > 4:
                hidden_energy.append((x[0, 4:] ** 2).mean().item())
            else:
                hidden_energy.append(0)
            rgb_energy.append((x[0, : 3] ** 2).mean().item())

            if p > best_psnr:
                best_psnr = p
                best_step = step
                best_state = x.clone()

            if step <= GIF_STEPS: 
                gif_frames.append(x[0]. detach().clone())

    gif_path = f"{model_dir}/growth.gif"
    save_gif(gif_frames, gif_path, bg="black", fps=GIF_FPS)
    print(f"  Saved GIF:  {gif_path}")

    m["psnr_hist"] = psnr_hist
    m["best_psnr"] = best_psnr
    m["best_step"] = best_step
    m["final_psnr"] = psnr_hist[-1]
    m["fre"] = F.mse_loss(best_state[0, :4], target[:4]).item()
    m["ssim"] = calc_ssim(best_state[0, :3], target[:3])

    # ===== TεC =====
    m["tec_25"] = next((i for i, p in enumerate(psnr_hist) if p >= 25), -1)
    m["tec_30"] = next((i for i, p in enumerate(psnr_hist) if p >= 30), -1)
    m["tec_35"] = next((i for i, p in enumerate(psnr_hist) if p >= 35), -1)

    # ===== CSI =====
    post_start = min(best_step, len(psnr_hist) - 50)
    m["csi"] = np.var(psnr_hist[post_start:post_start + 50])

    # ===== SSD =====
    ssd_hist = []
    x_drift = best_state. clone()
    with torch.no_grad():
        for _ in range(DRIFT_STEPS):
            x_prev = x_drift.clone()
            x_drift = model(x_drift, fire_rate=1.0, target=target if name == "AdaNCA" else None)
            ssd_hist.append(((x_drift - x_prev) ** 2).mean().item())
    m["ssd"] = np.mean(ssd_hist)
    m["ssd_hist"] = ssd_hist

    # ===== BRS =====
    x_n = best_state.clone() + torch.randn_like(best_state) * 0.3
    with torch.no_grad():
        for _ in range(100):
            x_n = model(x_n, fire_rate=1.0, target=target if name == "AdaNCA" else None)
    m["brs_noise"] = calc_psnr(x_n[0, :4], target[:4])

    x_e = best_state.clone()
    x_e[: , : , size // 4: size // 2, size // 4:size // 2] = 0
    with torch.no_grad():
        for _ in range(100):
            x_e = model(x_e, fire_rate=1.0, target=target if name == "AdaNCA" else None)
    m["brs_erase"] = calc_psnr(x_e[0, :4], target[:4])
    m["brs"] = (m["brs_noise"] + m["brs_erase"]) / 2

    # ===== Efficiency =====
    m["per"] = m["best_psnr"] / np.log(m["params"] + 1)
    m["tnq"] = m["best_psnr"] / (m["train_time"] + 1e-6)
    tec = m["tec_30"] if m["tec_30"] > 0 else GROWTH_STEPS
    m["nes"] = m["best_psnr"] / (m["params"] * tec) * 1e6

    # ===== HSER =====
    avg_hidden = np.mean(hidden_energy[-50:])
    avg_rgb = np.mean(rgb_energy[-50:])
    m["hser"] = avg_hidden / (avg_rgb + 1e-10)

    results[name] = m
    print(f"  PSNR*: {m['best_psnr']:.2f} | TεC30: {m['tec_30']} | CSI: {m['csi']:.4f}")
    print(f"  SSD: {m['ssd']:.2e} | BRS: {m['brs']:.2f} | NES: {m['nes']:.4f}")

# ============================================================
# PER-MODEL DASHBOARD
# ============================================================
for name, m in results.items():
    out = f"{EVAL_DIR}/{name. lower()}"
    os.makedirs(out, exist_ok=True)
    c = COLORS. get(name, "gray")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{name} - Evaluation Dashboard", fontweight="bold", fontsize=14)

    axes[0, 0].plot(m["psnr_hist"], color=c, lw=2)
    axes[0, 0].axhline(m["best_psnr"], color="r", ls="--", alpha=0.5)
    axes[0, 0].scatter([m["best_step"]], [m["best_psnr"]], c="red", s=100, marker="*")
    axes[0, 0].set_title("PSNR Trajectory")
    axes[0, 0].grid(True, alpha=0.3)

    tec_25_val = m["tec_25"] if m["tec_25"] > 0 else GROWTH_STEPS
    tec_30_val = m["tec_30"] if m["tec_30"] > 0 else GROWTH_STEPS
    tec_35_val = m["tec_35"] if m["tec_35"] > 0 else GROWTH_STEPS
    axes[0, 1].bar(["25dB", "30dB", "35dB"], [tec_25_val, tec_30_val, tec_35_val], color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[0, 1].set_title("Time-to-e (TeC)")

    if m["best_step"] < len(m["psnr_hist"]) - 20:
        post = m["psnr_hist"][m["best_step"]:]
    else:
        post = m["psnr_hist"][-50:]
    axes[0, 2].plot(post, color=c, lw=2)
    axes[0, 2].set_title(f"Stability (CSI={m['csi']:.4f})")

    axes[0, 3]. semilogy(m["ssd_hist"], color=c, lw=2)
    axes[0, 3].set_title(f"Drift (SSD={m['ssd']:.2e})")

    axes[1, 0].bar(["Noise", "Erase", "Combined"], [m["brs_noise"], m["brs_erase"], m["brs"]], color=["#9b59b6", "#1abc9c", "#e74c3c"])
    axes[1, 0].set_title("Basin Robustness (BRS)")

    axes[1, 1].bar(["PER", "TNQx100", "NES"], [m["per"], m["tnq"] * 100, m["nes"]], color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[1, 1]. set_title("Efficiency Metrics")

    axes[1, 2].bar(["HSER"], [m["hser"]], color=c)
    axes[1, 2].set_title("Hidden Energy Ratio")

    axes[1, 3].axis("off")
    summary_txt = (
        f"{name}\n"
        f"-------------------------\n"
        f"PSNR*: {m['best_psnr']:.2f} dB\n"
        f"SSIM:   {m['ssim']:.4f}\n"
        f"FRE:   {m['fre']:.6f}\n\n"
        f"TeC(25/30/35): {m['tec_25']}/{m['tec_30']}/{m['tec_35']}\n"
        f"CSI:    {m['csi']:.4f}\n"
        f"SSD:   {m['ssd']:.2e}\n"
        f"BRS:    {m['brs']:.2f}\n\n"
        f"PER:   {m['per']:.3f}\n"
        f"TNQ:   {m['tnq']:.4f}\n"
        f"NES:   {m['nes']:.4f}\n"
        f"HSER:   {m['hser']:.4f}"
    )
    axes[1, 3].text(0.05, 0.95, summary_txt, fontsize=10, family="monospace", va="top")
    plt.tight_layout()
    plt.savefig(f"{out}/dashboard.png", dpi=150)
    plt.close()
    print(f"  Saved dashboard for {name}")

print(f"\nAll evaluation outputs saved to: {EVAL_DIR}")