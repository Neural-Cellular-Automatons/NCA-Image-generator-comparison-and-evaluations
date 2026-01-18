# NCA Extended Comparison Suite - GPU Optimized (T4 friendly)
# Stable dynamics, OOM-safe, deterministic growth capture, final GIFs
# Variants: Canonical, MeshNCA, DiffNCA, AdaNCA, LightAttnNCA, SpikingNCA, HierNCA, VarNCA, GraphNCA

import os, time, warnings, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# -----------------------
# ========== CONFIG ======
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPUs available: {torch.cuda.device_count()}")

CONFIG: Dict = {
    # Paths
    "image_path": "/kaggle/input/nca-basic-dataset/car.png",
    "save_dir": "/kaggle/working/nca_comparison_extended",

    # Models to compare (reduce list to speed up)
    "models": [
        "Canonical", "MeshNCA", "DiffNCA", "AdaNCA",
        "LightAttnNCA", "SpikingNCA", "HierNCA", "VarNCA", "GraphNCA"
    ],

    # Image settings
    "img_size": 128,
    "padding": 12,

    # Core state
    "channels": 16,           # first 4 are RGBA
    "hidden_size": 128,
    "fire_rate": 0.5,

    # Training
    "pool_size": 512,
    "batch": 8,
    "lr": 2e-3,
    "lr_decay_step": 2000,
    "train_steps": 3000,
    "min_steps": 32,
    "max_steps": 64,

    # Optimization
    "mixed_precision": True,
    "grad_norm_clip": 1.0,

    # Aux loss (VarNCA)
    "aux_loss_weight": 1e-3,

    # Visualization
    "viz_milestones": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
    "growth_steps": [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
    "save_animations": True,            # final GIF per model
    "save_checkpoint_gifs": False,      # optional GIFs at milestones
    "viz_background": "black",          # black|white|gray|checker|raw

    # Safety knobs
    "dx_clip": None,                    # e.g., 0.3 to clamp updates

    # Per-model overrides to improve stability and avoid OOM
    "lr_overrides": {
        "Canonical": 1e-3,
        "DiffNCA": 1e-3,
        "SpikingNCA": 1e-3,
        "VarNCA": 1e-3
    },
    "fr_overrides": {
        "Canonical": 0.7
    },
    "batch_overrides": {
        "LightAttnNCA": 4,   # attention ≈ heavier
        "SpikingNCA": 6,     # more state interactions
        "VarNCA": 6          # stochastic heads
    },

    # DiffNCA bounded diffusion
    "diffusion_init": 0.02,
    "diffusion_scale": 0.2,

    # SpikingNCA warmup and bounds
    "spike_warmup_steps": 600,  # outer steps before enforcing spiking
    "spike_thresh": 0.2,
    "spike_leak": 0.0,

    "device": device,
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True

os.makedirs(CONFIG["save_dir"], exist_ok=True)

print(f"\n{'='*70}")
print(f"NCA Extended Comparison - T4 GPU Optimized [Stable + OOM-safe]")
print(f"Models: {CONFIG['models']} | Steps: {CONFIG['train_steps']}")
print(f"Image: {CONFIG['img_size']}x{CONFIG['img_size']} | Batch: {CONFIG['batch']}")
print(f"{'='*70}\n")

# -----------------------
# ===== Utils =====
# -----------------------
def load_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    img_np = np.float32(img) / 255.0
    img_np[..., :3] *= img_np[..., 3:]  # premultiply
    return img_np

def to_rgb(x: torch.Tensor, bg: str | tuple = "black") -> torch.Tensor:
    if x.dim() == 4:
        rgb = x[:, :3]; alpha = torch.clamp(x[:, 3:4], 0, 1)
    else:
        rgb = x[:3]; alpha = torch.clamp(x[3:4], 0, 1)

    if isinstance(bg, str):
        bg = bg.lower()
        if bg == "raw":   return rgb
        if bg == "black": return rgb
        if bg == "white": return rgb + (1.0 - alpha)
        if bg == "gray":
            bg_rgb = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype)
            bg_rgb = bg_rgb.view((1,3,1,1)) if x.dim()==4 else bg_rgb.view(3,1,1)
            return rgb + (1.0 - alpha) * bg_rgb
        if bg == "checker":
            if x.dim()==4:
                b, _, h, w = x.shape
                yy = torch.arange(h, device=x.device).view(1,1,h,1)
                xx = torch.arange(w, device=x.device).view(1,1,1,w)
                pattern = ((yy // 8 + xx // 8) % 2).float()
                bg0 = torch.tensor([0.7,0.7,0.7], device=x.device).view(1,3,1,1)
                bg1 = torch.tensor([0.9,0.9,0.9], device=x.device).view(1,3,1,1)
                bg_rgb = bg0*(1-pattern) + bg1*pattern
                return rgb + (1.0 - alpha) * bg_rgb
            else:
                _, h, w = x.shape
                yy = torch.arange(h, device=x.device).view(1,h,1)
                xx = torch.arange(w, device=x.device).view(1,1,w)
                pattern = ((yy // 8 + xx // 8) % 2).float()
                bg0 = torch.tensor([0.7,0.7,0.7], device=x.device).view(3,1,1)
                bg1 = torch.tensor([0.9,0.9,0.9], device=x.device).view(3,1,1)
                bg_rgb = bg0*(1-pattern) + bg1*pattern
                return rgb + (1.0 - alpha) * bg_rgb
        raise ValueError(f"Unknown bg='{bg}'")
    else:
        r,g,b = bg
        if x.dim()==4:
            bg_rgb = torch.tensor([r,g,b], device=x.device, dtype=x.dtype).view(1,3,1,1)
        else:
            bg_rgb = torch.tensor([r,g,b], device=x.device, dtype=x.dtype).view(3,1,1)
        return rgb + (1.0 - alpha) * bg_rgb

def to_rgba(x: torch.Tensor) -> torch.Tensor:
    return x[:, :4] if x.dim() == 4 else x[:4]

def expand_target_to_channels(tgt: torch.Tensor, channels: int) -> torch.Tensor:
    """Expand RGBA target to match CA state channels (pad zeros)."""
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(0)
    b, c, h, w = tgt.shape
    if c >= channels:
        return tgt[:, :channels]
    pad = torch.zeros(b, channels - c, h, w, device=tgt.device, dtype=tgt.dtype)
    return torch.cat([tgt, pad], dim=1)

def make_seed(size: int, n: int, channels: int, device) -> torch.Tensor:
    seed = torch.zeros(n, channels, size, size, device=device)
    c = size // 2
    seed[:, 3:, c, c] = 1.0
    return seed

def save_image_grid(images, labels, title, path, bg="black"):
    n = len(images); cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]
    for i, (img, lbl) in enumerate(zip(images, labels)):
        if isinstance(img, torch.Tensor):
            img = to_rgb(img, bg=bg).permute(1,2,0).clamp(0,1).cpu().numpy()
        axes[i].imshow(img); axes[i].set_title(lbl, fontsize=10); axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.suptitle(title, fontsize=13, fontweight='bold'); plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()

def save_gif(frames: List[torch.Tensor], path: str, bg="black", fps=20):
    from PIL import Image as PILImage
    if not frames: return
    imgs = []
    for t in frames:
        np_img = to_rgb(t, bg=bg).permute(1,2,0).clamp(0,1).cpu().numpy()
        imgs.append(PILImage.fromarray((np_img*255).astype(np.uint8)))
    duration = int(1000 / fps)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

# -----------------------
# ===== Base CA =====
# -----------------------
class BaseCAModel(nn.Module):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.dx_clip = dx_clip
        self.perception = nn.Conv2d(channels, channels*3, 3, padding=1, bias=False)
        self.dmodel = nn.Sequential(
            nn.Conv2d(channels*3, hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, channels, 1)
        )
        nn.init.zeros_(self.dmodel[-1].weight)
        nn.init.zeros_(self.dmodel[-1].bias)
        self._initialized = False

    def _init_weights(self):
        if self._initialized: return
        if getattr(self, "perception", None) is None:
            self._initialized = True; return
        with torch.no_grad():
            dev = self.perception.weight.device
            ident = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32, device=dev)/6.0
            dx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=dev)/8.0
            dy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=dev)/8.0
            self.perception.weight.zero_()
            for i in range(self.channels):
                self.perception.weight[i*3+0, i] = ident
                self.perception.weight[i*3+1, i] = dx
                self.perception.weight[i*3+2, i] = dy
        self._initialized = True

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return self.perception(x)

    def update(self, x: torch.Tensor, y: torch.Tensor, fire_rate: Optional[float]=None) -> torch.Tensor:
        dx = self.dmodel(y)
        if self.dx_clip is not None: dx = dx.clamp(-self.dx_clip, self.dx_clip)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        # Lower alive threshold so early growth doesn't die
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()

    def pop_aux_loss(self) -> torch.Tensor:
        val = getattr(self, "_aux_loss", None)
        if val is None: return torch.tensor(0.0, device=next(self.parameters()).device)
        self._aux_loss = None; return val

    def forward(self, x: torch.Tensor, fire_rate: Optional[float]=None, target: Optional[torch.Tensor]=None):
        if not self._initialized: self._init_weights()
        y = self.perceive(x)
        return self.update(x, y, fire_rate)

class CAModel(BaseCAModel):
    pass

class MeshNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.perception = nn.Conv2d(channels+2, (channels+2)*3, 3, padding=1, bias=False)
        self.dmodel = nn.Sequential(nn.Conv2d(channels*3, hidden_size, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden_size, channels, 1))
        nn.init.zeros_(self.dmodel[-1].weight); nn.init.zeros_(self.dmodel[-1].bias)
        self._initialized = False

    def _init_weights(self):
        if self._initialized: return
        with torch.no_grad():
            dev = self.perception.weight.device
            ident = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32, device=dev)/6.0
            dx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=dev)/8.0
            dy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=dev)/8.0
            self.perception.weight.zero_()
            for i in range(self.channels + 2):
                self.perception.weight[i*3+0, i] = ident
                self.perception.weight[i*3+1, i] = dx
                self.perception.weight[i*3+2, i] = dy
        self._initialized = True

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        b,c,h,w = x.shape
        yy = torch.linspace(-1,1,h,device=x.device).view(1,1,h,1).expand(b,1,h,w)
        xx = torch.linspace(-1,1,w,device=x.device).view(1,1,1,w).expand(b,1,h,w)
        x_aug = torch.cat([x, xx, yy], 1)
        return self.perception(x_aug)[:, :c*3]

# Bounded diffusion for stability
class DiffNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None,
                 init_diffusion: float=0.02, diffusion_scale: float=0.2):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.laplacian = nn.Conv2d(channels, channels, 3, padding=1, bias=False, groups=channels)
        self._lap_init = False
        init_ratio = max(min(init_diffusion / diffusion_scale, 0.999), 0.001)
        self.diffusion_logit = nn.Parameter(torch.log(torch.tensor(init_ratio/(1-init_ratio), dtype=torch.float32)))
        self._diff_scale = diffusion_scale

    def _init_laplacian(self):
        if self._lap_init: return
        with torch.no_grad():
            dev = self.laplacian.weight.device
            lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev)
            self.laplacian.weight.zero_()
            for i in range(self.channels):
                self.laplacian.weight[i, 0] = lap
        self._lap_init = True

    def update(self, x: torch.Tensor, y: torch.Tensor, fire_rate: Optional[float]=None) -> torch.Tensor:
        if not self._lap_init: self._init_laplacian()
        diff = self._diff_scale * torch.sigmoid(self.diffusion_logit)
        dx = self.dmodel(y) + diff * self.laplacian(x)
        if self.dx_clip is not None: dx = dx.clamp(-self.dx_clip, self.dx_clip)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()

class AdaNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.gate = nn.Sequential(
            nn.Conv2d(channels*2, hidden_size//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size//2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, fire_rate: Optional[float]=None, target: Optional[torch.Tensor]=None):
        if not self._initialized: self._init_weights()
        y = self.perceive(x)
        dx = self.dmodel(y)
        if self.dx_clip is not None: dx = dx.clamp(-self.dx_clip, self.dx_clip)
        if target is not None and self.training:
            tgt = target.unsqueeze(0) if target.dim()==3 else target
            tgt_state = expand_target_to_channels(tgt, x.shape[1]).expand(x.shape[0], -1, -1, -1)
            guide = torch.cat([x, x - tgt_state], 1)
            dx = dx * self.gate(guide)
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()

# Lightweight attention (no unfold): channel/spatial reweighting
class LightAttnNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None, attn_mid: int=32):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        # Perception keeps C*3; We generate an attention map to modulate it
        self.attn = nn.Sequential(
            nn.Conv2d(channels, attn_mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_mid, channels*3, 1),
            nn.Sigmoid()
        )

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        y = super().perceive(x)             # [B, 3C, H, W]
        a = self.attn(x)                    # [B, 3C, H, W] in [0,1]
        return y * (0.5 + 0.5*a)            # mild gating for stability

# Spiking with warmup, bounded membrane
class SpikingNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5,
                 mem_idx: int=4, thresh: float=0.2, leak: float=0.0, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.mem_idx = mem_idx; self.thresh = thresh; self.leak = leak
        self.mem_head = nn.Conv2d(channels*3, 1, 1)
        self.spike_warmup_steps = 0
        self._outer_step = 0

    def update(self, x: torch.Tensor, y: torch.Tensor, fire_rate: Optional[float]=None) -> torch.Tensor:
        dx = self.dmodel(y)
        if self.dx_clip is not None: dx = dx.clamp(-self.dx_clip, self.dx_clip)
        if self.training and self._outer_step < self.spike_warmup_steps:
            fr = self.fire_rate if fire_rate is None else fire_rate
            mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            dmem = torch.tanh(self.mem_head(y))           # bound drive
            p = x[:, self.mem_idx:self.mem_idx+1] + dmem - self.leak
            p = torch.clamp(p, -2.0, 2.0)                 # keep finite
            spiked = (p >= self.thresh).float()
            p = p - self.thresh * spiked
            if self.training:
                fr = self.fire_rate if fire_rate is None else fire_rate
                mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
                x = x + dx * (mask * spiked)
            else:
                x = x + dx * spiked
            x[:, self.mem_idx:self.mem_idx+1] = p
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return x * alive.float()

class HierNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, pool=2, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.low_proj = nn.Conv2d(channels, channels, 1)
        self.pool = pool

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized: self._init_weights()
        y = super().perceive(x)
        x_low = F.avg_pool2d(x, kernel_size=self.pool, stride=self.pool)
        inj = F.interpolate(self.low_proj(x_low), size=x.shape[-2:], mode='nearest')
        b, c, _, _ = x.shape
        y[:, :c] = y[:, :c] + inj
        return y

class VarNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.mu_head = nn.Conv2d(hidden_size, channels, 1)
        self.logvar_head = nn.Conv2d(hidden_size, channels, 1)
        self.encoder = nn.Sequential(nn.Conv2d(channels*3, hidden_size, 1), nn.ReLU(inplace=True))
        self._aux_loss = None

    def update(self, x: torch.Tensor, y: torch.Tensor, fire_rate: Optional[float]=None) -> torch.Tensor:
        h = self.encoder(y)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 10)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        dx = mu + eps * std
        if self.dx_clip is not None: dx = dx.clamp(-self.dx_clip, self.dx_clip)
        kl = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
        self._aux_loss = kl.mean()
        fr = self.fire_rate if fire_rate is None else fire_rate
        if self.training:
            mask = (torch.rand(x.shape[0],1,x.shape[2],x.shape[3],device=x.device) <= fr).float()
            x = x + dx * mask
        else:
            x = x + dx
        alive = F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.01
        return x * alive.float()

class GraphNCA(BaseCAModel):
    def __init__(self, channels=16, hidden_size=96, fire_rate=0.5, dx_clip: Optional[float]=None):
        super().__init__(channels, hidden_size, fire_rate, dx_clip=dx_clip)
        self.dir_attn = nn.Conv2d(channels, 4*channels, 1)
        self.msg_proj = nn.Conv2d(channels, channels, 1)
        self.perception = None
        self.register_buffer("_sobel_dx", torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)/8.0)

    def shift(self, x, direction: str):
        if direction == 'N': return F.pad(x, (0,0,1,0))[:, :, :-1, :]
        if direction == 'S': return F.pad(x, (0,0,0,1))[:, :, 1:, :]
        if direction == 'W': return F.pad(x, (1,0,0,0))[:, :, :, :-1]
        if direction == 'E': return F.pad(x, (0,1,0,0))[:, :, :, 1:]
        return x

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        b,c,h,w = x.shape
        n = self.shift(x,'N'); s = self.shift(x,'S'); wv = self.shift(x,'W'); e = self.shift(x,'E')
        msgs = torch.stack([n,s,wv,e], dim=1).view(b*4, c, h, w)
        msgs = self.msg_proj(msgs).view(b,4,c,h,w)
        logits = self.dir_attn(x).view(b,4,c,h,w)
        attn = F.softmax(logits, dim=1)
        agg = (attn * msgs).sum(dim=1)
        dx_k = self._sobel_dx.to(x.device).view(1,1,3,3).repeat(c,1,1,1)
        sobel_x = F.conv2d(x, dx_k, padding=1, groups=c)
        return torch.cat([x, agg, sobel_x], dim=1)

# -----------------------
# ===== Factory =====
# -----------------------
def make_model(name: str, cfg: Dict):
    ch = cfg["channels"]; hs = cfg["hidden_size"]
    fr = cfg["fr_overrides"].get(name, cfg["fire_rate"])
    dx_clip = cfg.get("dx_clip")
    if name == "DiffNCA":
        return DiffNCA(ch, hs, fr, dx_clip, cfg["diffusion_init"], cfg["diffusion_scale"])
    if name == "SpikingNCA":
        m = SpikingNCA(ch, hs, fr, mem_idx=4, thresh=cfg["spike_thresh"], leak=cfg["spike_leak"], dx_clip=dx_clip)
        m.spike_warmup_steps = int(cfg["spike_warmup_steps"])
        return m
    mapping = {
        "Canonical": CAModel,
        "MeshNCA":  MeshNCA,
        "AdaNCA":   AdaNCA,
        "LightAttnNCA": LightAttnNCA,
        "HierNCA":  HierNCA,
        "VarNCA":   VarNCA,
        "GraphNCA": GraphNCA,
    }
    if name not in mapping:
        raise ValueError(f"Unknown model: {name}")
    return mapping[name](ch, hs, fr, dx_clip)

# -----------------------
# ===== Pool & Metrics =====
# -----------------------
class SamplePool:
    def __init__(self, size, state_shape, device):
        self.size = size
        _, c, h, w = state_shape
        self.pool = make_seed(h, size, c, device)

    def sample(self, n):
        idx = np.random.choice(self.size, n, replace=False)
        return self.pool[idx].clone(), idx

    def commit(self, idx, states):
        self.pool[idx] = states.detach()

def loss_fn(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = to_rgba(x)
    tgt = target.unsqueeze(0).expand(pred.shape[0], -1, -1, -1)
    return F.mse_loss(pred, tgt)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.reshape(-1); t = target.reshape(-1)
    return F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()

# -----------------------
# ===== Growth helpers =====
# -----------------------
def grow_from_seed(model: nn.Module,
                   seed: torch.Tensor,
                   target: Optional[torch.Tensor],
                   total_steps: int,
                   capture_every: int = 1,
                   fire_rate: float = 1.0,
                   outer_step: int = 0) -> List[torch.Tensor]:
    was_training = model.training
    model.eval()
    frames: List[torch.Tensor] = []
    x = seed.clone().to(next(model.parameters()).device)
    steps_done = 0
    with torch.no_grad():
        while steps_done < total_steps:
            if isinstance(model, SpikingNCA): model._outer_step = outer_step
            x = model(x, fire_rate=fire_rate, target=target if isinstance(model, AdaNCA) else None)
            steps_done += 1
            if steps_done % capture_every == 0:
                frames.append(x[0].cpu().clone())
    if was_training: model.train()
    return frames

# -----------------------
# ===== Training =====
# -----------------------
def forward_step(model, x, fire_rate, target, outer_step):
    if isinstance(model, SpikingNCA): model._outer_step = outer_step
    out = model(x, fire_rate=fire_rate, target=target)
    x = out[0] if isinstance(out, tuple) else out
    aux = model.pop_aux_loss() if hasattr(model, "pop_aux_loss") else torch.tensor(0.0, device=x.device)
    return x, aux

def train_model(model_name, target_np, cfg):
    print(f"\n{'='*70}\nTraining: {model_name}\n{'='*70}")

    p = cfg["padding"]
    pad_target = np.pad(target_np, [(p,p), (p,p), (0,0)], mode='constant')
    target = torch.from_numpy(pad_target).permute(2,0,1).to(device)
    size = target.shape[1]

    model = make_model(model_name, cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    lr = cfg["lr_overrides"].get(model_name, cfg["lr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0 if s < cfg["lr_decay_step"] else 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["mixed_precision"])

    pool = SamplePool(cfg["pool_size"], (1, cfg["channels"], size, size), device)
    base_batch = cfg["batch_overrides"].get(model_name, cfg["batch"])
    seed = make_seed(size, 1, cfg["channels"], device)

    loss_log = []
    snapshots: Dict[int, List[torch.Tensor]] = {}
    start = time.time()

    for outer_step in range(cfg["train_steps"] + 1):
        # dynamic OOM-safe mini-batch
        mb = base_batch
        while True:
            try:
                batch_x, pool_idx = pool.sample(mb)
                x0 = batch_x.clone(); x0[0:1] = seed
                n_steps = np.random.randint(cfg["min_steps"], cfg["max_steps"])

                with torch.amp.autocast('cuda', enabled=cfg["mixed_precision"]):
                    x = x0; aux_acc = 0.0
                    for _ in range(n_steps):
                        x, aux = forward_step(model, x, None, target if isinstance(model, AdaNCA) else None, outer_step)
                        aux_acc = aux_acc + aux
                    mse = loss_fn(x, target)
                    loss = mse + cfg.get("aux_loss_weight", 0.0) * aux_acc

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_norm_clip"])
                scaler.step(optimizer); scaler.update(); scheduler.step()

                pool.commit(pool_idx, x)
                loss_log.append(float(loss.item()))
                break  # success, exit OOM retry loop

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if mb > 1:
                    mb = max(1, mb // 2)
                    print(f"\n⚠️ OOM in {model_name}: reducing mini-batch to {mb} and retrying this step…")
                    continue
                else:
                    raise

        if outer_step in cfg["viz_milestones"]:
            print(f"\n✓ Step {outer_step}: Capturing growth grid…")
            frames = []
            current = seed.clone(); ticks = 0
            model.eval()
            with torch.no_grad():
                max_s = max(cfg["growth_steps"])
                while ticks <= max_s:
                    if ticks in cfg["growth_steps"]:
                        frames.append(current[0].cpu().clone())
                    if isinstance(model, SpikingNCA): model._outer_step = outer_step
                    current = model(current, fire_rate=1.0, target=target if isinstance(model, AdaNCA) else None)
                    ticks += 1
            snapshots[outer_step] = frames
            model.train()

        if outer_step % 100 == 0:
            print(f"\rStep {outer_step:4d}/{cfg['train_steps']} | Loss: {loss.item():.6f} | log10: {np.log10(abs(loss.item())+1e-10):.3f}", end='')

    train_time = time.time() - start

    # Final eval + GIF
    print("\n\nFinal evaluation…")
    model.eval()
    with torch.no_grad():
        gif_frames = grow_from_seed(
            model, seed,
            target if isinstance(model, AdaNCA) else None,
            total_steps=240, capture_every=1, fire_rate=1.0, outer_step=cfg["train_steps"]
        )
        final_x = gif_frames[-1].unsqueeze(0).to(device)
        for _ in range(60):
            if isinstance(model, SpikingNCA): model._outer_step = cfg["train_steps"]
            final_x = model(final_x, fire_rate=1.0, target=target if isinstance(model, AdaNCA) else None)

        pred = to_rgba(final_x[0])
        final_loss = float(F.mse_loss(pred, target).item())
        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred[:3], target[:3])

    print(f"✓ Loss: {final_loss:.6f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | Time: {train_time/60:.2f} min")

    # Free some VRAM for next model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "parameters": params,
        "final_loss": final_loss,
        "psnr": psnr,
        "ssim": ssim,
        "train_time": train_time,
        "loss_history": loss_log,
        "final_image": final_x[0].cpu(),
        "snapshots": snapshots,
        "gif_frames": gif_frames,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }

# -----------------------
# ===== Main =====
# -----------------------
def run_comparison(cfg: Dict):
    print(f"\n{'#'*70}\n# NCA Extended Model Comparison\n{'#'*70}\n")

    # Load & save target
    if not os.path.exists(cfg["image_path"]):
        print("⚠️  Downloading fallback emoji…")
        import requests
        url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u1f697.png?raw=true'
        r = requests.get(url, timeout=10)
        cfg["image_path"] = '/kaggle/working/car.png'
        with open(cfg["image_path"], 'wb') as f:
            f.write(r.content)
        print("✓ Downloaded fallback icon")

    target_img = load_image(cfg["image_path"], cfg["img_size"])
    print(f"✓ Loaded: {target_img.shape}\n")

    target_t = torch.from_numpy(target_img).permute(2,0,1)
    plt.imsave(os.path.join(cfg["save_dir"], "target.png"),
               to_rgb(target_t, bg=cfg["viz_background"]).permute(1,2,0).clamp(0,1).numpy())

    all_results = []
    for model_name in cfg["models"]:
        try:
            result = train_model(model_name, target_img, cfg)
            all_results.append(result)

            model_dir = os.path.join(cfg["save_dir"], model_name.lower())
            os.makedirs(model_dir, exist_ok=True)

            # Final image
            plt.imsave(os.path.join(model_dir, "final.png"),
                       to_rgb(result["final_image"], bg=cfg["viz_background"]).permute(1,2,0).clamp(0,1).numpy())

            # Growth snapshots (grids) and optional GIFs per checkpoint
            for step, frames in result["snapshots"].items():
                if frames:
                    save_image_grid(
                        frames[:6],
                        [f"Step {s}" for s in cfg["growth_steps"][:6]],
                        f"{model_name} Growth (Checkpoint {step})",
                        os.path.join(model_dir, f"growth_{step:05d}.png"),
                        bg=cfg["viz_background"]
                    )
                    if cfg["save_checkpoint_gifs"]:
                        save_gif(frames, os.path.join(model_dir, f"growth_{step:05d}.gif"),
                                 bg=cfg["viz_background"], fps=15)

            # Final growth GIF (seed → 240 frames)
            if cfg["save_animations"] and result.get("gif_frames"):
                save_gif(result["gif_frames"], os.path.join(model_dir, "final_growth.gif"),
                         bg=cfg["viz_background"], fps=20)

            # Save model & metrics
            torch.save(result["model_state"], os.path.join(model_dir, "model.pth"))
            with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
                json.dump({
                    "model": result["model"],
                    "parameters": result["parameters"],
                    "final_loss": result["final_loss"],
                    "psnr": result["psnr"],
                    "ssim": result["ssim"],
                    "train_time_min": result["train_time"] / 60,
                }, f, indent=2)

            print(f"✓ Saved {model_name}\n")

        except torch.cuda.OutOfMemoryError as e:
            print(f"✗ OOM while training {model_name}: {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Failed {model_name}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("No models completed"); return

    # Comparisons
    print(f"\n{'='*70}\nGenerating Comparisons\n{'='*70}\n")

    final_images = [r["final_image"] for r in all_results]
    save_image_grid(final_images, [r["model"] for r in all_results],
                    "Final Results", os.path.join(cfg["save_dir"], "comparison_final.png"),
                    bg=cfg["viz_background"])

    # Loss curves
    plt.figure(figsize=(10,4))
    for r in all_results:
        plt.plot(np.log10(np.array(r["loss_history"]) + 1e-10), label=r["model"], alpha=0.85, linewidth=2)
    plt.xlabel("Step"); plt.ylabel("log10(Loss)"); plt.title("Training Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(cfg["save_dir"], "loss_curves.png"), dpi=120)
    plt.close()

    # Print metrics table
    print(f"\n{'='*70}\nPerformance Metrics\n{'='*70}")
    print(f"{'Model':<15} {'Params':<10} {'Loss':<12} {'PSNR':<10} {'SSIM':<8} {'Time(min)':<10}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['model']:<15} {r['parameters']:<10,} {r['final_loss']:<12.6f} "
              f"{r['psnr']:<10.2f} {r['ssim']:<8.4f} {r['train_time']/60:<10.2f}")

    best = max(all_results, key=lambda x: x["psnr"])
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": "TEJASKUMAR-REDDY-J",
        "device": str(device),
        "results": [{
            "model": r["model"],
            "parameters": r["parameters"],
            "final_loss": r["final_loss"],
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "train_time_min": r["train_time"] / 60,
        } for r in all_results],
        "best_model": best["model"],
    }
    with open(os.path.join(cfg["save_dir"], "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\n🏆 BEST: {best['model']} - PSNR: {best['psnr']:.2f} dB\n{'='*70}")
    print(f"✓ Done! Results: {cfg['save_dir']}")

if __name__ == "__main__":
    # Optional: run multiple images (comment out to run only one)
    image_paths = [
        CONFIG["image_path"]
    ]
    base_save = CONFIG["save_dir"]
    for path in image_paths:
        cfg = CONFIG.copy()
        cfg["image_path"] = path
        base_name = os.path.splitext(os.path.basename(path))[0]
        cfg["save_dir"] = os.path.join(base_save, base_name)
        os.makedirs(cfg["save_dir"], exist_ok=True)
        run_comparison(cfg)
        # free VRAM between datasets
        torch.cuda.empty_cache()
    print("\n✅ Complete!")