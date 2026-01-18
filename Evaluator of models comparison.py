# ============================================================
# NCA EVALUATION - Run AFTER training completes
# ============================================================

import os, json, torch, numpy as np
import matplotlib.pyplot as plt
import torch.nn. functional as F

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR = "/kaggle/working/nca_comparison_extended/car"  # YOUR TRAINING OUTPUT
EVAL_DIR = "/kaggle/working/nca_evaluation"
GROWTH_STEPS = 300
DRIFT_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EVAL_DIR, exist_ok=True)

COLORS = {
    'Canonical': '#1f77b4', 'MeshNCA': '#ff7f0e', 'DiffNCA': '#2ca02c',
    'AdaNCA':  '#d62728', 'LightAttnNCA': '#9467bd', 'SpikingNCA': '#8c564b',
    'HierNCA': '#e377c2', 'VarNCA': '#7f7f7f', 'GraphNCA': '#bcbd22',
}

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
    seed[: , 3:, size//2, size//2] = 1.0
    return seed

# ============================================================
# LOAD TARGET & SUMMARY
# ============================================================
print(f"Loading from: {TRAIN_DIR}")

from PIL import Image
target_img = Image.open(f"{TRAIN_DIR}/target.png").convert("RGBA")
target = torch.from_numpy(np.float32(target_img)/255.0).permute(2,0,1).to(device)
size = target.shape[-1]
print(f"Target:  {size}x{size}")

with open(f"{TRAIN_DIR}/summary.json") as f:
    summary = json.load(f)
model_names = [r['model'] for r in summary['results']]
print(f"Models: {model_names}")

# ============================================================
# EVALUATE ALL MODELS
# ============================================================
results = {}

for name in model_names: 
    model_dir = f"{TRAIN_DIR}/{name.lower()}"
    if not os.path. exists(f"{model_dir}/model.pth"):
        print(f"Skip {name}"); continue
    
    print(f"\n{'='*50}\nEvaluating:  {name}\n{'='*50}")
    
    # Load model
    model = make_model(name, CONFIG).to(device)
    model.load_state_dict(torch. load(f"{model_dir}/model.pth", map_location=device))
    model.eval()
    
    with open(f"{model_dir}/metrics.json") as f:
        tm = json.load(f)
    
    m = {'name': name, 'params': tm['parameters'], 'train_time': tm['train_time_min']*60}
    
    # ===== GROWTH TRAJECTORY =====
    seed = make_seed(size, 16, device)
    x = seed.clone()
    psnr_hist, hidden_energy, rgb_energy = [], [], []
    best_psnr, best_step, best_state = -999, 0, None
    
    with torch.no_grad():
        for step in range(GROWTH_STEPS):
            if hasattr(model, '_outer_step'): model._outer_step = 10000
            x = model(x, fire_rate=1.0, target=target if name=='AdaNCA' else None)
            
            p = calc_psnr(x[0,: 4], target[:4])
            psnr_hist.append(p)
            
            # For HSER
            hidden_energy.append((x[0, 4:]**2).mean().item() if x. shape[1] > 4 else 0)
            rgb_energy.append((x[0, : 3]**2).mean().item())
            
            if p > best_psnr: 
                best_psnr, best_step, best_state = p, step, x. clone()
    
    m['psnr_hist'] = psnr_hist
    m['best_psnr'] = best_psnr
    m['best_step'] = best_step
    m['final_psnr'] = psnr_hist[-1]
    m['fre'] = F.mse_loss(best_state[0,:4], target[:4]).item()
    m['ssim'] = calc_ssim(best_state[0,:3], target[:3])
    
    # ===== TεC (Time-to-ε Convergence) =====
    m['tec_25'] = next((i for i,p in enumerate(psnr_hist) if p >= 25), -1)
    m['tec_30'] = next((i for i,p in enumerate(psnr_hist) if p >= 30), -1)
    m['tec_35'] = next((i for i,p in enumerate(psnr_hist) if p >= 35), -1)
    
    # ===== CSI (Convergence Stability Index) =====
    post_start = min(best_step, len(psnr_hist)-50)
    m['csi'] = np.var(psnr_hist[post_start:post_start+50])
    
    # ===== SSD (Steady-State Drift) =====
    ssd_hist = []
    x_drift = best_state.clone()
    with torch.no_grad():
        for _ in range(DRIFT_STEPS):
            x_prev = x_drift.clone()
            x_drift = model(x_drift, fire_rate=1.0, target=target if name=='AdaNCA' else None)
            ssd_hist.append(((x_drift - x_prev)**2).mean().item())
    m['ssd'] = np.mean(ssd_hist)
    m['ssd_hist'] = ssd_hist
    
    # ===== BRS (Basin Robustness Score) =====
    # Noise
    x_n = best_state.clone() + torch.randn_like(best_state) * 0.3
    with torch.no_grad():
        for _ in range(100):
            x_n = model(x_n, fire_rate=1.0, target=target if name=='AdaNCA' else None)
    m['brs_noise'] = calc_psnr(x_n[0,:4], target[:4])
    
    # Erase
    x_e = best_state.clone()
    x_e[: , : , size//4:size//2, size//4:size//2] = 0
    with torch.no_grad():
        for _ in range(100):
            x_e = model(x_e, fire_rate=1.0, target=target if name=='AdaNCA' else None)
    m['brs_erase'] = calc_psnr(x_e[0,:4], target[:4])
    m['brs'] = (m['brs_noise'] + m['brs_erase']) / 2
    
    # ===== EFFICIENCY METRICS =====
    m['per'] = m['best_psnr'] / np.log(m['params'] + 1)  # Parameter Efficiency Ratio
    m['tnq'] = m['best_psnr'] / (m['train_time'] + 1e-6)  # Time-Normalized Quality
    tec = m['tec_30'] if m['tec_30'] > 0 else GROWTH_STEPS
    m['nes'] = m['best_psnr'] / (m['params'] * tec) * 1e6  # NCA Efficiency Score (NOVEL!)
    
    # ===== HSER (Hidden-State Energy Ratio) =====
    avg_hidden = np.mean(hidden_energy[-50:])
    avg_rgb = np.mean(rgb_energy[-50:])
    m['hser'] = avg_hidden / (avg_rgb + 1e-10)
    
    results[name] = m
    print(f"  PSNR*: {m['best_psnr']:.2f} | TεC30: {m['tec_30']} | CSI: {m['csi']:.4f}")
    print(f"  SSD: {m['ssd']:.2e} | BRS: {m['brs']:.2f} | NES: {m['nes']:.4f} | HSER: {m['hser']:.4f}")

# ============================================================
# INDIVIDUAL MODEL PLOTS
# ============================================================
for name, m in results.items():
    out = f"{EVAL_DIR}/{name.lower()}"
    os.makedirs(out, exist_ok=True)
    c = COLORS.get(name, 'gray')
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{name} - Full Evaluation', fontweight='bold', fontsize=14)
    
    # 1. PSNR trajectory
    axes[0,0]. plot(m['psnr_hist'], color=c, lw=2)
    axes[0,0].axhline(m['best_psnr'], color='r', ls='--', alpha=0.5)
    axes[0,0].scatter([m['best_step']], [m['best_psnr']], c='red', s=100, marker='*')
    for th in [25,30,35]:  axes[0,0].axhline(th, color='gray', ls=':', alpha=0.3)
    axes[0,0].set_title(f"PSNR (Best: {m['best_psnr']:.2f}dB @ {m['best_step']})")
    axes[0,0].set_xlabel('Step'); axes[0,0].grid(True, alpha=0.3)
    
    # 2. TεC
    tecs = [m['tec_25'] if m['tec_25']>0 else 300, m['tec_30'] if m['tec_30']>0 else 300, m['tec_35'] if m['tec_35']>0 else 300]
    axes[0,1].bar(['25dB','30dB','35dB'], tecs, color=['#3498db','#2ecc71','#e74c3c'])
    axes[0,1]. set_title('Time-to-ε (TεC) ↓better')
    
    # 3. CSI
    post = m['psnr_hist'][m['best_step']: ] if m['best_step'] < len(m['psnr_hist'])-20 else m['psnr_hist'][-50:]
    axes[0,2].plot(post, color=c, lw=2)
    axes[0,2].axhline(np.mean(post), color='r', ls='--')
    axes[0,2]. fill_between(range(len(post)), np.mean(post)-np.std(post), np.mean(post)+np.std(post), alpha=0.2, color=c)
    axes[0,2].set_title(f"Stability (CSI={m['csi']:.4f}) ↓better")
    
    # 4. SSD
    axes[0,3].semilogy(m['ssd_hist'], color=c, lw=2)
    axes[0,3].axhline(m['ssd'], color='r', ls='--')
    axes[0,3]. set_title(f"Drift (SSD={m['ssd']:.2e}) ↓better")
    
    # 5. BRS
    axes[1,0]. bar(['Original','Noise','Erase','Combined'], 
                  [m['best_psnr'], m['brs_noise'], m['brs_erase'], m['brs']], 
                  color=[c,'#9b59b6','#1abc9c','#e74c3c'])
    axes[1,0].set_title('Basin Robustness (BRS) ↑better')
    
    # 6. Efficiency
    axes[1,1].bar(['PER','TNQ×100','NES'], [m['per'], m['tnq']*100, m['nes']], color=['#3498db','#2ecc71','#e74c3c'])
    axes[1,1]. set_title('Efficiency Metrics ↑better')
    
    # 7. HSER
    axes[1,2].bar(['HSER'], [m['hser']], color=c)
    axes[1,2]. set_title(f"Hidden Energy Ratio (HSER={m['hser']:.4f})")
    
    # 8. Summary
    axes[1,3]. axis('off')
    txt = f"""
{name} Summary
{'─'*30}
ACCURACY
  PSNR*: {m['best_psnr']:.2f} dB
  FRE: {m['fre']:.6f}
  SSIM: {m['ssim']:.4f}

CONVERGENCE
  TεC(25/30/35): {m['tec_25']}/{m['tec_30']}/{m['tec_35']}
  CSI: {m['csi']:.4f}

STABILITY
  SSD: {m['ssd']:.2e}
  BRS: {m['brs']:.2f} dB

EFFICIENCY
  PER: {m['per']:.3f}
  TNQ: {m['tnq']:.4f}
  NES: {m['nes']:.4f}
  HSER: {m['hser']:.4f}
"""
    axes[1,3].text(0.05, 0.95, txt, fontsize=10, family='monospace', va='top', transform=axes[1,3].transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{out}/dashboard.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved {name} plots")

# ============================================================
# COMBINED COMPARISON
# ============================================================
names = list(results.keys())
colors = [COLORS.get(n, 'gray') for n in names]

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.suptitle('NCA Model Comparison - All Metrics', fontweight='bold', fontsize=16)

# Row 1
for n,m in results.items(): axes[0,0].plot(m['psnr_hist'], color=COLORS.get(n), lw=1.5, label=n)
axes[0,0].set_title('PSNR Trajectories'); axes[0,0].legend(fontsize=7); axes[0,0].grid(True, alpha=0.3)

axes[0,1].barh(names, [results[n]['best_psnr'] for n in names], color=colors)
axes[0,1].axvline(30, color='r', ls='--'); axes[0,1].set_title('Best PSNR* ↑')

axes[0,2].barh(names, [results[n]['tec_30'] if results[n]['tec_30']>0 else 300 for n in names], color=colors)
axes[0,2].set_title('TεC(30dB) ↓')

axes[0,3].barh(names, [results[n]['csi'] for n in names], color=colors)
axes[0,3].set_title('CSI ↓')

# Row 2
axes[1,0].barh(names, [results[n]['ssd'] for n in names], color=colors)
axes[1,0].set_xscale('log'); axes[1,0].set_title('SSD ↓')

axes[1,1].barh(names, [results[n]['brs'] for n in names], color=colors)
axes[1,1].set_title('BRS ↑')

axes[1,2].barh(names, [results[n]['per'] for n in names], color=colors)
axes[1,2].set_title('PER ↑')

axes[1,3].barh(names, [results[n]['nes'] for n in names], color=colors)
axes[1,3].set_title('★ NES (Novel) ↑')

# Row 3
axes[2,0]. barh(names, [results[n]['tnq'] for n in names], color=colors)
axes[2,0].set_title('TNQ ↑')

axes[2,1].barh(names, [results[n]['hser'] for n in names], color=colors)
axes[2,1].set_title('HSER')

axes[2,2].barh(names, [results[n]['ssim'] for n in names], color=colors)
axes[2,2].set_title('SSIM ↑')

# Summary
axes[2,3].axis('off')
best = {
    'PSNR*':  max(results.values(), key=lambda x: x['best_psnr'])['name'],
    'NES': max(results.values(), key=lambda x: x['nes'])['name'],
    'SSD': min(results.values(), key=lambda x: x['ssd'])['name'],
    'BRS':  max(results.values(), key=lambda x: x['brs'])['name'],
    'CSI': min(results.values(), key=lambda x: x['csi'])['name'],
}
txt = "BEST BY METRIC\n" + "─"*25 + "\n" + "\n".join([f"{k}:  {v}" for k,v in best.items()])
txt += f"\n\n🏆 OVERALL:  {best['NES']}"
axes[2,3].text(0.1, 0.5, txt, fontsize=12, family='monospace', va='center', transform=axes[2,3].transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{EVAL_DIR}/comparison.png", dpi=150)
plt.close()

# ============================================================
# SAVE JSON & PRINT TABLE
# ============================================================
out_data = {n: {k: v for k,v in m. items() if 'hist' not in k} for n,m in results.items()}
with open(f"{EVAL_DIR}/metrics.json", 'w') as f:
    json.dump(out_data, f, indent=2)

print(f"\n{'='*110}")
print(f"{'Model':<15} {'PSNR*':<8} {'TεC30': <7} {'CSI':<9} {'SSD':<11} {'BRS':<8} {'PER':<8} {'NES':<8} {'HSER':<8}")
print(f"{'='*110}")
for n,m in results.items():
    print(f"{n:<15} {m['best_psnr']:<8.2f} {m['tec_30']:<7} {m['csi']:<9.4f} {m['ssd']:<11.2e} {m['brs']: <8.2f} {m['per']:<8.3f} {m['nes']:<8.4f} {m['hser']:<8.4f}")

print(f"\n🏆 BEST (NES): {best['NES']}")
print(f"✓ Saved to:  {EVAL_DIR}")