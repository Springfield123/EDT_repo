import json, numpy as np, matplotlib.pyplot as plt

AGR_FILE = "tmp.agr"
BAND_I = 207
BAND_J = 208
START_IDX = 150
END_IDX = 281
HALF_WINDOW_FIT = 18
RESIDUAL_TOL = 0.08
SHADE_ALPHA = 0.16
SHOW_CORE_WINDOW = True
CORE_SHADE_ALPHA = 0.10
X_WINDOW_MODE = "core"   # "core" or "edt"
X_WINDOW_MARGIN = 0.045
MASS_K3C60 = 837.95 / 63.0
FREQ_K3C60 = 1500.90
OUTSTEM = "fig_S7_NACT_K3C60_EDT_localfit_datamap"

def parse_agr(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    datasets, current = [], []
    for line in lines:
        if line.startswith("@") or line.startswith("#") or line.strip() == "":
            if current:
                datasets.append(np.array(current, dtype=float))
                current = []
            continue
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                current.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if current:
        datasets.append(np.array(current, dtype=float))
    return datasets

def add_panel_label(ax, label):
    ax.text(-0.10, 1.02, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", ha="left", va="bottom")

def zero_point_U_squared(mass_amu, freq_cm_inv):
    hbar = 1.0545718e-34
    amu_to_kg = 1.66054e-27
    c_cm_per_s = 2.99792458e10
    angstrom_to_m = 1e-10
    m = mass_amu * amu_to_kg
    omega = 2 * np.pi * freq_cm_inv * c_cm_per_s
    U2_m2 = hbar / (2 * m * omega)
    return U2_m2 / (angstrom_to_m ** 2)

def fit_local_parabola(k, y):
    coeffs = np.polyfit(k, y, 2)
    a, b, c = coeffs
    k_apex = -b / (2 * a)
    y_apex = np.polyval(coeffs, k_apex)
    return {"a": float(a), "b": float(b), "c": float(c),
            "k_apex": float(k_apex), "y_apex": float(y_apex), "coeffs": coeffs}

def extract_local_window(k, y, center_idx, half_window):
    i0 = max(0, center_idx - half_window)
    i1 = min(len(k), center_idx + half_window + 1)
    return k[i0:i1], y[i0:i1], i0, i1

def contiguous_true_region(mask, center_idx):
    if not mask[center_idx]:
        return center_idx, center_idx + 1
    left = center_idx
    right = center_idx
    while left - 1 >= 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return left, right + 1

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

datasets = parse_agr(AGR_FILE)
kpts = datasets[BAND_I][START_IDX:END_IDX, 0]
curv_i_raw = datasets[BAND_I][START_IDX:END_IDX, 1]
curv_j_raw = datasets[BAND_J][START_IDX:END_IDX, 1]

U2 = zero_point_U_squared(MASS_K3C60, FREQ_K3C60)
curv_i = curv_i_raw * U2
curv_j = curv_j_raw * U2

idx_i = int(np.argmin(curv_i))
idx_j = int(np.argmax(curv_j))
k_i_fit, y_i_fit, _, _ = extract_local_window(kpts, curv_i, idx_i, HALF_WINDOW_FIT)
k_j_fit, y_j_fit, _, _ = extract_local_window(kpts, curv_j, idx_j, HALF_WINDOW_FIT)
fit_i = fit_local_parabola(k_i_fit, y_i_fit)
fit_j = fit_local_parabola(k_j_fit, y_j_fit)

fit_curve_i_full = np.polyval(fit_i["coeffs"], kpts)
fit_curve_j_full = np.polyval(fit_j["coeffs"], kpts)
resid_i = np.abs(curv_i - fit_curve_i_full) / max(np.max(np.abs(curv_i)), 1e-14)
resid_j = np.abs(curv_j - fit_curve_j_full) / max(np.max(np.abs(curv_j)), 1e-14)

left_i, right_i = contiguous_true_region(resid_i <= RESIDUAL_TOL, idx_i)
left_j, right_j = contiguous_true_region(resid_j <= RESIDUAL_TOL, idx_j)
left_edt = min(left_i, left_j)
right_edt = max(right_i, right_j)

left_core = max(left_i, left_j)
right_core = min(right_i, right_j)
if right_core <= left_core:
    left_core, right_core = left_edt, right_edt

k_left = float(kpts[left_edt])
k_right = float(kpts[right_edt - 1])
k_left_core = float(kpts[left_core])
k_right_core = float(kpts[right_core - 1])
k_center = 0.5 * (fit_i["k_apex"] + fit_j["k_apex"])
edt_width = k_right - k_left
core_width = k_right_core - k_left_core

# shared x-limits for panels A and B
if X_WINDOW_MODE == "core":
    x_anchor_left, x_anchor_right = k_left_core, k_right_core
else:
    x_anchor_left, x_anchor_right = k_left, k_right
x_ab0 = max(float(kpts[0]), x_anchor_left - X_WINDOW_MARGIN)
x_ab1 = min(float(kpts[-1]), x_anchor_right + X_WINDOW_MARGIN)
panel_ab_mask = (kpts >= x_ab0) & (kpts <= x_ab1)
panel_b_ymax = max(
    0.18,
    float(np.max(resid_i[panel_ab_mask])) * 1.10,
    float(np.max(resid_j[panel_ab_mask])) * 1.10,
    RESIDUAL_TOL * 2.2,
)

result = {
    "band_i_display": BAND_I + 1,
    "band_j_display": BAND_J + 1,
    "selected_k_range": [float(kpts[0]), float(kpts[-1])],
    "k_apex_i": fit_i["k_apex"],
    "k_apex_j": fit_j["k_apex"],
    "k0_center": k_center,
    "curvature_apex_i": fit_i["y_apex"],
    "curvature_apex_j": fit_j["y_apex"],
    "edt_k_left": k_left,
    "edt_k_right": k_right,
    "edt_width": edt_width,
    "edt_core_k_left": k_left_core,
    "edt_core_k_right": k_right_core,
    "edt_core_width": core_width,
    "residual_tolerance": RESIDUAL_TOL,
    "panel_ab_xlim": [x_ab0, x_ab1],
    "panel_b_ylim": [0.0, panel_b_ymax],
}
with open(f"{OUTSTEM}.json", "w") as f:
    json.dump(result, f, indent=2)

fig = plt.figure(figsize=(6.25, 6.0))
gs = fig.add_gridspec(
    2, 2,
    left=0.12, right=0.96, top=0.965, bottom=0.07,
    hspace=0.28, wspace=0.22,
    height_ratios=[1.18, 1.0]
)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])
for ax, label in [(axA, "(a)"), (axB, "(b)"), (axC, "(c)"), (axD, "(d)")]:
    add_panel_label(ax, label)

# Panel A
axA.axvspan(k_left, k_right, color="0.85", alpha=SHADE_ALPHA)
if SHOW_CORE_WINDOW:
    axA.axvspan(k_left_core, k_right_core, color="0.75", alpha=CORE_SHADE_ALPHA)
axA.plot(kpts, curv_i, "o", ms=4.2, label=f"Band {BAND_I+1}")
axA.plot(kpts, curv_j, "o", ms=4.2, label=f"Band {BAND_J+1}")
k_dense_i = np.linspace(k_i_fit[0], k_i_fit[-1], 220)
k_dense_j = np.linspace(k_j_fit[0], k_j_fit[-1], 220)
axA.plot(k_dense_i, np.polyval(fit_i["coeffs"], k_dense_i), "--", lw=1.2)
axA.plot(k_dense_j, np.polyval(fit_j["coeffs"], k_dense_j), "--", lw=1.2)
axA.plot(fit_i["k_apex"], fit_i["y_apex"], "s", ms=4.0)
axA.plot(fit_j["k_apex"], fit_j["y_apex"], "s", ms=4.0)
axA.set_xlim(x_ab0, x_ab1)
axA.set_xlabel("k")
axA.set_ylabel(r"d$^2$E/dU$^2$ (eV/$\mathrm{\AA}^2$)")
axA.grid(True, alpha=0.30)
axA.legend(frameon=False, loc="upper right")
axA.text(0.03, 0.96,
         rf"local EDT region" "\n" rf"$\Delta k_{{\mathrm{{EDT}}}}={edt_width:.3f}$" "\n" rf"zoomed $k$-window",
         transform=axA.transAxes, ha="left", va="top", fontsize=7.2)

# Panel B
axB.axvspan(k_left, k_right, color="0.85", alpha=SHADE_ALPHA)
if SHOW_CORE_WINDOW:
    axB.axvspan(k_left_core, k_right_core, color="0.75", alpha=CORE_SHADE_ALPHA)
axB.axvspan(float(kpts[left_i]), float(kpts[right_i - 1]), color="0.70", alpha=0.05)
axB.axvspan(float(kpts[left_j]), float(kpts[right_j - 1]), color="0.55", alpha=0.05)
axB.axhline(RESIDUAL_TOL, ls="--", lw=1.0, color="0.35")
axB.plot(kpts, resid_i, lw=1.2, label=f"Band {BAND_I+1}")
axB.plot(kpts, resid_j, lw=1.2, label=f"Band {BAND_J+1}")
axB.axvline(fit_i["k_apex"], ls=":", lw=0.9, alpha=0.9)
axB.axvline(fit_j["k_apex"], ls=":", lw=0.9, alpha=0.9)
axB.set_xlim(x_ab0, x_ab1)
axB.set_ylim(0.0, panel_b_ymax)
axB.set_xlabel("k")
axB.set_ylabel("normalized residual")
axB.grid(True, alpha=0.30)
axB.legend(frameon=False, loc="upper left")
axB.text(0.03, 0.94, "EDT threshold", transform=axB.transAxes,
         ha="left", va="top", fontsize=7.0, color="0.30")
axB.text(0.03, 0.84, "light spans: band-local validity\ndark span: shared core", transform=axB.transAxes,
         ha="left", va="top", fontsize=6.6, color="0.35")

# Panel C data-driven map
qx = np.linspace(kpts.min(), kpts.max(), 280)
qy = np.linspace(-0.45, 0.45, 220)
fit_mag_1d = np.sqrt(fit_curve_i_full**2 + fit_curve_j_full**2)
soft_i = np.clip(1.0 - resid_i / RESIDUAL_TOL, 0.0, 1.0)
soft_j = np.clip(1.0 - resid_j / RESIDUAL_TOL, 0.0, 1.0)
mask_soft = np.sqrt(soft_i * soft_j)
pair_mag_1d = fit_mag_1d * mask_soft
pair_mag_1d = pair_mag_1d / max(np.max(pair_mag_1d), 1e-14)
pair_mag_interp = np.interp(qx, kpts, pair_mag_1d)
sigma_q = 0.12
transverse = np.exp(-(qy**2) / (2 * sigma_q**2))
mag_map = np.outer(transverse, pair_mag_interp)
axC.imshow(mag_map, extent=[qx.min(), qx.max(), qy.min(), qy.max()], origin="lower", aspect="auto")
axC.axvspan(k_left, k_right, color="white", alpha=0.10)
if SHOW_CORE_WINDOW:
    axC.axvspan(k_left_core, k_right_core, color="white", alpha=0.18)
axC.axvline(k_center, color="white", lw=1.0, ls="--", alpha=0.9)
axC.text(k_center, qy.max()*0.92, r"$k_0$", color="white", ha="center", va="top", fontsize=7.5)
axC.text(k_left_core + 0.5*core_width, qy.min()+0.05, r"$\Delta k_{\mathrm{core}}$", color="white", ha="center", va="bottom", fontsize=7.3)
axC.set_xlim(qx.min(), qx.max()); axC.set_ylim(qy.min(), qy.max())
axC.set_xticks([]); axC.set_yticks([])

# Panel D interpretive loop field
KX, KY = np.meshgrid(qx, qy)
q0 = 0.0
sigma_k = max(core_width/2.0 if SHOW_CORE_WINDOW else edt_width/2.0, 0.03)
dx = -(KY - q0) / ((KX - k_center)**2 + (KY - q0)**2 + 0.02)
dy =  (KX - k_center) / ((KX - k_center)**2 + (KY - q0)**2 + 0.02)
envelope = np.exp(-((KX - k_center)**2/(2*sigma_k**2) + (KY - q0)**2/(2*(sigma_q*1.15)**2)))
dx_loc = dx * envelope
dy_loc = dy * envelope
axD.streamplot(qx, qy, dx_loc, dy_loc, density=1.05, linewidth=0.85, arrowsize=0.9)
axD.axvspan(k_left, k_right, color="0.85", alpha=0.14)
if SHOW_CORE_WINDOW:
    axD.axvspan(k_left_core, k_right_core, color="0.75", alpha=CORE_SHADE_ALPHA)
axD.axvline(k_center, color="0.35", lw=0.9, ls="--")
axD.text(k_center, qy.max()*0.92, r"$k_0$", ha="center", va="top", fontsize=7.5)
axD.text(k_left + 0.5*edt_width, qy.min()+0.05, r"$\Delta k_{\mathrm{EDT}}$", ha="center", va="bottom", fontsize=7.5)
if SHOW_CORE_WINDOW:
    axD.text(k_left_core + 0.5*core_width, qy.max()-0.05, r"$\Delta k_{\mathrm{core}}$", ha="center", va="top", fontsize=7.1)
axD.set_xlim(qx.min(), qx.max()); axD.set_ylim(qy.min(), qy.max())
axD.set_xticks([]); axD.set_yticks([])

fig.savefig(f"{OUTSTEM}.pdf", bbox_inches="tight")
fig.savefig(f"{OUTSTEM}.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", OUTSTEM + ".pdf")
print("Saved:", OUTSTEM + ".png")
print("Saved:", OUTSTEM + ".json")
print(json.dumps(result, indent=2))
print("Shared x-window (A,B):", [x_ab0, x_ab1])

