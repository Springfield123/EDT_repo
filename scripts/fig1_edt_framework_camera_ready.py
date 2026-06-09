
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Ellipse
import matplotlib as mpl

# -----------------------------
# Global style (Nature-like)
# -----------------------------
mpl.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(10.6, 6.7))
gs = fig.add_gridspec(
    2, 3,
    left=0.052, right=0.985, bottom=0.075, top=0.865,
    hspace=0.34, wspace=0.28
)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

def panel_label(ax, text):
    ax.text(-0.10, 1.05, text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

def hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# -------------------------------------------------
# Top banner
# -------------------------------------------------
fig.text(0.07, 0.925, "Degeneracy", fontsize=11, fontweight='bold')
fig.text(0.26, 0.925, "Entanglement", fontsize=11, fontweight='bold')
fig.text(0.47, 0.925, "Compilation", fontsize=11, fontweight='bold')
fig.text(0.69, 0.925, "Operational resources", fontsize=11, fontweight='bold')
for x0, x1 in [(0.17, 0.245), (0.39, 0.455), (0.61, 0.675)]:
    arr = FancyArrowPatch((x0, 0.928), (x1, 0.928),
                          transform=fig.transFigure,
                          arrowstyle='-|>', mutation_scale=11,
                          lw=0.9, color='black')
    fig.add_artist(arr)
#fig.text(0.052, 0.965,
#         "Figure 1 | Degeneracy-driven entanglement as a programmable quantum resource across computation, measurement, and sensing",
#         fontsize=10.2, fontweight='bold', ha='left')

# -----------------------------
# (A) Degeneracy → Entanglement
# -----------------------------
ax = axes[0]
x = np.linspace(-1.15, 1.15, 400)
y1 = 0.65 * x**2 + 0.12
y2 = -0.65 * x**2 - 0.12
ax.plot(x, y1, lw=1.8)
ax.plot(x, y2, lw=1.8)

# degeneracy region
ax.add_patch(Ellipse((0, 0), 0.55, 0.36, fill=False, linestyle='--', linewidth=1.1))
ax.text(0, 0.38, "near-degeneracy\n/ curvature source", ha='center', va='center', fontsize=8)

# curvature / entanglement arrows
for ang in [35, 145, 215, 325]:
    r = np.deg2rad(ang)
    x0, y0 = 0.36*np.cos(r), 0.24*np.sin(r)
    x1, y1a = 0.55*np.cos(r+0.25), 0.34*np.sin(r+0.25)
    ax.annotate("", xy=(x1, y1a), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', lw=0.9))
ax.text(-0.92, 0.87, "band / surface\ncrossing", fontsize=8, ha='left')
ax.text(0.62, -0.93, "entanglement\ngeneration", fontsize=8, ha='center')

#ax.set_title("Degeneracy-driven entanglement", pad=6)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.15, 1.15)
hide_axes(ax)
panel_label(ax, "(A)")

# -----------------------------
# (B) EDT manifold
# -----------------------------
ax = axes[1]
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), linestyle='--', lw=1.0, alpha=0.8)

# bright manifold ridge
t = np.linspace(-1.1, 1.05, 300)
xm = 0.82*np.cos(t)
ym = 0.46*np.sin(t) + 0.08*np.sin(2*t)
ax.plot(xm, ym, lw=2.2)
ax.plot(0.92*xm, 0.92*ym, lw=1.0, alpha=0.7)

# loop arrows along manifold
for s in [40, 120, 210]:
    i0 = s
    i1 = s + 18
    ax.annotate("", xy=(xm[i1], ym[i1]), xytext=(xm[i0], ym[i0]),
                arrowprops=dict(arrowstyle='->', lw=0.9))
ax.text(-0.9, 0.8, "full Hilbert space", fontsize=8, ha='left')
ax.text(0.18, 0.28, "EDT manifold", fontsize=8, ha='left')
ax.text(-0.62, -0.62, "loop-stabilized\ninterference", fontsize=8, ha='left')

ax.set_title("Structured entanglement manifold", pad=6)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.15, 1.15)
hide_axes(ax)
panel_label(ax, "(B)")

# -----------------------------
# (C) Compilation pipeline
# -----------------------------
ax = axes[2]
hide_axes(ax)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

boxes = [
    (0.05, 0.36, 0.26, 0.24, "Schur\nsectors"),
    (0.37, 0.36, 0.26, 0.24, "CG\ncoupling"),
    (0.70, 0.36, 0.24, 0.24, "AM-TTN /\ncircuit"),
]
for x0, y0, w, h, txt in boxes:
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=1.0))
    ax.text(x0 + w/2, y0 + h/2, txt, ha='center', va='center', fontsize=8)

for (x0, x1) in [(0.26, 0.39), (0.61, 0.73)]:
    ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                arrowprops=dict(arrowstyle='-|>', lw=0.95))

# source input above pipeline
ax.add_patch(Rectangle((0.12, 0.68), 0.76, 0.24, fill=False, linestyle='--', linewidth=1.4))
ax.text(0.50, 0.80,
        "degeneracy / curvature input\n(NACT, band crossings, phonons)",
        ha='center', va='center', fontsize=9)
ax.annotate("", xy=(0.50, 0.68), xytext=(0.50, 0.60),
            arrowprops=dict(arrowstyle='-|>', lw=1.4))
#ax.text(0.50, 0.63, "symmetry-resolved compilation", ha='center', fontsize=8.5)

ax.text(0.49, 0.16, "physics → symmetry → architecture", ha='center', fontsize=8)
#ax.set_title("Compilation pipeline", pad=6)
panel_label(ax, "(C)")

# -----------------------------
# (D) Computation (refined)
# -----------------------------
ax = axes[3]
hide_axes(ax)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-1.55, 1.6)

rng = np.random.default_rng(3)
N = 8
th = np.linspace(0, 2*np.pi, N, endpoint=False)
xs = 0.72*np.cos(th)
ys = 0.72*np.sin(th)

# left: unstructured graph
for i in range(N):
    for j in range(i+1, N):
        if rng.random() < 0.55:
            ax.plot([xs[i]-1.2, xs[j]-1.2], [ys[i], ys[j]],
                    lw=0.9, alpha=0.35)
ax.scatter(xs-1.2, ys, s=20, zorder=3)
ax.text(-1.2, 1.17, "unstructured\ncomputation", ha='center', va='center', fontsize=8)

# right: same graph faint
for i in range(N):
    for j in range(i+1, N):
        ax.plot([xs[i]+1.2, xs[j]+1.2], [ys[i], ys[j]],
                lw=0.7, alpha=0.10)

# highlighted EDT path/subspace
path = [7, 0, 1, 2, 3]
for i in range(len(path)-1):
    a, b = path[i], path[i+1]
    ax.plot([xs[a]+1.2, xs[b]+1.2], [ys[a], ys[b]],
            lw=2.2, solid_capstyle='round')
highlight_nodes = path
other_nodes = [i for i in range(N) if i not in highlight_nodes]
ax.scatter(xs[other_nodes]+1.2, ys[other_nodes], s=18, alpha=0.35, zorder=3)
ax.scatter(xs[highlight_nodes]+1.2, ys[highlight_nodes], s=28, zorder=4)
ax.add_patch(Ellipse((1.2, 0.0), 1.85, 0.95, fill=False, linestyle='--', linewidth=0.9))
ax.text(1.2, 1.17, "EDT-guided\ncomputation", ha='center', va='center', fontsize=8)

# central arrow
ax.annotate("Hilbert-space\ncompression", xy=(0.52, 0.0), xytext=(-0.15, 0.0),
            ha='center', va='center', fontsize=8,
            arrowprops=dict(arrowstyle='-|>', lw=0.9))

# mini labels
ax.text(-1.2, -1.13, "uniform search /\nbroad pathway load", ha='center', fontsize=8)
ax.text(1.2, -1.13, "task-relevant\npathway concentration", ha='center', fontsize=8)

#ax.set_title("Computation via structured resources", pad=6)
panel_label(ax, "(D)")

# -----------------------------
# (E) Measurement (EDT shadows)
# -----------------------------
ax = axes[4]
hide_axes(ax)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

rng = np.random.default_rng(5)
pts = rng.uniform(0.06, 0.94, size=(50, 2))
ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.22)
cluster = np.column_stack([0.68 + 0.10*rng.random(22), 0.64 + 0.14*rng.random(22)])
ax.scatter(cluster[:, 0], cluster[:, 1], s=16, zorder=3)

# cone arrows
for x0 in [0.18, 0.30, 0.42]:
    ax.annotate("", xy=(x0, 0.50), xytext=(x0, 0.87),
                arrowprops=dict(arrowstyle='->', lw=0.8, alpha=0.45))
for x0 in [0.73, 0.79, 0.85]:
    ax.annotate("", xy=(x0, 0.72), xytext=(x0, 0.94),
                arrowprops=dict(arrowstyle='->', lw=0.9))
ax.text(0.24, 0.91, "uniform\nsampling", ha='center', fontsize=8)
ax.text(0.78, 0.95, "EDT-aligned\nsampling", ha='center', fontsize=8)
ax.text(0.17, 0.08, "full operator space", fontsize=8, ha='left')
ax.text(0.61, 0.48, "restricted\noperator sector", fontsize=8, ha='left')

#ax.set_title("Measurement (aligned shadows)", pad=6)
panel_label(ax, "(E)")

# -----------------------------
# (F) NV sensing
# -----------------------------
ax = axes[5]
hide_axes(ax)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# NV center
diamond = np.array([[0.50, 0.22], [0.80, 0.50], [0.50, 0.78], [0.20, 0.50]])
ax.plot(*diamond.T, lw=1.0)
ax.plot([diamond[-1,0], diamond[0,0]], [diamond[-1,1], diamond[0,1]], lw=1.0)
ax.add_patch(Circle((0.50, 0.50), 0.09, fill=False, linewidth=1.0))
ax.text(0.50, 0.50, "NV", ha='center', va='center', fontsize=8)

# signal arrows and oscillation
ax.annotate("", xy=(0.43, 0.58), xytext=(0.18, 0.82),
            arrowprops=dict(arrowstyle='-|>', lw=1.0))
ax.annotate("", xy=(0.58, 0.42), xytext=(0.84, 0.18),
            arrowprops=dict(arrowstyle='-|>', lw=1.0))
t = np.linspace(0, 1, 200)
ax.plot(0.10 + 0.25*t, 0.18 + 0.07*np.sin(8*np.pi*t), lw=1.2)
ax.plot(0.65 + 0.22*t, 0.80 + 0.03*np.sin(8*np.pi*t), lw=1.2)
ax.text(0.18, 0.88, "external field", fontsize=8, ha='left')
ax.text(0.63, 0.88, "selective\nresponse", fontsize=8, ha='left')
ax.text(0.10, 0.07, "coherence + signal\nlocalization", fontsize=8, ha='left')

#ax.set_title("Sensing (NV resource localization)", pad=6)
panel_label(ax, "(F)")

# subtle row guides
for y in [0.497]:
    fig.lines.append(plt.Line2D([0.055, 0.985], [y, y],
                                transform=fig.transFigure,
                                lw=0.4, alpha=0.25))

plt.savefig("fig1_edt_framework_camera_ready.pdf", bbox_inches='tight')
plt.savefig("fig1_edt_framework_camera_ready.png", bbox_inches='tight')
plt.show()
