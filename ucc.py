import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# ── DATA: District-level BAF means (from Table S2) ──────────
baf_lulc = [0.21, 0.11, 0.22, 0.16, 0.14, 0.08, 0.05, 0.04,
            0.16, 0.04, 0.04, 0.06, 0.09, 0.06, 0.09, 0.11,
            0.07, 0.20, 0.19, 0.09, 0.13, 0.17]

baf_gsmp = [0.30, 0.23, 0.32, 0.25, 0.26, 0.16, 0.10, 0.13,
            0.23, 0.09, 0.08, 0.11, 0.17, 0.16, 0.18, 0.19,
            0.16, 0.31, 0.26, 0.21, 0.22, 0.19]

# ── DATA: District-level CCI means (from Table S3) ──────────
cci_lulc = [0.108, 0.119, 0.122, 0.236, 0.153, 0.101, 0.095,
            0.083, 0.117, 0.078, 0.098, 0.118, 0.089, 0.106,
            0.156, 0.184, 0.114, 0.131, 0.228, 0.111, 0.108, 0.227]

cci_gsmp = [0.153, 0.195, 0.187, 0.286, 0.233, 0.144, 0.145,
            0.134, 0.175, 0.112, 0.148, 0.171, 0.154, 0.177,
            0.223, 0.292, 0.157, 0.240, 0.300, 0.172, 0.180, 0.274]

# ── DATA: Urban density proxy (neighborhoods/hectare) ────────
n_neighborhoods = [27,21,12,20,29,14,14,13,9,10,17,14,
                   13,20,19,9,14,18,13,20,13,12]
areas_ha = [4656.4,4700.2,2921.8,6156.8,5316.8,2137.8,1532.6,
            1315.9,1974.08,818.5,1203.08,1600.7,1287.3,1455.1,
            2816.2,1651.4,825.27,3786.1,2036.1,2357.9,5153.03,5900.1]
density = [n / a for n, a in zip(n_neighborhoods, areas_ha)]

# ============================================================
# FUNCTION 1: Gini Coefficient
# ============================================================
def gini_coefficient(values):
    """
    Calculate Gini coefficient for distributional equity.
    G = 0: perfect equality; G = 1: maximum inequality.
    Formula:
    G = (2 * sum(rank_i * y_i)) / (n * sum(y_i)) - (n + 1) / n
    """
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    ranks = np.arange(1, n + 1)
    G = (2 * np.sum(ranks * arr)) / (n * np.sum(arr)) - (n + 1) / n
    return round(G, 3)

# ============================================================
# RESULTS: Gini Coefficients
# ============================================================
print("=" * 55)
print("GINI COEFFICIENTS (distributional inequality)")
print("=" * 55)
print(f"BAF  LULC-Current : G = {gini_coefficient(baf_lulc)}")
print(f"BAF  GSMP         : G = {gini_coefficient(baf_gsmp)}")
print(f"  Change: {(gini_coefficient(baf_gsmp)-gini_coefficient(baf_lulc))/gini_coefficient(baf_lulc)*100:.1f}%")
print()
print(f"CCI  LULC-Current : G = {gini_coefficient(cci_lulc)}")
print(f"CCI  GSMP         : G = {gini_coefficient(cci_gsmp)}")
print(f"  Change: {(gini_coefficient(cci_gsmp)-gini_coefficient(cci_lulc))/gini_coefficient(cci_lulc)*100:.1f}%")

# ============================================================
# RESULTS: Wilcoxon Signed-Rank Test
# ============================================================
print()
print("=" * 55)
print("WILCOXON SIGNED-RANK TEST (paired, non-parametric)")
print("H0: No difference between LULC-Current and GSMP")
print("=" * 55)
w_baf, p_baf = stats.wilcoxon(baf_lulc, baf_gsmp)
w_cci, p_cci = stats.wilcoxon(cci_lulc, cci_gsmp)
print(f"BAF : W = {w_baf:.0f},  p = {p_baf:.6f}  {'***' if p_baf<0.001 else ''}")
print(f"CCI : W = {w_cci:.0f},  p = {p_cci:.6f}  {'***' if p_cci<0.001 else ''}")
print("Note: W=0 means all districts improved (no ties/negative differences)")

# ============================================================
# RESULTS: Pearson Correlation
# ============================================================
print()
print("=" * 55)
print("PEARSON CORRELATION (urban density vs. index values)")
print("Density proxy = neighborhoods per hectare")
print("=" * 55)
r1, p1 = stats.pearsonr(density, baf_lulc)
r2, p2 = stats.pearsonr(density, baf_gsmp)
r3, p3 = stats.pearsonr(density, cci_lulc)
r4, p4 = stats.pearsonr(density, cci_gsmp)
print(f"BAF  LULC vs density : r = {r1:.3f}, p = {p1:.4f}")
print(f"BAF  GSMP vs density : r = {r2:.3f}, p = {p2:.4f}")
print(f"CCI  LULC vs density : r = {r3:.3f}, p = {p3:.4f}")
print(f"CCI  GSMP vs density : r = {r4:.3f}, p = {p4:.4f}")

# ============================================================
# PLOT: Gini Lorenz Curves
# ============================================================
def lorenz_curve(values):
    arr = np.sort(np.array(values, dtype=float))
    x = np.linspace(0, 1, len(arr))
    y = np.cumsum(arr) / np.sum(arr)
    return np.insert(x, 0, 0), np.insert(y, 0, 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, lulc, gsmp, name in [
    (axes[0], baf_lulc, baf_gsmp, "BAF"),
    (axes[1], cci_lulc, cci_gsmp, "CCI")
]:
    x_l, y_l = lorenz_curve(lulc)
    x_g, y_g = lorenz_curve(gsmp)
    ax.plot([0, 1], [0, 1], '--k', alpha=0.5, label="Perfect equality")
    ax.plot(x_l, y_l, 'r-o', ms=4, label=f"LULC-Current (G={gini_coefficient(lulc):.3f})")
    ax.plot(x_g, y_g, 'b-s', ms=4, label=f"GSMP (G={gini_coefficient(gsmp):.3f})")
    ax.set_xlabel("Cumulative share of districts", fontsize=11)
    ax.set_ylabel(f"Cumulative share of {name}", fontsize=11)
    ax.set_title(f"Lorenz Curve — {name} Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Lorenz_Curves_BAF_CCI.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nLorenz curve plot saved as Lorenz_Curves_BAF_CCI.png")