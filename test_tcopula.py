# test_tcopula.py
# =============================================================
# Quick Test Script for t-Copula Estimation Functions
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import your modules
from functions import preprocess_indices
from tcopula import fit_t_copula, t_copula_neg_loglik

print("=" * 60)
print("Testing t-Copula Estimation Functions")
print("=" * 60)

# =============================================================
# 1. Load and Preprocess Data
# =============================================================
print("\n1. Loading data...")
indices = pd.read_excel("data/qrm25HSG_indexes.xlsx")

# Get weekly returns
weekly_indices, Theta1_w, Theta2_w = preprocess_indices(indices, frequency="weekly")
print(f"   Weekly returns: {len(Theta1_w)} observations")
print(f"   SPI mean: {Theta1_w.mean():.6f}, std: {Theta1_w.std():.6f}")
print(f"   SPX mean: {Theta2_w.mean():.6f}, std: {Theta2_w.std():.6f}")

# Get daily returns for comparison
daily_indices, Theta1_d, Theta2_d = preprocess_indices(indices, frequency="daily")
print(f"   Daily returns: {len(Theta1_d)} observations")

# =============================================================
# 2. Test Full Pipeline on Weekly Data
# =============================================================
print("\n" + "=" * 60)
print("2. Testing full estimation pipeline (weekly data)")
print("=" * 60)

# Step 1: estimate Gaussian marginals N(mu_i, sigma_i^2)
mu1_w, sigma1_w = Theta1_w.mean(), Theta1_w.std(ddof=1)
mu2_w, sigma2_w = Theta2_w.mean(), Theta2_w.std(ddof=1)

# Step 2: transform to uniforms via normal CDF:
# U_i = Phi((Theta_i - mu_i) / sigma_i)
z1_w = (Theta1_w - mu1_w) / sigma1_w
z2_w = (Theta2_w - mu2_w) / sigma2_w
u1_w = norm.cdf(z1_w)
u2_w = norm.cdf(z2_w)

# Step 3: fit t-copula on uniforms
rho_w, nu_w, conv_w = fit_t_copula(u1_w, u2_w)

print("\nEstimated Parameters:")
print(f"  SPI Marginal: μ₁ = {mu1_w:.6f}, σ₁ = {sigma1_w:.6f}")
print(f"  SPX Marginal: μ₂ = {mu2_w:.6f}, σ₂ = {sigma2_w:.6f}")
print(f"  t-Copula: ρ = {rho_w:.4f}, ν = {nu_w:.2f}")
print(f"  Converged: {conv_w}")

# Compare with empirical correlation of returns
emp_corr_weekly = np.corrcoef(Theta1_w, Theta2_w)[0, 1]
print(f"\n  Empirical Correlation (Pearson): {emp_corr_weekly:.4f}")
print(f"  Copula Correlation (ρ):          {rho_w:.4f}")
print(f"  Difference: {abs(emp_corr_weekly - rho_w):.4f}")

# =============================================================
# A. Empirical Summary Statistics and Distribution Diagnostics
# =============================================================
import seaborn as sns
from scipy.stats import skew, kurtosis, normaltest, probplot

print("\n" + "=" * 60)
print("Empirical Distribution Analysis of Weekly Log-Returns (Θ)")
print("=" * 60)


def summarize_series(name, data):
    print(f"\n{name}:")
    print(f"  N = {len(data)}")
    print(f"  Mean = {np.mean(data):.6f}")
    print(f"  Std  = {np.std(data, ddof=1):.6f}")
    print(f"  Skewness = {skew(data):.4f}")
    print(f"  Kurtosis (excess) = {kurtosis(data, fisher=True):.4f}")
    print(f"  Min = {np.min(data):.4f}")
    print(f"  Max = {np.max(data):.4f}")
    print(f"  1% / 5% / 95% / 99% quantiles = {np.percentile(data, [1, 5, 95, 99])}")

    stat, pval = normaltest(data)
    print(f"  D’Agostino–Pearson normality test p = {pval:.4e}")
    if pval < 0.05:
        print("  → Reject normality (heavy tails or skewed)")
    else:
        print("  → Fail to reject normality")


summarize_series("SPI weekly log-returns", Theta1_w)
summarize_series("SPX weekly log-returns", Theta2_w)

corr = np.corrcoef(Theta1_w, Theta2_w)[0, 1]
print(f"\nCorrelation between SPI and SPX weekly returns: {corr:.4f}")

# =============================================================
# B. Visualization: Distribution and QQ plots
# =============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

sns.histplot(Theta1_w, kde=True, bins=30, ax=axes[0, 0])
axes[0, 0].set_title("SPI Weekly Log-Returns")

sns.histplot(Theta2_w, kde=True, bins=30, ax=axes[1, 0])
axes[1, 0].set_title("SPX Weekly Log-Returns")

sns.boxplot(x=Theta1_w, ax=axes[0, 1])
axes[0, 1].set_title("SPI Boxplot")
sns.boxplot(x=Theta2_w, ax=axes[1, 1])
axes[1, 1].set_title("SPX Boxplot")

probplot(Theta1_w, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title("SPI QQ-Plot (Normal)")
probplot(Theta2_w, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title("SPX QQ-Plot (Normal)")

plt.tight_layout()
plt.savefig("weekly_returns_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved: weekly_returns_diagnostics.png")

# =============================================================
# C. Empirical distribution comment
# =============================================================
spi_sk = skew(Theta1_w)
spi_ku = kurtosis(Theta1_w)
spx_sk = skew(Theta2_w)
spx_ku = kurtosis(Theta2_w)

print("\nInterpretation:")
if spi_ku > 3 or spx_ku > 3:
    print("  Both series exhibit strong excess kurtosis → heavy tails likely.")
if abs(spi_sk) > 0.5 or abs(spx_sk) > 0.5:
    print("  Noticeable skewness detected → asymmetry in returns.")
if corr > 0.7:
    print("  High cross-market correlation, indicating joint movements.")
else:
    print("  Moderate correlation.")

# =============================================================
# 3. Test on Daily Data
# =============================================================
print("\n" + "=" * 60)
print("3. Testing on daily data")
print("=" * 60)

mu1_d, sigma1_d = Theta1_d.mean(), Theta1_d.std(ddof=1)
mu2_d, sigma2_d = Theta2_d.mean(), Theta2_d.std(ddof=1)

z1_d = (Theta1_d - mu1_d) / sigma1_d
z2_d = (Theta2_d - mu2_d) / sigma2_d
u1_d = norm.cdf(z1_d)
u2_d = norm.cdf(z2_d)

rho_d, nu_d, conv_d = fit_t_copula(u1_d, u2_d)

print("\nEstimated Parameters:")
print(f"  t-Copula: ρ = {rho_d:.4f}, ν = {nu_d:.2f}")
print(f"  Converged: {conv_d}")

emp_corr_daily = np.corrcoef(Theta1_d, Theta2_d)[0, 1]
print(f"\n  Empirical Correlation (Pearson): {emp_corr_daily:.4f}")
print(f"  Copula Correlation (ρ):          {rho_d:.4f}")

# =============================================================
# 4. Test on Subsample (Rolling Window Scenario)
# =============================================================
print("\n" + "=" * 60)
print("4. Testing on subsample (simulating rolling window)")
print("=" * 60)

window_size = 500
Theta1_window = Theta1_d[-window_size:]
Theta2_window = Theta2_d[-window_size:]

mu1_win, sigma1_win = Theta1_window.mean(), Theta1_window.std(ddof=1)
mu2_win, sigma2_win = Theta2_window.mean(), Theta2_window.std(ddof=1)

z1_win = (Theta1_window - mu1_win) / sigma1_win
z2_win = (Theta2_window - mu2_win) / sigma2_win
u1_win = norm.cdf(z1_win)
u2_win = norm.cdf(z2_win)

rho_win, nu_win, conv_win = fit_t_copula(u1_win, u2_win)

print(f"\nLast {window_size} days:")
print(f"  t-Copula: ρ = {rho_win:.4f}, ν = {nu_win:.2f}")
print(f"  Converged: {conv_win}")

# =============================================================
# 5. Visualize the Data and Copula Fit
# =============================================================
print("\n" + "=" * 60)
print("5. Visualizing data and copula transformation")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].scatter(Theta1_w, Theta2_w, alpha=0.5, s=20)
axes[0, 0].set_xlabel('SPI Log Returns')
axes[0, 0].set_ylabel('SPX Log Returns')
axes[0, 0].set_title('Weekly Returns Scatter Plot')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(y=0, linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, linestyle='--', alpha=0.3)

axes[0, 1].scatter(u1_w, u2_w, alpha=0.5, s=20)
axes[0, 1].set_xlabel('U₁ (SPI uniform)')
axes[0, 1].set_ylabel('U₂ (SPX uniform)')
axes[0, 1].set_title('Copula Domain (Uniform Marginals)')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

axes[1, 0].hist(Theta1_w, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('SPI Log Returns')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'SPI Distribution (μ={mu1_w:.5f}, σ={sigma1_w:.5f})')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

axes[1, 1].hist(Theta2_w, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('SPX Log Returns')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'SPX Distribution (μ={mu2_w:.5f}, σ={sigma2_w:.5f})')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('test_tcopula_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved: test_tcopula_visualization.png")
plt.close()

# =============================================================
# 6. Compare Different Sample Periods
# =============================================================
print("\n" + "=" * 60)
print("6. Comparing estimates across different time periods")
print("=" * 60)

periods = {
    'First 100 weeks': (Theta1_w[:100], Theta2_w[:100]),
    'Middle 100 weeks': (Theta1_w[len(Theta1_w)//2-50:len(Theta1_w)//2+50],
                         Theta2_w[len(Theta2_w)//2-50:len(Theta2_w)//2+50]),
    'Last 100 weeks': (Theta1_w[-100:], Theta2_w[-100:]),
    'Full sample': (Theta1_w, Theta2_w)
}

comparison_results = []
for name, (theta1_sample, theta2_sample) in periods.items():
    mu1_p, sigma1_p = theta1_sample.mean(), theta1_sample.std(ddof=1)
    mu2_p, sigma2_p = theta2_sample.mean(), theta2_sample.std(ddof=1)

    z1_p = (theta1_sample - mu1_p) / sigma1_p
    z2_p = (theta2_sample - mu2_p) / sigma2_p
    u1_p = norm.cdf(z1_p)
    u2_p = norm.cdf(z2_p)

    rho_p, nu_p, conv_p = fit_t_copula(u1_p, u2_p)
    emp_corr = np.corrcoef(theta1_sample, theta2_sample)[0, 1]

    comparison_results.append({
        'Period': name,
        'N': len(theta1_sample),
        'Emp_Corr': emp_corr,
        'Copula_ρ': rho_p,
        'Copula_ν': nu_p,
        'Converged': conv_p
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n" + comparison_df.to_string(index=False))

# =============================================================
# 7. Test Direct Copula Fitting (using uniform inputs)
# =============================================================
print("\n" + "=" * 60)
print("7. Testing direct copula fitting on uniform data")
print("=" * 60)

u1_test = u1_w
u2_test = u2_w

rho_direct, nu_direct, converged_direct = fit_t_copula(u1_test, u2_test)

print(f"\nDirect copula fitting:")
print(f"  ρ = {rho_direct:.4f}")
print(f"  ν = {nu_direct:.2f}")
print(f"  Converged: {converged_direct}")

print(f"\nComparison with full pipeline:")
print(f"  Pipeline ρ: {rho_w:.4f}")
print(f"  Direct ρ:   {rho_direct:.4f}")
print(f"  Difference: {abs(rho_w - rho_direct):.6f}")

# =============================================================
# 8. External validation with copulae package (for M3)
# =============================================================
print("\n" + "=" * 60)
print("8. Cross-checking t-Copula fit with copulae package (Model M3)")
print("=" * 60)

from copulae import TCopula

t_copula_pkg = TCopula(dim=2)
U = np.column_stack([u1_test, u2_test])
t_copula_pkg.fit(U, method='ml')

rho_pkg = float(t_copula_pkg.params.rho)
nu_pkg = float(t_copula_pkg.params.df)

print(f"\nCopulae package estimates:")
print(f"  ρ (rho): {rho_pkg:.4f}")
print(f"  ν (nu):  {nu_pkg:.4f}")

print("\nComparison with your implementation:")
print(f"  Your ρ: {rho_direct:.4f}")
print(f"  Your ν: {nu_direct:.4f}")
print(f"  Δρ = {abs(rho_pkg - rho_direct):.6f}")
print(f"  Δν = {abs(nu_pkg - nu_direct):.6f}")

# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("\n✓ Data loaded successfully")
print("✓ Full pipeline tested on weekly and daily data")
print("✓ Rolling window scenario tested")
print("✓ Visualizations created")
print("✓ Different time periods compared")
print("✓ Direct copula fitting validated")

if conv_w and conv_d:
    print("\n✓ ALL TESTS PASSED")
else:
    print("\n⚠ WARNING: Some optimizations did not converge")

print("\nKey findings:")
print(f"  - Weekly returns: ρ = {rho_w:.4f}, ν = {nu_w:.2f}")
print(f"  - Daily returns:  ρ = {rho_d:.4f}, ν = {nu_d:.2f}")
print(f"  - Degrees of freedom suggest "
      f"{'heavy tails' if nu_w < 10 else 'moderate tails' if nu_w < 30 else 'light tails'}")

print("\n" + "=" * 60)

# Log-likelihood profile in ν (holding ρ = ρ̂ fixed)
nus = np.linspace(2.1, 50, 100)
lls = []
for nu in nus:
    ll = -t_copula_neg_loglik([rho_w, nu], u1_w, u2_w)
    lls.append(ll)

plt.plot(nus, lls)
plt.xlabel('ν')
plt.ylabel('Log-Likelihood')
plt.title('t-Copula Log-Likelihood Profile in ν')
plt.show()
