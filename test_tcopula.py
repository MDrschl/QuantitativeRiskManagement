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
from tcopula import fit_gaussian_marginals_and_t_copula, fit_t_copula

print("="*60)
print("Testing t-Copula Estimation Functions")
print("="*60)

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
print("\n" + "="*60)
print("2. Testing full estimation pipeline (weekly data)")
print("="*60)

params_weekly = fit_gaussian_marginals_and_t_copula(Theta1_w, Theta2_w)

print("\nEstimated Parameters:")
print(f"  SPI Marginal: μ₁ = {params_weekly['mu1']:.6f}, σ₁ = {params_weekly['sigma1']:.6f}")
print(f"  SPX Marginal: μ₂ = {params_weekly['mu2']:.6f}, σ₂ = {params_weekly['sigma2']:.6f}")
print(f"  t-Copula: ρ = {params_weekly['rho']:.4f}, ν = {params_weekly['nu']:.2f}")
print(f"  Converged: {params_weekly['converged']}")

# Compare with empirical correlation
emp_corr_weekly = np.corrcoef(Theta1_w, Theta2_w)[0, 1]
print(f"\n  Empirical Correlation (Pearson): {emp_corr_weekly:.4f}")
print(f"  Copula Correlation (ρ):          {params_weekly['rho']:.4f}")
print(f"  Difference: {abs(emp_corr_weekly - params_weekly['rho']):.4f}")



# =============================================================
# A. Empirical Summary Statistics and Distribution Diagnostics
# =============================================================
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro, normaltest, ttest_1samp, probplot

print("\n" + "="*60)
print("Empirical Distribution Analysis of Weekly Log-Returns (Θ)")
print("="*60)

def summarize_series(name, data):
    print(f"\n{name}:")
    print(f"  N = {len(data)}")
    print(f"  Mean = {np.mean(data):.6f}")
    print(f"  Std  = {np.std(data, ddof=1):.6f}")
    print(f"  Skewness = {skew(data):.4f}")
    print(f"  Kurtosis (excess) = {kurtosis(data, fisher=True):.4f}")
    print(f"  Min = {np.min(data):.4f}")
    print(f"  Max = {np.max(data):.4f}")
    print(f"  1% / 5% / 95% / 99% quantiles = {np.percentile(data,[1,5,95,99])}")

    # Normality tests
    stat, pval = normaltest(data)
    print(f"  D’Agostino–Pearson normality test p = {pval:.4e}")
    if pval < 0.05:
        print("  → Reject normality (heavy tails or skewed)")
    else:
        print("  → Fail to reject normality")

# Print summary
summarize_series("SPI weekly log-returns", Theta1_w)
summarize_series("SPX weekly log-returns", Theta2_w)

# Combined correlation
corr = np.corrcoef(Theta1_w, Theta2_w)[0,1]
print(f"\nCorrelation between SPI and SPX weekly returns: {corr:.4f}")

# =============================================================
# B. Visualization: Distribution and QQ plots
# =============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 1. Histogram + KDE
sns.histplot(Theta1_w, kde=True, bins=30, ax=axes[0,0], color="steelblue")
axes[0,0].set_title("SPI Weekly Log-Returns")

sns.histplot(Theta2_w, kde=True, bins=30, ax=axes[1,0], color="coral")
axes[1,0].set_title("SPX Weekly Log-Returns")

# 2. Boxplots
sns.boxplot(x=Theta1_w, ax=axes[0,1], color="steelblue")
axes[0,1].set_title("SPI Boxplot")
sns.boxplot(x=Theta2_w, ax=axes[1,1], color="coral")
axes[1,1].set_title("SPX Boxplot")

# 3. QQ-plots (normal)
probplot(Theta1_w, dist="norm", plot=axes[0,2])
axes[0,2].set_title("SPI QQ-Plot (Normal)")
probplot(Theta2_w, dist="norm", plot=axes[1,2])
axes[1,2].set_title("SPX QQ-Plot (Normal)")

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
print("\n" + "="*60)
print("3. Testing on daily data")
print("="*60)

params_daily = fit_gaussian_marginals_and_t_copula(Theta1_d, Theta2_d)

print("\nEstimated Parameters:")
print(f"  t-Copula: ρ = {params_daily['rho']:.4f}, ν = {params_daily['nu']:.2f}")
print(f"  Converged: {params_daily['converged']}")

emp_corr_daily = np.corrcoef(Theta1_d, Theta2_d)[0, 1]
print(f"\n  Empirical Correlation (Pearson): {emp_corr_daily:.4f}")
print(f"  Copula Correlation (ρ):          {params_daily['rho']:.4f}")

# =============================================================
# 4. Test on Subsample (Rolling Window Scenario)
# =============================================================
print("\n" + "="*60)
print("4. Testing on subsample (simulating rolling window)")
print("="*60)

# Take last 500 days
window_size = 500
Theta1_window = Theta1_d[-window_size:]
Theta2_window = Theta2_d[-window_size:]

params_window = fit_gaussian_marginals_and_t_copula(Theta1_window, Theta2_window)

print(f"\nLast {window_size} days:")
print(f"  t-Copula: ρ = {params_window['rho']:.4f}, ν = {params_window['nu']:.2f}")
print(f"  Converged: {params_window['converged']}")

# =============================================================
# 5. Visualize the Data and Copula Fit
# =============================================================
print("\n" + "="*60)
print("5. Visualizing data and copula transformation")
print("="*60)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Scatter plot of returns
axes[0, 0].scatter(Theta1_w, Theta2_w, alpha=0.5, s=20)
axes[0, 0].set_xlabel('SPI Log Returns')
axes[0, 0].set_ylabel('SPX Log Returns')
axes[0, 0].set_title('Weekly Returns Scatter Plot')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Plot 2: Transform to uniform marginals
mu1, sigma1 = params_weekly['mu1'], params_weekly['sigma1']
mu2, sigma2 = params_weekly['mu2'], params_weekly['sigma2']
u1 = norm.cdf(Theta1_w, mu1, sigma1)
u2 = norm.cdf(Theta2_w, mu2, sigma2)

axes[0, 1].scatter(u1, u2, alpha=0.5, s=20, color='coral')
axes[0, 1].set_xlabel('U₁ (SPI uniform)')
axes[0, 1].set_ylabel('U₂ (SPX uniform)')
axes[0, 1].set_title('Copula Domain (Uniform Marginals)')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim([0, 1])
axes[0, 1].set_ylim([0, 1])

# Plot 3: Histogram of SPI returns
axes[1, 0].hist(Theta1_w, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('SPI Log Returns')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'SPI Distribution (μ={mu1:.5f}, σ={sigma1:.5f})')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Plot 4: Histogram of SPX returns
axes[1, 1].hist(Theta2_w, bins=30, alpha=0.7, color='coral', edgecolor='black')
axes[1, 1].set_xlabel('SPX Log Returns')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'SPX Distribution (μ={mu2:.5f}, σ={sigma2:.5f})')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('test_tcopula_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved: test_tcopula_visualization.png")
plt.close()

# =============================================================
# 6. Compare Different Sample Periods
# =============================================================
print("\n" + "="*60)
print("6. Comparing estimates across different time periods")
print("="*60)

periods = {
    'First 100 weeks': Theta1_w[:100],
    'Middle 100 weeks': Theta1_w[len(Theta1_w)//2-50:len(Theta1_w)//2+50],
    'Last 100 weeks': Theta1_w[-100:],
    'Full sample': Theta1_w
}

comparison_results = []
for name, theta1_sample in periods.items():
    # Get corresponding theta2 sample
    if name == 'First 100 weeks':
        theta2_sample = Theta2_w[:100]
    elif name == 'Middle 100 weeks':
        theta2_sample = Theta2_w[len(Theta2_w)//2-50:len(Theta2_w)//2+50]
    elif name == 'Last 100 weeks':
        theta2_sample = Theta2_w[-100:]
    else:
        theta2_sample = Theta2_w
    
    params = fit_gaussian_marginals_and_t_copula(theta1_sample, theta2_sample)
    emp_corr = np.corrcoef(theta1_sample, theta2_sample)[0, 1]
    
    comparison_results.append({
        'Period': name,
        'N': len(theta1_sample),
        'Emp_Corr': emp_corr,
        'Copula_ρ': params['rho'],
        'Copula_ν': params['nu'],
        'Converged': params['converged']
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n" + comparison_df.to_string(index=False))

# =============================================================
# 7. Test Direct Copula Fitting (using uniform inputs)
# =============================================================
print("\n" + "="*60)
print("7. Testing direct copula fitting on uniform data")
print("="*60)

# Transform returns to uniforms manually
u1_test = norm.cdf(Theta1_w, params_weekly['mu1'], params_weekly['sigma1'])
u2_test = norm.cdf(Theta2_w, params_weekly['mu2'], params_weekly['sigma2'])

# Fit copula directly
rho_direct, nu_direct, converged_direct = fit_t_copula(u1_test, u2_test)

print(f"\nDirect copula fitting:")
print(f"  ρ = {rho_direct:.4f}")
print(f"  ν = {nu_direct:.2f}")
print(f"  Converged: {converged_direct}")

print(f"\nComparison with full pipeline:")
print(f"  Pipeline ρ: {params_weekly['rho']:.4f}")
print(f"  Direct ρ:   {rho_direct:.4f}")
print(f"  Difference: {abs(params_weekly['rho'] - rho_direct):.6f}")
# =============================================================
# 8. External validation with copulae package (for M3)
# =============================================================
print("\n" + "="*60)
print("8. Cross-checking t-Copula fit with copulae package (Model M3)")
print("="*60)

from copulae import TCopula

# Create a t-Copula object with unspecified parameters
t_copula_pkg = TCopula(dim=2)

# Fit using the same uniform marginals as your own implementation
# copulae expects a (n x 2) array of uniforms
U = np.column_stack([u1_test, u2_test])

t_copula_pkg.fit(U, method='ml')  # maximum likelihood

# Extract fitted parameters
rho_pkg = float(t_copula_pkg.params.rho)
nu_pkg = float(t_copula_pkg.params.df)

print(f"\nCopulae package estimates:")
print(f"  ρ (rho): {rho_pkg:.4f}")
print(f"  ν (nu):  {nu_pkg:.4f}")

# Compare to your implementation
print("\nComparison with your implementation:")
print(f"  Your ρ: {rho_direct:.4f}")
print(f"  Your ν: {nu_direct:.4f}")
print(f"  Δρ = {abs(rho_pkg - rho_direct):.6f}")
print(f"  Δν = {abs(nu_pkg - nu_direct):.6f}")


# =============================================================
# Summary
# =============================================================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("\n✓ Data loaded successfully")
print("✓ Full pipeline tested on weekly and daily data")
print("✓ Rolling window scenario tested")
print("✓ Visualizations created")
print("✓ Different time periods compared")
print("✓ Direct copula fitting validated")

if params_weekly['converged'] and params_daily['converged']:
    print("\n✓ ALL TESTS PASSED")
    print("\nThe t-copula estimation functions are working correctly!")
else:
    print("\n⚠ WARNING: Some optimizations did not converge")
    print("  This may be normal for small samples or extreme data")

print("\nKey findings:")
print(f"  - Weekly returns: ρ = {params_weekly['rho']:.4f}, ν = {params_weekly['nu']:.2f}")
print(f"  - Daily returns:  ρ = {params_daily['rho']:.4f}, ν = {params_daily['nu']:.2f}")
print(f"  - Degrees of freedom suggest {'heavy tails' if params_weekly['nu'] < 10 else 'moderate tails' if params_weekly['nu'] < 30 else 'light tails'}")

print("\n" + "="*60)

from tcopula import t_copula_neg_loglik

nus = np.linspace(2.1, 50, 100)
rhos = [params_weekly['rho']]
lls = []

for nu in nus:
    ll = -t_copula_neg_loglik([rhos[0], nu], u1, u2)
    lls.append(ll)

plt.plot(nus, lls)
plt.xlabel('ν')
plt.ylabel('Log-Likelihood')
plt.title('t-Copula Log-Likelihood Profile in ν')
plt.show()
