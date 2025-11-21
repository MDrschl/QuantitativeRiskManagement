# main.py
# =============================================================
# Main Script for QRM Credit Portfolio Simulation
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import os
os.chdir("/Users/MaximilianDroschl/Master/HS25/QRM/Assignment/code")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis

from functions import (
    preprocess_indices,
    simulation,
    portfolio_loss,
    risk_measures,
    loss_statistics,
    default_statistics,
    dynamic_var_es_window
)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================
# 1. Load Data
# =============================================================
print("="*60)
print("Loading Data")
print("="*60)

portfolio = pd.read_excel("data/qrm25HSG_creditportfolio.xlsx")
indices = pd.read_excel("data/qrm25HSG_indexes.xlsx")

print(f"Portfolio size: {len(portfolio)} counterparties")
print(f"Total Exposure: ${portfolio['Exposure USD'].sum():,.0f}")
print(f"Mean Exposure: ${portfolio['Exposure USD'].mean():,.0f}")
print(f"Indices data shape: {indices.shape}")

# Weekly returns
weekly_indices, Theta1_w, Theta2_w = preprocess_indices(indices, frequency="weekly")
print(f"Weekly returns sample size: {len(Theta1_w)}")

# Daily returns for analysis
daily_indices, Theta1_d, Theta2_d = preprocess_indices(indices, frequency="daily")
print(f"Daily returns sample size: {len(Theta1_d)}")

# =============================================================
# 1a. Index Returns Analysis
# =============================================================
print("\n" + "="*60)
print("Analyzing Index Returns")
print("="*60)

# 1a.1 Q-Q Plot: Weekly Log Returns (SPI vs SPX)

plt.figure(figsize=(12, 6))

# --- SPI QQ plot vs Normal ---
ax1 = plt.subplot(1, 2, 1)
stats.probplot(Theta1_w, dist="norm", plot=ax1)
ax1.set_title("Q-Q Plot vs Normal: SPI Weekly Log Returns", fontsize=12, fontweight='bold')
ax1.set_xlabel("Theoretical Quantiles (Normal)")
ax1.set_ylabel("Sample Quantiles (SPI)")
ax1.grid(alpha=0.3)

# --- SPX QQ plot vs Normal ---
ax2 = plt.subplot(1, 2, 2)
stats.probplot(Theta2_w, dist="norm", plot=ax2)
ax2.set_title("Q-Q Plot vs Normal: SPX Weekly Log Returns", fontsize=12, fontweight='bold')
ax2.set_xlabel("Theoretical Quantiles (Normal)")
ax2.set_ylabel("Sample Quantiles (SPX)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("qq_plots_normal_spi_spx.png", dpi=300, bbox_inches="tight")
print("Saved: qq_plots_normal_spi_spx.png")
plt.show()

# 1a.2 Rolling 100-Week Moving Average of Log Returns
print("\nComputing 100-week rolling statistics...")
weekly_df = weekly_indices.copy()
weekly_df['SPI_MA_100'] = weekly_df['SPI_logret'].rolling(window=100).mean()
weekly_df['SPX_MA_100'] = weekly_df['SPX_logret'].rolling(window=100).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# SPI Rolling Average
axes[0].plot(weekly_df['Date'], weekly_df['SPI_logret'], 
             alpha=0.3, linewidth=0.5, color='lightblue', label='Weekly Returns')
axes[0].plot(weekly_df['Date'], weekly_df['SPI_MA_100'], 
             linewidth=2, color='steelblue', label='100-Week Moving Average')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_title('SPI Weekly Log Returns with 100-Week Moving Average', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Log Return', fontsize=11)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(alpha=0.3)

# SPX Rolling Average
axes[1].plot(weekly_df['Date'], weekly_df['SPX_logret'], 
             alpha=0.3, linewidth=0.5, color='lightcoral', label='Weekly Returns')
axes[1].plot(weekly_df['Date'], weekly_df['SPX_MA_100'], 
             linewidth=2, color='coral', label='100-Week Moving Average')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('SPX Weekly Log Returns with 100-Week Moving Average', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Log Return', fontsize=11)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_moving_average_100w.png', dpi=300, bbox_inches='tight')
print("Saved: rolling_moving_average_100w.png")
plt.show()


# Summary statistics
print("\n" + "-"*60)
print("Index Returns Summary Statistics")
print("-"*60)
print(f"\nSPI Daily Log Returns:")
print(f"  Mean: {np.mean(Theta1_d):.6f}")
print(f"  Std Dev: {np.std(Theta1_d):.6f}")
print(f"  Skewness: {skew(Theta1_d):.4f}")
print(f"  Kurtosis: {kurtosis(Theta1_d):.4f}")

print(f"\nSPX Daily Log Returns:")
print(f"  Mean: {np.mean(Theta2_d):.6f}")
print(f"  Std Dev: {np.std(Theta2_d):.6f}")
print(f"  Skewness: {skew(Theta2_d):.4f}")
print(f"  Kurtosis: {kurtosis(Theta2_d):.4f}")

# =============================================================
# 2. Run Full-Sample Simulations
# =============================================================
print("\n" + "="*60)
print("Running Full-Sample Simulations")
print("="*60)

print("\nModel M1: Empirical resampling...")
Y_k_M1, d_k_M1, Sigma_M1, params_M1 = simulation(
    portfolio, Theta1_w, Theta2_w, 
    n_simulations=10000, model='M1', seed=42
)

print("Model M2: Gaussian copula...")
Y_k_M2, d_k_M2, Sigma_M2, params_M2 = simulation(
    portfolio, Theta1_w, Theta2_w, 
    n_simulations=10000, model='M2', seed=42
)

print("Model M3: t-Copula with ML estimation...")
Y_k_M3, d_k_M3, Sigma_M3, params_M3 = simulation(
    portfolio, Theta1_w, Theta2_w, 
    n_simulations=10000, model='M3', seed=42
)

if params_M3 is not None:
    print(f"\nM3 t-Copula Parameters (ML estimation):")
    print(f"  Correlation (ρ): {params_M3['rho']:.4f}")
    print(f"  Degrees of freedom (ν): {params_M3['nu']:.2f}")
    print(f"  Converged: {params_M3['converged']}")

# =============================================================
# 3. Compute Portfolio Losses
# =============================================================
print("\n" + "="*60)
print("Computing Portfolio Losses")
print("="*60)

E_k = portfolio['Exposure USD'].values
R_k = portfolio['R_k'].values

L_M1 = portfolio_loss(Y_k_M1, d_k_M1, E_k, R_k)
L_M2 = portfolio_loss(Y_k_M2, d_k_M2, E_k, R_k)
L_M3 = portfolio_loss(Y_k_M3, d_k_M3, E_k, R_k)

print("Loss statistics computed for all models")

# =============================================================
# 4. Comprehensive Loss Distribution Statistics
# =============================================================
print("\n" + "="*60)
print("COMPREHENSIVE LOSS DISTRIBUTION ANALYSIS")
print("="*60)

# Compute statistics for each model
loss_stats_M1 = loss_statistics(L_M1, "M1 (Empirical)")
loss_stats_M2 = loss_statistics(L_M2, "M2 (Gaussian)")
loss_stats_M3 = loss_statistics(L_M3, "M3 (t-Copula ML)")

loss_stats_df = pd.DataFrame([loss_stats_M1, loss_stats_M2, loss_stats_M3])

print("\n" + "-"*60)
print("LOSS DISTRIBUTION STATISTICS")
print("-"*60)
print(loss_stats_df.to_string(index=False))

# Save to CSV
loss_stats_df.to_csv('loss_distribution_statistics.csv', index=False)
print("\nSaved: loss_distribution_statistics.csv")

# =============================================================
# 5. Default Statistics
# =============================================================
print("\n" + "-"*60)
print("DEFAULT STATISTICS")
print("-"*60)

default_stats_M1 = default_statistics(Y_k_M1, d_k_M1, portfolio, "M1 (Empirical)")
default_stats_M2 = default_statistics(Y_k_M2, d_k_M2, portfolio, "M2 (Gaussian)")
default_stats_M3 = default_statistics(Y_k_M3, d_k_M3, portfolio, "M3 (t-Copula ML)")

default_stats_df = pd.DataFrame([default_stats_M1, default_stats_M2, default_stats_M3])
print(default_stats_df.to_string(index=False))

# Save to CSV
default_stats_df.to_csv('default_statistics.csv', index=False)
print("\nSaved: default_statistics.csv")

# =============================================================
# 6. Detailed Percentile Analysis
# =============================================================
print("\n" + "-"*60)
print("DETAILED PERCENTILE ANALYSIS")
print("-"*60)

percentiles = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
percentile_df = pd.DataFrame({
    'Percentile': [f"P{p}" for p in percentiles],
    'M1 (Empirical)': [np.percentile(L_M1, p) for p in percentiles],
    'M2 (Gaussian)': [np.percentile(L_M2, p) for p in percentiles],
    'M3 (t-Copula ML)': [np.percentile(L_M3, p) for p in percentiles]
})

print(percentile_df.to_string(index=False))
percentile_df.to_csv('loss_percentiles.csv', index=False)
print("\nSaved: loss_percentiles.csv")

# =============================================================
# 7. Visualizations
# =============================================================
print("\n" + "="*60)
print("Creating Visualizations")
print("="*60)

# 7.1 Loss Distribution Histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (L, model_name, color) in enumerate([
    (L_M1, 'M1 (Empirical)', 'steelblue'),
    (L_M2, 'M2 (Gaussian)', 'coral'),
    (L_M3, 'M3 (t-Copula ML)', 'mediumseagreen')
]):
    axes[idx].hist(L, bins=100, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    axes[idx].axvline(np.mean(L), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(L):,.0f}')
    axes[idx].axvline(np.median(L), color='orange', linestyle='--', linewidth=2, label=f'Median: ${np.median(L):,.0f}')
    axes[idx].set_title(f'{model_name}\nSkew: {skew(L):.3f}, Kurt: {kurtosis(L):.3f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Portfolio Loss (USD)', fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_distributions_detailed.png', dpi=300, bbox_inches='tight')
print("Saved: loss_distributions_detailed.png")
plt.show()

# 7.2 Overlaid Density Plot
plt.figure(figsize=(12, 6))
plt.hist(L_M1, bins=100, alpha=0.4, label='M1 (Empirical)', density=True, color='steelblue')
plt.hist(L_M2, bins=100, alpha=0.4, label='M2 (Gaussian)', density=True, color='coral')
plt.hist(L_M3, bins=100, alpha=0.4, label='M3 (t-Copula ML)', density=True, color='mediumseagreen')
plt.title('Portfolio Loss Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Loss (USD)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_distributions_overlay.png', dpi=300, bbox_inches='tight')
print("Saved: loss_distributions_overlay.png")
plt.show()

# 7.3 Tail Comparison (Focus on extreme losses)
plt.figure(figsize=(12, 6))
plt.hist(L_M1, bins=100, alpha=0.5, label='M1 (Empirical)', density=True, color='steelblue')
plt.hist(L_M2, bins=100, alpha=0.5, label='M2 (Gaussian)', density=True, color='coral')
plt.hist(L_M3, bins=100, alpha=0.5, label='M3 (t-Copula ML)', density=True, color='mediumseagreen')
plt.xlim(np.percentile(np.concatenate([L_M1, L_M2, L_M3]), 90), 
         np.percentile(np.concatenate([L_M1, L_M2, L_M3]), 100))
plt.title('Loss Distribution Tails (Above 90th Percentile)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Loss (USD)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_distributions_tail.png', dpi=300, bbox_inches='tight')
print("Saved: loss_distributions_tail.png")
plt.show()

# 7.4 Box Plots
fig, ax = plt.subplots(figsize=(10, 6))
box_data = [L_M1, L_M2, L_M3]
bp = ax.boxplot(box_data, labels=['M1 (Empirical)', 'M2 (Gaussian)', 'M3 (t-Copula ML)'],
                patch_artist=True, showmeans=True)

colors = ['steelblue', 'coral', 'mediumseagreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Portfolio Loss (USD)', fontsize=12)
ax.set_title('Loss Distribution Comparison (Box Plots)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('loss_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: loss_boxplots.png")
plt.show()

# 7.5 Q-Q Plots (Compare M2 vs M1 and M3 vs M1)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# M2 vs M1
axes[0].scatter(np.sort(L_M1), np.sort(L_M2), alpha=0.5, s=10, color='coral')
axes[0].plot([L_M1.min(), L_M1.max()], [L_M1.min(), L_M1.max()], 
             'r--', lw=2, label='45° line')
axes[0].set_xlabel('M1 (Empirical) Quantiles', fontsize=11)
axes[0].set_ylabel('M2 (Gaussian) Quantiles', fontsize=11)
axes[0].set_title('Q-Q Plot: M2 vs M1', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# M3 vs M1
axes[1].scatter(np.sort(L_M1), np.sort(L_M3), alpha=0.5, s=10, color='mediumseagreen')
axes[1].plot([L_M1.min(), L_M1.max()], [L_M1.min(), L_M1.max()], 
             'r--', lw=2, label='45° line')
axes[1].set_xlabel('M1 (Empirical) Quantiles', fontsize=11)
axes[1].set_ylabel('M3 (t-Copula) Quantiles', fontsize=11)
axes[1].set_title('Q-Q Plot: M3 vs M1', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_qq_plots.png', dpi=300, bbox_inches='tight')
print("Saved: loss_qq_plots.png")
plt.show()

# 7.6 Cumulative Distribution Functions
plt.figure(figsize=(12, 6))
for L, label, color in [
    (L_M1, 'M1 (Empirical)', 'steelblue'),
    (L_M2, 'M2 (Gaussian)', 'coral'),
    (L_M3, 'M3 (t-Copula ML)', 'mediumseagreen')
]:
    sorted_L = np.sort(L)
    cdf = np.arange(1, len(L) + 1) / len(L)
    plt.plot(sorted_L, cdf, label=label, linewidth=2, color=color)

plt.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95th Percentile')
plt.axhline(0.99, color='darkred', linestyle='--', alpha=0.5, label='99th Percentile')
plt.xlabel('Portfolio Loss (USD)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_cdf.png', dpi=300, bbox_inches='tight')
print("Saved: loss_cdf.png")
plt.show()

# 7.7 Default Count Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (Y_k, d_k, model_name, color) in enumerate([
    (Y_k_M1, d_k_M1, 'M1 (Empirical)', 'steelblue'),
    (Y_k_M2, d_k_M2, 'M2 (Gaussian)', 'coral'),
    (Y_k_M3, d_k_M3, 'M3 (t-Copula ML)', 'mediumseagreen')
]):
    I_k = (Y_k <= d_k).astype(int)
    n_defaults = I_k.sum(axis=1)
    
    axes[idx].hist(n_defaults, bins=range(0, int(n_defaults.max()) + 2), 
                   alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    axes[idx].axvline(np.mean(n_defaults), color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {np.mean(n_defaults):.2f}')
    axes[idx].set_title(f'{model_name}\nMean Defaults: {np.mean(n_defaults):.2f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Number of Defaults', fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('default_count_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: default_count_distributions.png")
plt.show()

# =============================================================
# 8. Risk Measures
# =============================================================
print("\n" + "="*60)
print("Computing Risk Measures")
print("="*60)

models = {
    'M1 (Empirical)': L_M1, 
    'M2 (Gaussian)': L_M2, 
    'M3 (t-Copula ML)': L_M3
}
alphas = [0.90, 0.95, 0.99]

risk_results = []
for model_name, L in models.items():
    for alpha in alphas:
        var, es = risk_measures(L, alpha)
        risk_results.append({
            'Model': model_name,
            'Alpha': alpha,
            'VaR': var,
            'ES': es,
            'Mean Loss': L.mean(),
            'Std Dev': L.std()
        })

risk_df = pd.DataFrame(risk_results)
print("\n" + risk_df.to_string(index=False))
risk_df.to_csv('risk_measures_fullsample.csv', index=False)
print("\nSaved: risk_measures_fullsample.csv")

# Risk measure visualizations (bars)
var_pivot = risk_df.pivot(index='Model', columns='Alpha', values='VaR')
es_pivot  = risk_df.pivot(index='Model', columns='Alpha', values='ES')

models_order = var_pivot.index.tolist()
alpha_list = sorted(var_pivot.columns.tolist())
alpha_labels = [f"{int(a * 100)}%" for a in alpha_list]

x = np.arange(len(models_order))
bar_width = 0.25

# VaR bar chart
plt.figure(figsize=(10, 6))
for j, alpha in enumerate(alpha_list):
    plt.bar(x + j * bar_width, var_pivot[alpha].values, 
            width=bar_width, label=f"VaR {int(alpha * 100)}%")

plt.xticks(x + bar_width, models_order, rotation=0)
plt.ylabel('VaR (USD)', fontsize=12)
plt.title('Value-at-Risk by Model and Confidence Level', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('risk_measures_VaR_bars.png', dpi=300, bbox_inches='tight')
print("Saved: risk_measures_VaR_bars.png")
plt.show()

# ES bar chart
plt.figure(figsize=(10, 6))
for j, alpha in enumerate(alpha_list):
    plt.bar(x + j * bar_width, es_pivot[alpha].values, 
            width=bar_width, label=f"ES {int(alpha * 100)}%")

plt.xticks(x + bar_width, models_order, rotation=0)
plt.ylabel('Expected Shortfall (USD)', fontsize=12)
plt.title('Expected Shortfall by Model and Confidence Level', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('risk_measures_ES_bars.png', dpi=300, bbox_inches='tight')
print("Saved: risk_measures_ES_bars.png")
plt.show()

# =============================================================
# 9. Dynamic VaR and ES (Rolling Window)
# =============================================================
print("\n" + "="*60)
print("Computing Dynamic Risk Measures")
print("This may take several minutes...")
print("="*60)

dynamic_risk = dynamic_var_es_window(
    portfolio,
    indices,
    window=500,
    n_simulations=5000,
    alpha=0.95
)

dynamic_risk.to_csv('dynamic_risk_measures.csv', index=False)
print("\nSaved: dynamic_risk_measures.csv")
print(f"Rolling window results: {len(dynamic_risk)} time points")

# Dynamic risk visualizations
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M1'], 
             label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M2'], 
             label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M3'], 
             label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
axes[0].set_title('Dynamic Value-at-Risk (95%, Rolling 500-day Window)', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('VaR (USD)', fontsize=12)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(alpha=0.3)

axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M1'], 
             label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M2'], 
             label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M3'], 
             label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
axes[1].set_title('Dynamic Expected Shortfall (95%, Rolling 500-day Window)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('ES (USD)', fontsize=12)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dynamic_risk_measures.png', dpi=300, bbox_inches='tight')
print("Saved: dynamic_risk_measures.png")
plt.show()


# =============================================================
# 10. Dynamic Covariance Structure
# =============================================================

print("\n" + "="*60)
print("Analyzing Dynamic Correlations from All Models")
print("="*60)

if 'rho_M1' in dynamic_risk.columns:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot correlation estimates from all three models
    ax.plot(dynamic_risk['Date'], dynamic_risk['rho_M1'], 
            label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
    ax.plot(dynamic_risk['Date'], dynamic_risk['rho_M2'], 
            label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
    ax.plot(dynamic_risk['Date'], dynamic_risk['rho_M3'], 
            label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
    
    # Add overall correlation as reference line
    overall_corr_weekly = np.corrcoef(Theta1_w, Theta2_w)[0, 1]
    ax.axhline(y=overall_corr_weekly, color='red', linestyle='--', linewidth=2, 
               label=f'Full-Sample Correlation: {overall_corr_weekly:.4f}', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Correlation Coefficient (ρ)', fontsize=12)
    ax.set_title('Dynamic Correlation Estimates (Rolling 500-day Window)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('dynamic_correlation_all_models.png', dpi=300, bbox_inches='tight')
    print("Saved: dynamic_correlation_all_models.png")
    plt.show()
    
    # Summary statistics for correlations
    print("\nDynamic Correlation Summary Statistics:")
    print("-" * 60)
    
    for model in ['M1', 'M2', 'M3']:
        col = f'rho_{model}'
        if col in dynamic_risk.columns:
            valid_corr = dynamic_risk[col].dropna()
            print(f"\n{model}:")
            print(f"  Mean:   {valid_corr.mean():.4f}")
            print(f"  Median: {valid_corr.median():.4f}")
            print(f"  Std:    {valid_corr.std():.4f}")
            print(f"  Min:    {valid_corr.min():.4f}")
            print(f"  Max:    {valid_corr.max():.4f}")
            print(f"  Q1:     {valid_corr.quantile(0.25):.4f}")
            print(f"  Q3:     {valid_corr.quantile(0.75):.4f}")
    
    print(f"\nFull-Sample Correlation (Weekly): {overall_corr_weekly:.4f}")
    
    # Create comparison plot: Correlation difference between models
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Correlation differences
    axes[0].plot(dynamic_risk['Date'], 
                 dynamic_risk['rho_M2'] - dynamic_risk['rho_M1'],
                 label='M2 - M1 (Gaussian vs Empirical)', 
                 linewidth=1.5, alpha=0.8, color='coral')
    axes[0].plot(dynamic_risk['Date'], 
                 dynamic_risk['rho_M3'] - dynamic_risk['rho_M1'],
                 label='M3 - M1 (t-Copula vs Empirical)', 
                 linewidth=1.5, alpha=0.8, color='mediumseagreen')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_ylabel('Correlation Difference', fontsize=12)
    axes[0].set_title('Correlation Differences Between Models', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Standard deviation of correlations (rolling 50-window)
    axes[1].plot(dynamic_risk['Date'], 
                 dynamic_risk['rho_M1'].rolling(window=50).std(),
                 label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
    axes[1].plot(dynamic_risk['Date'], 
                 dynamic_risk['rho_M2'].rolling(window=50).std(),
                 label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
    axes[1].plot(dynamic_risk['Date'], 
                 dynamic_risk['rho_M3'].rolling(window=50).std(),
                 label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Rolling Std Dev (50-day)', fontsize=12)
    axes[1].set_title('Volatility of Correlation Estimates', 
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_comparison_analysis.png")
    plt.show()
    
    # Optional: Compare copula correlation vs marginal correlation for M3
    if 'rho_M3_copula' in dynamic_risk.columns:
        converged_data = dynamic_risk[dynamic_risk['converged_M3'] == True].copy()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(converged_data['Date'], converged_data['rho_M3'], 
                label='Marginal Correlation (from Σ)', 
                linewidth=1.5, alpha=0.8, color='mediumseagreen')
        ax.plot(converged_data['Date'], converged_data['rho_M3_copula'], 
                label='Copula Correlation Parameter', 
                linewidth=1.5, alpha=0.8, color='darkgreen', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('M3: Marginal vs Copula Correlation (Converged Windows Only)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig('m3_correlation_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: m3_correlation_comparison.png")
        plt.show()

else:
    print("Warning: Correlation columns not found in dynamic_risk DataFrame.")
    print("Please ensure the updated dynamic_var_es_window function is being used.")

# =============================================================
# 11. Analyze t-Copula Parameters Over Time
# =============================================================
if 'rho_M3' in dynamic_risk.columns and 'nu_M3' in dynamic_risk.columns:
    print("\n" + "="*60)
    print("Analyzing t-Copula Parameters Over Time")
    print("="*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    converged_data = dynamic_risk[dynamic_risk['converged_M3'] == True].copy()
    
    axes[0].plot(converged_data['Date'], converged_data['rho_M3'], 
                 color='steelblue', linewidth=1.5)
    axes[0].set_title('t-Copula Correlation Parameter (ρ) Over Time', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('ρ', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    axes[1].plot(converged_data['Date'], converged_data['nu_M3'], 
                 color='coral', linewidth=1.5)
    axes[1].set_title('t-Copula Degrees of Freedom (ν) Over Time', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('ν', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('copula_parameters_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved: copula_parameters_over_time.png")
    plt.show()
    
    print("\nt-Copula Parameter Statistics (Converged Windows Only):")
    print(f"Number of converged windows: {len(converged_data)}/{len(dynamic_risk)}")
    print(f"\nCorrelation (ρ):")
    print(f"  Mean: {converged_data['rho_M3'].mean():.4f}")
    print(f"  Std:  {converged_data['rho_M3'].std():.4f}")
    print(f"  Min:  {converged_data['rho_M3'].min():.4f}")
    print(f"  Max:  {converged_data['rho_M3'].max():.4f}")
    print(f"\nDegrees of Freedom (ν):")
    print(f"  Mean: {converged_data['nu_M3'].mean():.2f}")
    print(f"  Std:  {converged_data['nu_M3'].std():.2f}")
    print(f"  Min:  {converged_data['nu_M3'].min():.2f}")
    print(f"  Max:  {converged_data['nu_M3'].max():.2f}")

# =============================================================
# 11. Summary Statistics
# =============================================================
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

print("\nCovariance Matrix (M1 - Empirical):")
print(Sigma_M1)

print("\nCovariance Matrix (M2 - Gaussian):")
print(Sigma_M2)

print("\nCovariance Matrix (M3 - t-Copula):")
print(Sigma_M3)

# =============================================================
# Final Summary
# =============================================================
print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print("\nGenerated files:")
print("  CSV Files:")
print("    - loss_distribution_statistics.csv")
print("    - default_statistics.csv")
print("    - loss_percentiles.csv")
print("    - risk_measures_fullsample.csv")
print("    - dynamic_risk_measures.csv")
print("\n  Visualization Files:")
print("    Index Analysis:")
print("      - qq_plot_daily_returns.png")
print("      - rolling_moving_average_100w.png")
print("      - rolling_correlation_100w.png")
print("    Loss Distribution Analysis:")
print("      - loss_distributions_detailed.png")
print("      - loss_distributions_overlay.png")
print("      - loss_distributions_tail.png")
print("      - loss_boxplots.png")
print("      - loss_qq_plots.png")
print("      - loss_cdf.png")
print("      - default_count_distributions.png")
print("    Risk Measures:")
print("      - risk_measures_VaR_bars.png")
print("      - risk_measures_ES_bars.png")
print("      - dynamic_risk_measures.png")
print("      - copula_parameters_over_time.png")