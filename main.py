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

# Import functions from local modules
from functions import (
    preprocess_indices,
    simulation,
    portfolio_loss,
    risk_measures,
    dynamic_var_es_weekly_window
)

# Set working directory (adjust as needed)
# os.chdir("/Users/MaximilianDroschl/Master/HS25/QRM/Assignment/code")

# =============================================================
# 1. Load Data
# =============================================================
print("="*60)
print("Loading Data")
print("="*60)

portfolio = pd.read_excel("data/qrm25HSG_creditportfolio.xlsx")
indices = pd.read_excel("data/qrm25HSG_indexes.xlsx")

print(f"Portfolio size: {len(portfolio)} counterparties")
print(f"Indices data shape: {indices.shape}")

# Preprocess for weekly returns
weekly_indices, Theta1_w, Theta2_w = preprocess_indices(indices, frequency="weekly")
print(f"Weekly returns sample size: {len(Theta1_w)}")

# =============================================================
# 2. Run Full-Sample Simulations
# =============================================================
print("\n" + "="*60)
print("Running Full-Sample Simulations")
print("="*60)

# M1: Empirical resampling
print("\nModel M1: Empirical resampling...")
Y_k_M1, d_k_M1, Sigma_M1, params_M1 = simulation(
    portfolio, Theta1_w, Theta2_w, 
    n_simulations=10000, model='M1', seed=42
)

# M2: Gaussian copula
print("Model M2: Gaussian copula...")
Y_k_M2, d_k_M2, Sigma_M2, params_M2 = simulation(
    portfolio, Theta1_w, Theta2_w, 
    n_simulations=10000, model='M2', seed=42
)

# M3: t-Copula with ML estimation
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

print(f"Loss statistics computed for all models")

# =============================================================
# 4. Visualize Loss Distributions
# =============================================================
print("\n" + "="*60)
print("Visualizing Loss Distributions")
print("="*60)

plt.figure(figsize=(12, 6))
plt.hist(L_M1, bins=100, alpha=0.5, label='M1 (Empirical)', density=True, color='steelblue')
plt.hist(L_M2, bins=100, alpha=0.5, label='M2 (Gaussian)', density=True, color='coral')
plt.hist(L_M3, bins=100, alpha=0.5, label='M3 (t-Copula ML)', density=True, color='mediumseagreen')
plt.title('Portfolio Loss Distribution Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Loss (USD)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.savefig('loss_distributions.png', dpi=300, bbox_inches='tight')
#print("Saved: loss_distributions.png")
plt.show()

# =============================================================
# 5. Risk Measures at Multiple Confidence Levels
# =============================================================
print("\n" + "="*60)
print("Computing Risk Measures")
print("="*60)

models = {
    'M1 (Empirical)': L_M1, 
    'M2 (Gaussian)': L_M2, 
    'M3 (t-Copula ML)': L_M3
}
alphas = [0.95]

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

# Save results
#risk_df.to_csv('risk_measures_fullsample.csv', index=False)
print("\nSaved: risk_measures_fullsample.csv")

# =============================================================
# 6. Dynamic VaR and ES (Rolling Window)
# =============================================================
print("\n" + "="*60)
print("Computing Dynamic Risk Measures")
print("This may take several minutes...")
print("="*60)

dynamic_risk = dynamic_var_es_weekly_window(
    portfolio,
    indices,
    window=500,
    n_simulations=5000,
    alpha=0.99
)

# Save rolling window results
#dynamic_risk.to_csv('dynamic_risk_measures.csv', index=False)
print("\nSaved: dynamic_risk_measures.csv")
print(f"Rolling window results: {len(dynamic_risk)} time points")

# =============================================================
# 7. Visualize Dynamic Risk Measures
# =============================================================
print("\n" + "="*60)
print("Visualizing Dynamic Risk Measures")
print("="*60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# VaR over time
axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M1'], 
             label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M2'], 
             label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
axes[0].plot(dynamic_risk['Date'], dynamic_risk['VaR_M3'], 
             label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
axes[0].set_title('Dynamic Value-at-Risk (99%, Rolling 500-day Window)', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('VaR (USD)', fontsize=12)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(alpha=0.3)

# ES over time
axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M1'], 
             label='M1 (Empirical)', linewidth=1.5, alpha=0.8, color='steelblue')
axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M2'], 
             label='M2 (Gaussian)', linewidth=1.5, alpha=0.8, color='coral')
axes[1].plot(dynamic_risk['Date'], dynamic_risk['ES_M3'], 
             label='M3 (t-Copula ML)', linewidth=1.5, alpha=0.8, color='mediumseagreen')
axes[1].set_title('Dynamic Expected Shortfall (99%, Rolling 500-day Window)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('ES (USD)', fontsize=12)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('dynamic_risk_measures.png', dpi=300, bbox_inches='tight')
print("Saved: dynamic_risk_measures.png")
plt.show()

# =============================================================
# 8. Analyze t-Copula Parameters Over Time
# =============================================================
if 'rho_M3' in dynamic_risk.columns and 'nu_M3' in dynamic_risk.columns:
    print("\n" + "="*60)
    print("Analyzing t-Copula Parameters Over Time")
    print("="*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Filter out non-converged results for cleaner plots
    converged_data = dynamic_risk[dynamic_risk['converged_M3'] == True].copy()
    
    # Correlation over time
    axes[0].plot(converged_data['Date'], converged_data['rho_M3'], 
                 color='steelblue', linewidth=1.5)
    axes[0].set_title('t-Copula Correlation Parameter (ρ) Over Time', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('ρ', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Degrees of freedom over time
    axes[1].plot(converged_data['Date'], converged_data['nu_M3'], 
                 color='coral', linewidth=1.5)
    axes[1].set_title('t-Copula Degrees of Freedom (ν) Over Time', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('ν', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('copula_parameters_over_time.png', dpi=300, bbox_inches='tight')
    print("Saved: copula_parameters_over_time.png")
    plt.show()
    
    # Summary statistics (only for converged estimates)
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
# 9. Summary Statistics
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

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print("\nGenerated files:")
print("  - loss_distributions.png")
print("  - risk_measures_fullsample.csv")
print("  - dynamic_risk_measures.csv")
print("  - dynamic_risk_measures.png")
print("  - copula_parameters_over_time.png")