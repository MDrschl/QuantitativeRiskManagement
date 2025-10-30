# main.py
# =============================================================
# Main Script for QRM Credit Portfolio Simulation
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import os
print(os.getcwd())
os.chdir("/Users/MaximilianDroschl/Master/HS25/QRM/Assignment/code")

from functions import (
    preprocess_indices,
    simulation,
    portfolio_loss,
    risk_measures,
    dynamic_var_es_weekly_window
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# 1. Load Data
# =============================================================
portfolio = pd.read_excel("data/qrm25HSG_creditportfolio.xlsx")
indices = pd.read_excel("data/qrm25HSG_indexes.xlsx")

# Preprocess
# For weekly returns
weekly_indices, Theta1_w, Theta2_w = preprocess_indices(indices, frequency="weekly")


# =============================================================
# Step (v) and (vi): Run Simulations
# =============================================================
Y_k_M1, d_k_M1, Sigma_M1 = simulation(portfolio, Theta1_w, Theta2_w, n_simulations=10000, model='M1')
Y_k_M2, d_k_M2, Sigma_M2 = simulation(portfolio, Theta1_w, Theta2_w, n_simulations=10000, model='M2')
Y_k_M3, d_k_M3, Sigma_M3 = simulation(portfolio, Theta1_w, Theta2_w, n_simulations=10000, model='M2')

# =============================================================
# Step (vii): Portfolio Loss Distributions
# =============================================================
E_k = portfolio['Exposure USD'].values
R_k = portfolio['R_k'].values

L_M1 = portfolio_loss(Y_k_M1, d_k_M1, E_k, R_k)
L_M2 = portfolio_loss(Y_k_M2, d_k_M2, E_k, R_k)
L_M3 = portfolio_loss(Y_k_M3, d_k_M3, E_k, R_k)

plt.figure(figsize=(10,6))
plt.hist(L_M1, bins=50, alpha=0.6, label='M1 (Empirical)')
plt.hist(L_M2, bins=50, alpha=0.6, label='M2 (Gaussian)')
plt.hist(L_M3, bins=50, alpha=0.6, label='M3 (t-Copula)')
plt.title('Portfolio Loss Distribution under M1, M2, and M3')
plt.xlabel('Portfolio Loss (USD)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================
# Step (viii): Compute VaR and ES
# =============================================================
models = {'M1 (Empirical)': L_M1, 'M2 (Gaussian)': L_M2, 'M3 (t-Copula)': L_M3}
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
print(risk_df)



# ============================================================================
# Step (ix): Dynamic VaR and ES (rolling 500-day window)
# ============================================================================
dynamic_risk = dynamic_var_es_weekly_window(
    portfolio,
    indices,
    window=500,
    n_simulations=5000,
    alpha=0.99
)

plt.figure(figsize=(10,5))
plt.plot(dynamic_risk['Date'], dynamic_risk['VaR_M1'], label='VaR M1')
plt.plot(dynamic_risk['Date'], dynamic_risk['VaR_M2'], label='VaR M2')
plt.plot(dynamic_risk['Date'], dynamic_risk['VaR_M3'], label='VaR M3')
plt.title('Dynamic Value-at-Risk (weekly returns, rolling 500 daily window)')
plt.legend()
plt.show()
