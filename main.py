# main.py
# =============================================================
# Main Script for QRM Credit Portfolio Simulation
# Maximilian Droschl and Dmitrii Bashelkhanov
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
from functions import preprocess_indices, simulation, portfolio_loss, risk_measures

# =============================================================
# 1. Load Data
# =============================================================
portfolio = pd.read_excel("data/qrm25HSG_creditportfolio.xlsx")
indices = pd.read_excel("data/qrm25HSG_indexes.xlsx")

# Preprocess
indices_merged, Theta1, Theta2 = preprocess_indices(indices)

# =============================================================
# 2. Run Simulations
# =============================================================
Y_k_M1, d_k_M1, Sigma_M1 = simulation(portfolio, Theta1, Theta2, n_simulations=10000, model='M1')
Y_k_M2, d_k_M2, Sigma_M2 = simulation(portfolio, Theta1, Theta2, n_simulations=10000, model='M2')
Y_k_M3, d_k_M3, Sigma_M3 = simulation(portfolio, Theta1, Theta2, n_simulations=10000, model='M3')

# =============================================================
# 3. Portfolio Loss Distributions
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

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- M1 ---
axes[0].hist(L_M1, bins=50, color='steelblue', alpha=0.7)
axes[0].set_title('Portfolio Loss Distribution — Model M1 (Empirical)')
axes[0].set_ylabel('Frequency')

# --- M2 ---
axes[1].hist(L_M2, bins=50, color='darkorange', alpha=0.7)
axes[1].set_title('Portfolio Loss Distribution — Model M2 (Gaussian)')
axes[1].set_ylabel('Frequency')

# --- M3 ---
axes[2].hist(L_M3, bins=50, color='seagreen', alpha=0.7)
axes[2].set_title('Portfolio Loss Distribution — Model M3 (t-Copula)')
axes[2].set_xlabel('Portfolio Loss (USD)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# =============================================================
# 4. Compute VaR and ES
# =============================================================
models = {'M1 (Empirical)': L_M1, 'M2 (Gaussian)': L_M2, 'M3 (t-Copula)': L_M3}
alphas = [0.95, 0.99]

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
print("\n=== Portfolio Risk Metrics ===")
print(risk_df)



window = 500
