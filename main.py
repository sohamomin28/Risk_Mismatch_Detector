import pandas as pd
import yfinance as yf


# PHASE 1: THE CLIENT BOOK AND DATA ENGINE
# Goal : Load the bank's client database anf prepare the starting portfolio values

# Step 1: Load the mock client database from a CSV file
df = pd.read_csv('clients.csv')
print(df)

# Normalize the weights(safety check)
df['Total_Weight'] = df['Stock_Weight'] + df['Bond_Weight']
df['Stock_Weight'] = df['Stock_Weight'] / df['Total_Weight']
df['Bond_Weight'] = df['Bond_Weight'] / df['Total_Weight']
                                           
# Step 2: Fetch 1 year of daily market prices (Stocks vs Bonds)

tickers = ['SPY', 'AGG']
market_data = yf.download(tickers, period='1y', auto_adjust=True)

close_prices = market_data['Close']
print(close_prices.head())

# Step 3: Portfolio Conversion

start_prices = close_prices.iloc[0]
spy_start_prices = start_prices['SPY']
agg_start_prices = start_prices['AGG']

# calculate how many shares of SPY and AGG each client bought
df['Stocks_Shares'] = df['Initial_Value'] * df['Stock_Weight'] / spy_start_prices
df['Bonds_Shares'] = df['Initial_Value'] * df['Bond_Weight'] / agg_start_prices

# The final "Bank Book"
print("\n--Updated Client Portfolios with Shares--")
print(df[['Name', 'Stocks_Shares', 'Bonds_Shares']])

# PHASE 2: THE TRACKER (Portfolio drift logic)
# Goal: Track the daily value of each client's portfolio and calculate returns

import numpy as np

# Vectorized calculation of daily portfolio values
# We reshape prices into colimn vectors to multiply across the client row vector
# This creates a 'Date X Clients' matrix of values for stocks and bonds, which we then sum to get total portfolio values
stock_values = close_prices['SPY'].values.reshape(-1, 1) * df['Stocks_Shares'].values
bond_values = close_prices['AGG'].values.reshape(-1, 1) * df['Bonds_Shares'].values

# Total portfolio and weight analysis
total_values = stock_values + bond_values
current_stock_weights = stock_values / total_values

# Drift Calculation (Absolute Variance)
# Detects if the portfolio is too risky or too conservative
drift_matrix = np.abs(current_stock_weights - df['Stock_Weight'].values)

# The Alert System
# We will flag any client whose portfolio drift exceeds 5% (0.05)
Drift_threshold = 0.05
print("\n--Risk Mismatch Alerts--")

for i, name in enumerate(df['Name']):
    breach_mask = drift_matrix[:, i] > Drift_threshold
    if breach_mask.any():
        first_breach_idx = np.argmax(breach_mask)
        breach_date = close_prices.index[first_breach_idx].date()
        severity = drift_matrix[first_breach_idx, i]
        print(f"ALARAM : {name} | Date: {breach_date} | Drift: {severity:.2%}")
    else:
        print(f"OK: {name} | Portfolio is within tolerance.")


# Create a summary dataframe for reporting

summary_df = []

for i, name in enumerate(df['Name']):
    client_drift = drift_matrix[:, i]

    if breach_mask.any():
        first_breach_idx = np.argmax(breach_mask)
        breach_date = close_prices.index[first_breach_idx].date()
    else:
        breach_date = "N/A"

    summary_df.append({
        'Name': name,
        'Max_Drift': np.max(client_drift),
        'Avg_Drift': np.mean(client_drift),
        'Breach_Date': breach_date,
        'Days_Out_of_Compliance' : np.sum(client_drift > Drift_threshold),
        'Health_Score' : np.mean(client_drift) * 100 
    })

summary_df = pd.DataFrame(summary_df)
print("\n--Client Portfolio Health Summary--")
print(summary_df)

summary_df.to_csv('client_portfolio_health_summary.csv', index=False)

# PHASE 3: THE BRAIN (VOLATILITY & RISK ANALYSIS)

#1. Calculate Daily Returns
total_values_df = pd.DataFrame(total_values, index=close_prices.index, columns=df['Name'])
daily_percent_change = total_values_df.pct_change().dropna()

#2. Compute Annualized Volatility (Risk)
annualized_volatility = daily_percent_change.std() * np.sqrt(252)

#3. Risk-Adjusted Performance (Sharpe Ratio)
risk_free_rate = 0.02 
annual_average_return = daily_percent_change.mean() * 252
sharpe_ratios = (annual_average_return - risk_free_rate) / annualized_volatility

from scipy import stats 

# scipy is used to find the threshold of the worst 5% of days.
mean_return = daily_percent_change.mean()
std_return = daily_percent_change.std()
var_95 = stats.norm.ppf(0.95, loc=mean_return, scale=std_return)


#4. Generate a Risk Report
print("\n--Risk Analysis Report--")

for i, name in enumerate(df['Name']):
    vol = annualized_volatility[name]
    sharpe = sharpe_ratios[name]
    
    # Access the NumPy array using the integer index 'i'
    client_var = var_95[i] 

    risk_profile = "Aggressive/High" if vol > 0.18 else "Balanced/Moderate"

    print(f"Client: {name}")
    print(f"Annualized Volatility: {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"1-Day VaR (95%): {client_var:.2%}")
    print(f"Category: {risk_profile}\n")


# PHASE 4: Predictive Intelligence & Client Retention Modeling

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. CREATE SCALED DATA (1,000 Simulated Clients)
# This proves your model works for a whole bank, not just 5 rows
n_samples = 1000
np.random.seed(42)

sim_data = pd.DataFrame({
    'Max_Drift': np.random.uniform(0.01, 0.25, n_samples),
    'Volatility': np.random.uniform(0.05, 0.40, n_samples),
    'Sharpe_Ratio': np.random.uniform(-0.5, 2.5, n_samples)
})

# 2. THE CHURN LOGIC (The "Target")
# We add a bit of randomness so the model has to actually "think"
base_churn = ((sim_data['Max_Drift'] > 0.12) | (sim_data['Sharpe_Ratio'] < 0.3)).astype(int)
noise = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10]) # 10% unpredictable behavior
y = np.where(noise == 1, 1 - base_churn, base_churn)

# 3. TRAIN THE BRAIN
X_train, X_test, y_train, y_test = train_test_split(sim_data, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. APPLY TO YOUR REAL CLIENTS (The Impact)
# Let's see which of your actual CSV clients are likely to leave
real_clients_features = pd.DataFrame({
    'Max_Drift': summary_df['Max_Drift'],
    'Volatility': annualized_volatility.values,
    'Sharpe_Ratio': sharpe_ratios.values
})

# Get the probability (e.g., 0.85 instead of just "Yes")
churn_probs = model.predict_proba(real_clients_features)[:, 1]

print("\n-- AI-Driven Retention Priority List --")
for i, name in enumerate(df['Name']):
    prob = churn_probs[i]
    status = "HIGH RISK" if prob > 0.7 else "STABLE"
    print(f"Client: {name} | Churn Probability: {prob:.1%} | Status: {status}")

# 5. THE BUSINESS INSIGHT
importances = pd.Series(model.feature_importances_, index=sim_data.columns)
print("\n-- What is driving clients away? (Feature Importance) --")
print(importances.sort_values(ascending=False))

# Phase 5: The Suitability Engine (Regulatory Shield)

# 1. Define the Compliance Logic
def check_suitability(row):
    # Rule 1: Age-based risk limit (Senior clients shouldn't be too heavy in stocks)
    if row['Age'] > 60 and row['Current_Stock_Weight'] > 0.70:
        return "FAIL: Excessive Age-Based Risk"
    
    # Rule 2: Concentration Limit (No one should be > 90% in one asset)
    if row['Current_Stock_Weight'] > 0.90:
        return "FAIL: Concentration Breach"
    
    # Rule 3: Extreme Drift (A 15% drift is a major management failure)
    if row['Max_Drift'] > 0.15:
        return "FAIL: Extreme Portfolio Drift"
    
    return "PASS"

# 2. Build the Master Report
# We combine the base info (df), health stats (summary_df), and ML (churn_probs)
master_report = pd.DataFrame({
    'Name': df['Name'],
    'Age': df['Age'],
    'Current_Stock_Weight': current_stock_weights[-1], # Weights from the last day
    'Max_Drift': summary_df['Max_Drift'],
    'Churn_Prob': churn_probs,
    'Initial Value': df['Initial_Value']
})

# 3. Apply the Suitability Shield
master_report['Suitability_Status'] = master_report.apply(check_suitability, axis=1)

# 4. Final Output
print("\n" + "="*50)
print("PHASE 5: FINAL REGULATORY & RETENTION REPORT")
print("="*50)
print(master_report[['Name', 'Age', 'Suitability_Status', 'Churn_Prob']])

# Save the final result
master_report.to_csv('final_bank_report.csv', index=False)

# Phase 6: The Rebalancing Engine (The Fix)

def generate_trade_list(row):
    
    target_stock_dollars = row['Total_Value'] * row['Target_Stock_Weight']
    target_bond_dollars = row['Total_Value'] * row['Target_Bond_Weight']

    stock_gap = target_stock_dollars - row['Current_Stock_Value']
    bond_gap = target_bond_dollars - row['Current_Bond_Value']

    stock_action = f"SELL ${-stock_gap:.2f}" if stock_gap < 0 else f"BUY ${stock_gap:.2f}"
    bond_action = f"SELL ${-bond_gap:.2f}" if bond_gap < 0 else f"BUY ${bond_gap:.2f}"

    return stock_action, bond_action

# Prepare the data for the Rebalancer
master_report['Total_Value'] = total_values[-1] # Latest total value
master_report['Current_Stock_Value'] = stock_values[-1]
master_report['Current_Bond_Value'] = bond_values[-1]
master_report['Target_Stock_Weight'] = df['Stock_Weight']
master_report['Target_Bond_Weight'] = df['Bond_Weight']

# Apply the Rebalancing Logic
trades = master_report.apply(generate_trade_list, axis=1)
master_report['Stock_Trade'], master_report['Bond_Trade'] = zip(*trades)

print("\n" + "="*50)
print("PHASE 6: TRADE EXECUTION LIST (REBALANCER)")
print("="*50)
print(master_report[['Name', 'Suitability_Status', 'Stock_Trade', 'Bond_Trade']])