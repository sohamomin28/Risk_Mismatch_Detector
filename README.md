# Portfolio drift monitoring system 📉 🛡️

A real-time Investment Portfolio Monitoring System and Regulatory Compliance Engine. This tool tracks "Portfolio Drift," calculates advanced risk metrics (VaR/Sharpe), and uses Machine Learning to predict client churn before it happens.

## 📌 Overview
The **Risk Mismatch Detector** serves as a "Digital Compliance Officer" for wealth management firms. In a volatile market, a balanced 50/50 portfolio can accidentally drift into a risky 70/30 split. This engine automates the detection of these imbalances, calculates risk exposure, and generates the exact trades needed to rebalance the account, ensuring clients stay within their legal and personal risk mandates.

## ⚠️ The Problem: "Manual Oversight & Portfolio Drift"
Human advisors often manage hundreds of clients. Checking for "Drift" (when market moves change a client's risk profile) manually is slow and error-prone. Missing a drift or a suitability breach can lead to massive financial loss for the client and legal fines for the bank.

## ✨ Features
*   **Real-Time Drift Tracking:** Uses `yfinance` to pull live market data and calculate variance against target weights via vectorized matrix math.
*   **Predictive Churn Modeling:** A **Random Forest** classifier identifying which clients are likely to leave based on portfolio stress and performance.
*   **Regulatory Shield:** Hard-coded compliance gates that flag "high-risk" allocations for sensitive client profiles (e.g., age-based stock limits).
*   **Advanced Risk Analytics:** Calculates **Value at Risk (VaR)** and **Sharpe Ratio** to measure efficiency and "worst-case" daily loss scenarios.
*   **Automated Rebalancing Engine:** Generates precise trade instructions to restore portfolios to their target state.

## 🛠 Tech Stack
*   **Language:** Python (Pandas, NumPy for Matrix Math)
*   **Data Acquisition:** `yfinance` API
*   **Machine Learning:** Scikit-Learn (Random Forest)
*   **Visualization:** Plotly & Streamlit
*   **Interface:** Streamlit (Integrated Wealth Intelligence Dashboard)

## 🏗 System Architecture
The engine processes data through a multi-stage pipeline:
1.  **Ingestion:** Loads `clients.csv` and converts dollar allocations into physical ETF shares (SPY/AGG).
2.  **Analysis:** Performs vectorized calculations to monitor daily drift and volatility for the entire client book.
3.  **Modeling:** Trains a predictive AI model to link drift and low performance to potential client attrition.
4.  **Compliance & Fix:** Runs suitability checks and calculates the "Execution Gap" to output trade instructions.
5.  **Reporting:** Exports a `final_bank_report.csv` for management oversight.

## 📂 Folder Structure
*   **app.py**: Main Application Entrypoint. Contains the Streamlit UI, the math engine, and the ML logic.
*   **clients.csv**: The input database of client names, initial investments, and target strategies.
*   **client_portfolio_health_summary.csv**: A detailed log of daily drift, breaches, and risk metrics for every client.
*   **final_bank_report.csv**: The executive summary identifying high-risk clients and the specific trades required to fix them.

## 🚀 Impact
This system transforms a bank from **Reactive** to **Proactive**. By identifying "High Risk" clients automatically through explainable metrics, an advisor can intervene, fix the portfolio drift, and save the relationship before the client decides to churn.

## 👤 Contact
**Project Lead:** Soha Momin  
**Email:** msoha28@my.yorku.ca
