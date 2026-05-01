#Phase 7: Integrated Wealth Intelligence Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data created by your engine
# This is the "Bridge" between your two files
df = pd.read_csv('final_bank_report.csv')

st.title("🏦 Private Wealth: Compliance & Risk Command Center")

# 2. Row 1: Big KPI Cards
# We calculate these using the 'df' variable we just loaded
col1, col2, col3 = st.columns(3)

# Use 'Initial_Value' for AUM since it represents the actual dollar amount
total_aum = df['Initial Value'].sum()
col1.metric("Total AUM", f"${total_aum:,.0f}")

# Count how many rows don't say "PASS"
breaches = len(df[df['Suitability_Status'] != "PASS"])
col2.metric("Critical Breaches", breaches)

# Calculate the average of the Churn column
avg_churn = df['Churn_Prob'].mean()
col3.metric("Avg. Churn Risk", f"{avg_churn:.1%}")

# 3. Show the Table (Optional but helpful to see it's working)
st.subheader("Client Audit List")
st.dataframe(df)

st.sidebar.header("Dashboard Filters")
client_search = st.sidebar.selectbox("Select a Client", df['Name'].unique())

# 2. THIS IS THE KEY: Create a subset of data for ONLY that person
filtered_df = df[df['Name'] == client_search]

# 3. Display Metrics with Error Handling
if not filtered_df.empty:
    # We use .iloc[0] to get the value for the specific row found
    
    # METRIC 1: Using Stock Weight since 'Total_Value' is missing in your CSV
    stock_val = filtered_df['Current_Stock_Weight'].iloc[0]
    col1.metric("Stock Weight", f"{stock_val:.1%}")
    
    # METRIC 2: Suitability Status (PASS/FAIL)
    suitability = filtered_df['Suitability_Status'].iloc[0]
    col2.metric("Suitability", suitability)
    
    # METRIC 3: Churn Risk from your ML model
    churn = filtered_df['Churn_Prob'].iloc[0]
    col3.metric("Churn Risk", f"{churn:.1%}")
else:
    st.error("No data found for the selected client.")

def color_compliance(val):
    color = '#ffcccc' if "FAIL" in str(val) else '#ccffcc'
    return f'background-color: {color}; color: black;'

st.dataframe(df.style.map(color_compliance, subset=['Suitability_Status']))

# The interacitve chart
st.subheader("Portfolio efficiency: Risk vs Reward")

fig = px.scatter(
    df, 
    x="Max_Drift",      
    y="Churn_Prob",     
    color="Suitability_Status",
    size="Age",         
    hover_name="Name",
    template="plotly_dark"
)

st.plotly_chart(fig, width='stretch')

# 4. The Audit Table (Already done)
st.subheader("Final Audit & Trade Instructions")
st.dataframe(df.style.map(color_compliance, subset=['Suitability_Status']))

# The Compliance "Health Checck"

st.subheader("Overall Compliance Health")
fig_pie = px.pie(
    df, 
    names='Suitability_Status', 
    color='Suitability_Status',
    color_discrete_map={'PASS': '#2ECC71', 'FAIL: Excessive Age-Based Risk': '#E74C3C'},
    hole=0.4 # Makes it a Donut Chart (looks more modern)
)
st.plotly_chart(fig_pie, width='stretch')

# Churn Probability Distribution 
st.subheader("Churn Risk Distribution")
fig_hist = px.histogram(
    df, 
    x="Churn_Prob", 
    nbins=10,
    title="Frequency of Client Churn Risk",
    color_discrete_sequence=['#5DADE2'] # Professional Blue
)
st.plotly_chart(fig_hist, width='stretch')

# Risk Factor Correlation Matrix

st.subheader("Risk Factor Correlation Matrix")

# 1. Select only the numerical columns for the heatmap
# Based on your CSV columns: Name, Age, Current_Stock_Weight, Max_Drift, Churn_Prob
numeric_df = df[['Age', 'Current_Stock_Weight', 'Max_Drift', 'Churn_Prob']]

# 2. Create the correlation matrix
corr = numeric_df.corr()

# 3. Plot using Matplotlib and Seaborn
fig_heat, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
ax.set_title("How Risk Variables Interact")

st.pyplot(fig_heat)