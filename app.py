import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY missing in .env file!")
else:
    openai.api_key = api_key

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Banking Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Banking Fraud Detection Dashboard (Agentic AI + MCP)")
st.write("Choose your input method, explore dashboard insights, and run AI analysis.")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.header("📥 Select Input Method")
    input_mode = st.radio(
        "Choose Data Source:",
        ["Upload CSV File", "Enter JSON Data Manually"]
    )

    st.markdown("---")
    st.header("⚙️ AI Settings")
    model_choice = st.selectbox(
        "Select AI Model:",
        ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=1
    )
    st.caption("✅ Using OpenAI Secure API")

# ---------------------------------------------------
# Fraud Detection Function
# ---------------------------------------------------
def detect_fraud(transactions, model_choice):
    try:
        response = openai.ChatCompletion.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a banking fraud detection expert."},
                {"role": "user", "content": f"""
Apply MCP Framework and detect fraud.

MODEL: Detect fraud in financial transactions.
CONTEXT:
- Amount > ₹50,000 → High Risk
- Foreign location → High Risk
- Night time (12–4 AM) → High Risk
- Unknown merchant → Medium Risk

POLICY:
- Give Risk Level
- Give Reason
- Give Confidence Score

TRANSACTIONS:
{transactions}
"""}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

# ---------------------------------------------------
# Load Data Based on Input Mode
# ---------------------------------------------------
df = None

if input_mode == "Upload CSV File":
    uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV File Uploaded Successfully!")

else:
    st.subheader("📥 Enter JSON Data")
    example_json = """[
        {"id": 1, "amount": 45000, "location": "Mumbai", "time": "14:30", "merchant": "Flipkart"},
        {"id": 2, "amount": 98000, "location": "Russia", "time": "02:10", "merchant": "Unknown Store"},
        {"id": 3, "amount": 1200, "location": "Mumbai", "time": "03:45", "merchant": "Zomato"},
        {"id": 4, "amount": 500000, "location": "Delhi", "time": "13:00", "merchant": "Luxury Watches"}
    ]"""
    user_json_input = st.text_area("Paste JSON here:", example_json, height=250)

    if st.button("✅ Load JSON Data"):
        try:
            data = json.loads(user_json_input)
            df = pd.DataFrame(data)
            st.success("✅ JSON Loaded Successfully!")
        except:
            st.error("❌ Invalid JSON Format!")

# ---------------------------------------------------
# If no data yet
# ---------------------------------------------------
if df is None:
    st.info("📤 Please upload a file or enter JSON to continue.")
    st.stop()

# ---------------------------------------------------
# Show Data Preview
# ---------------------------------------------------
st.subheader("📌 Data Preview")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# Risk Flag Logic
# ---------------------------------------------------
df["risk_flag"] = df.apply(
    lambda x: "High" if x["amount"] > 50000 or "Russia" in str(x.get("location", "")) else "Low",
    axis=1
)

# ---------------------------------------------------
# KPI Dashboard Cards
# ---------------------------------------------------
st.markdown("### 📊 Dashboard KPIs")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", len(df))
c2.metric("Total Amount (₹)", f"{df['amount'].sum():,.2f}")
c3.metric("High-Risk Transactions", len(df[df["risk_flag"] == "High"]))
c4.metric("Fraud Probability", f"{(len(df[df['risk_flag']=='High'])/len(df))*100:.2f}%")

st.markdown("---")

# ---------------------------------------------------
# Alerts
# ---------------------------------------------------
st.subheader("🚨 Fraud Alerts")

high_risk_df = df[df["risk_flag"] == "High"]

if not high_risk_df.empty:
    st.error(f"⚠️ {len(high_risk_df)} HIGH-RISK transactions detected!")
    st.dataframe(high_risk_df.style.background_gradient(cmap="Reds"))
else:
    st.success("✅ No high-risk transactions detected.")

st.markdown("---")

# ---------------------------------------------------
# Graphs
# ---------------------------------------------------
st.header("📈 Visualizations")

# ---- Graph 1: Amount Distribution ----
st.subheader("Transaction Amount Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.histplot(df["amount"], kde=True, ax=ax1, color="steelblue")
st.pyplot(fig1)
plt.close(fig1)

# ---- Graph 2: Location Count ----
if "location" in df.columns:
    st.subheader("Transactions by Location")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(y="location", data=df, palette="viridis", ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)

# ---- Graph 3: Risk Pie Chart ----
st.subheader("High vs Low Risk")
fig3, ax3 = plt.subplots()
risk_counts = df["risk_flag"].value_counts()
ax3.pie(
    risk_counts,
    labels=risk_counts.index,
    autopct="%1.1f%%",
    colors=["red", "green"],
    explode=[0.1, 0]
)
st.pyplot(fig3)
plt.close(fig3)

# ---- Graph 4: Scatter Plot ----
if "time" in df.columns:
    st.subheader("Amount vs Transaction Time")
    df["hour"] = pd.to_datetime(df["time"], errors="coerce").dt.hour
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df,
        x="hour",
        y="amount",
        hue="risk_flag",
        palette={"High": "red", "Low": "green"},
        s=120
    )
    st.pyplot(fig4)
    plt.close(fig4)

st.markdown("---")

# ---------------------------------------------------
# AI FRAUD DETECTION BUTTON
# ---------------------------------------------------
st.header("🤖 AI Fraud Detection (Agentic MCP)")

if st.button("🚀 Run AI Fraud Detection"):
    st.info("🔍 AI is analyzing your data... Please wait...")
    transactions_dict = df.to_dict(orient="records")
    result = detect_fraud(transactions_dict, model_choice)
    st.subheader("✅ AI Analysis Result")
    st.write(result)
