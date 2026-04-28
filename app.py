import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from groq import Groq

st.set_page_config(page_title="India EV Adoption Analysis", layout="wide")
st.title("Electric Vehicle Adoption Rate Analysis by State in India")

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("EV_Dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ── Detect key columns ─────────────────────────────────────────────────────────
col_state = [c for c in df.columns if "state" in c.lower()][0]
col_sales = [c for c in df.columns if any(k in c.lower() for k in ["sale", "count", "unit", "total"])][0]
col_cat   = next((c for c in df.columns if "categor" in c.lower()), None)
col_vtype = next((c for c in df.columns if "type" in c.lower()), None)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Model", "💬 Ask AI Analyst"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    state_sales = df.groupby(col_state)[col_sales].sum().sort_values(ascending=False).head(10)
    st.bar_chart(state_sales)

    bottom_states = df.groupby(col_state)[col_sales].sum().sort_values().head(10)
    st.bar_chart(bottom_states)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL (FIXED XGBOOST)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("ML Model — Predicting EV Sales (XGBoost)")

    df_model = df.copy()

    # Detect date column
    date_col = next((c for c in df_model.columns if "date" in c.lower()), None)

    # time features
    if date_col:
        df_model[date_col] = pd.to_datetime(df_model[date_col], errors="coerce")
        df_model["year"] = df_model[date_col].dt.year
        df_model["month"] = df_model[date_col].dt.month
        feature_cols = [col_state, col_cat, col_vtype, "year", "month"]
    else:
        feature_cols = [col_state, col_cat, col_vtype]

    feature_cols = [c for c in feature_cols if c is not None]

    
    model_df = df_model.groupby(feature_cols)[col_sales].sum().reset_index()

    # Encoding
    model_df_enc = pd.get_dummies(
        model_df,
        columns=[c for c in feature_cols if model_df[c].dtype == "object"]
    )

    X = model_df_enc.drop(col_sales, axis=1)
    y = model_df_enc[col_sales]

    
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Log transform
    y_log = np.log1p(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # Model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Metrics
    r2 = r2_score(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    st.metric("R² Score", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:,.0f}")

    # Scatter plot
    st.subheader("Actual vs Predicted EV Sales")
    fig, ax = plt.subplots()
    ax.scatter(y_test_actual, y_pred, alpha=0.5)
    ax.plot(
        [y_test_actual.min(), y_test_actual.max()],
        [y_test_actual.min(), y_test_actual.max()],
        "r--"
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI ANALYST (UNCHANGED LOGIC + ENV KEY SUPPORT)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Ask the AI Analyst")

    total_sales = int(df[col_sales].sum())
    top_state = df.groupby(col_state)[col_sales].sum().idxmax()

    data_context = f"""
    Total EV Sales: {total_sales}
    Top State: {top_state}
    """

    
    api_key = st.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY")
    question = st.text_input("Your question")

    if st.button("Ask AI") and api_key and question:
        try:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "user",
                    "content": f"{data_context}\n\nQuestion: {question}"
                }]
            )

            st.success(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")