import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ── Raw Data Preview ───────────────────────────────────────────────────────────
with st.expander("Raw Data Preview"):
    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head(10))

# ── Detect key columns ─────────────────────────────────────────────────────────
col_state = [c for c in df.columns if "state" in c.lower()][0]
col_sales = [c for c in df.columns if any(k in c.lower() for k in ["sale", "count", "unit", "total"])][0]
col_cat   = next((c for c in df.columns if "categor" in c.lower()), None)
col_vtype = next((c for c in df.columns if "type"    in c.lower()), None)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Model", "💬 Ask AI Analyst"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    # 1. Top 10 States by EV Sales
    st.subheader("Top 10 States by EV Sales")
    state_sales = df.groupby(col_state)[col_sales].sum().sort_values(ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    state_sales.plot(kind="bar", ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_xlabel("State")
    ax1.set_ylabel("Total EV Sales")
    ax1.set_title("Top 10 States by EV Sales")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig1)

    # 2. Bottom 10 States
    st.subheader("Bottom 10 States by EV Sales")
    bottom_states = df.groupby(col_state)[col_sales].sum().sort_values().head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bottom_states.plot(kind="bar", ax=ax2, color="tomato", edgecolor="white")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Total EV Sales")
    ax2.set_title("Bottom 10 States — Low Adoption Regions")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)

    # 3. Sales by Vehicle Category
    if col_cat:
        st.subheader("Sales by Vehicle Category")
        cat_sales = df.groupby(col_cat)[col_sales].sum().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.pie(cat_sales, labels=cat_sales.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"], startangle=140)
        ax3.set_title("EV Sales Split by Vehicle Category")
        st.pyplot(fig3)

    # 4. Sales by Vehicle Type
    if col_vtype:
        st.subheader("Sales by Vehicle Type")
        type_sales = df.groupby(col_vtype)[col_sales].sum().sort_values(ascending=False)
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        type_sales.plot(kind="barh", ax=ax4, color="mediumseagreen", edgecolor="white")
        ax4.set_xlabel("Total Sales")
        ax4.set_title("EV Sales by Vehicle Type")
        ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        st.pyplot(fig4)

    # 5. Karnataka vs Top 5 States
    st.subheader("Karnataka vs Top 5 States")
    all_state_sales = df.groupby(col_state)[col_sales].sum().sort_values(ascending=False)
    top5 = all_state_sales.head(5)
    ka_val = df[df[col_state].str.contains("Karnataka", case=False)][col_sales].sum()
    compare = top5.copy()
    if not any("karnataka" in s.lower() for s in compare.index):
        compare["Karnataka"] = int(ka_val)
    colors = ["gold" if "karnataka" in s.lower() else "cornflowerblue" for s in compare.index]
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    compare.plot(kind="bar", ax=ax5, color=colors, edgecolor="white")
    ax5.set_title("Karnataka vs Top 5 States  (Gold = Karnataka)")
    ax5.set_ylabel("Total EV Sales")
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("ML Model — Predicting EV Sales (XGBoost)")

    df_model = df.copy()

    # Detect date column
    date_col = next((c for c in df_model.columns if "date" in c.lower()), None)

    # Add time features if date exists
    if date_col:
        df_model[date_col] = pd.to_datetime(df_model[date_col], errors="coerce")
        df_model["year"]   = df_model[date_col].dt.year
        df_model["month"]  = df_model[date_col].dt.month
        feature_cols = [col_state, col_cat, col_vtype, "year", "month"]
    else:
        feature_cols = [col_state, col_cat, col_vtype]

    feature_cols = [c for c in feature_cols if c is not None]

    # Aggregate
    model_df = df_model.groupby(feature_cols)[col_sales].sum().reset_index()

    # Encode categorical columns
    model_df_enc = pd.get_dummies(
        model_df,
        columns=[c for c in feature_cols if model_df[c].dtype == "object"]
    )

    X = model_df_enc.drop(col_sales, axis=1)
    y = model_df_enc[col_sales]

    # Clean numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Log transform
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # XGBoost
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

    # Predictions — reverse log transform
    y_pred_log    = model.predict(X_test)
    y_pred        = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Metrics
    r2   = r2_score(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("RMSE",     f"{rmse:,.0f} units")

    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    importance.sort_values().plot(kind="barh", ax=ax6, color="darkorange", edgecolor="white")
    ax6.set_title("Top 10 Feature Importances")
    ax6.set_xlabel("Importance Score")
    st.pyplot(fig6)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted EV Sales")
    fig7, ax7 = plt.subplots(figsize=(6, 5))
    ax7.scatter(y_test_actual, y_pred, alpha=0.5, color="teal")
    ax7.plot(
        [y_test_actual.min(), y_test_actual.max()],
        [y_test_actual.min(), y_test_actual.max()],
        "r--", linewidth=1.5
    )
    ax7.set_xlabel("Actual Sales")
    ax7.set_ylabel("Predicted Sales")
    ax7.set_title("Actual vs Predicted EV Sales")
    st.pyplot(fig7)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GROQ AI ANALYST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("💬 Ask the AI Analyst")
    st.write("Ask anything about India's EV adoption — powered by Groq (Llama 3).")

    # Rich data context from actual dataset
    total_sales    = int(df[col_sales].sum())
    top_state_name = df.groupby(col_state)[col_sales].sum().idxmax()
    top_state_val  = int(df.groupby(col_state)[col_sales].sum().max())
    num_states     = df[col_state].nunique()
    avg_per_state  = int(df.groupby(col_state)[col_sales].sum().mean())
    top5_list      = ", ".join(df.groupby(col_state)[col_sales].sum()
                                .sort_values(ascending=False).head(5).index.tolist())

    cat_text = ""
    if col_cat:
        c_data   = df.groupby(col_cat)[col_sales].sum().sort_values(ascending=False)
        cat_text = "\n".join([f"  - {k}: {v:,}" for k, v in c_data.items()])

    ka_total = int(df[df[col_state].str.contains("Karnataka", case=False)][col_sales].sum())

    data_context = f"""
India EV Sales Dataset Summary:
- Total EV sales across all states: {total_sales:,}
- Number of states in dataset: {num_states}
- Top state by EV sales: {top_state_name} ({top_state_val:,} units)
- Top 5 states: {top5_list}
- Average EV sales per state: {avg_per_state:,}
- Karnataka (home of Bengaluru): {ka_total:,} units
- Vehicle category breakdown:
{cat_text}
- Two-wheelers dominate EV sales in India
"""

    # Support both manual key input and environment variable
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free at console.groq.com — no credit card needed"
    ) or os.getenv("GROQ_API_KEY", "")

    question = st.text_input(
        "Your question",
        placeholder="Which state has the highest EV adoption and why?"
    )

    if st.button("Ask AI") and api_key and question:
        try:
            client = Groq(api_key=api_key)

            prompt = f"""You are a data analyst specializing in India's electric vehicle market.
Here is a summary of the dataset you are working with:

{data_context}

Answer the following question based on this data. Be concise, data-driven, and insightful.
If asked about EV policy or infrastructure in India, you may add brief general context.

Question: {question}"""

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.success(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")

    elif not api_key:
        st.info("👆 Enter your free Groq API key above to get started.")