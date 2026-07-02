import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")

# -------------------------------------------------
# SIMPLE LOCK / LOGIN SYSTEM
# -------------------------------------------------
USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Demand Forecast Dashboard Login")
    st.write("Enter username and password to open dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# -------------------------------------------------
# PREPROCESS DATA
# -------------------------------------------------
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()

    required_cols = ["date", "store", "item", "sales"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["store"] = pd.to_numeric(df["store"], errors="coerce")
    df["item"] = pd.to_numeric(df["item"], errors="coerce")

    df = df.dropna(subset=["date", "store", "item", "sales"]).copy()

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["month_name"] = df["date"].dt.strftime("%B")

    return df

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
def train_model(df):
    features = ["store", "item", "year", "month", "day"]
    X = df[features]
    y = df["sales"]

    if len(df) < 10:
        st.warning("Dataset is too small. Please upload at least 10 rows.")
        st.stop()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_iter=500,
        learning_rate=0.08,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # Training predictions
    train_pred = model.predict(X_train)
    train_results_df = df.loc[idx_train, ["date", "store", "item", "sales", "month_name"]].copy()
    train_results_df["predicted_sales"] = np.round(train_pred, 2)
    train_results_df["difference"] = np.round(train_results_df["sales"] - train_results_df["predicted_sales"], 2)
    train_results_df["abs_difference"] = np.round(np.abs(train_results_df["difference"]), 2)

    # Test predictions
    test_pred = model.predict(X_test)
    test_results_df = df.loc[idx_test, ["date", "store", "item", "sales", "month_name"]].copy()
    test_results_df["predicted_sales"] = np.round(test_pred, 2)
    test_results_df["difference"] = np.round(test_results_df["sales"] - test_results_df["predicted_sales"], 2)
    test_results_df["abs_difference"] = np.round(np.abs(test_results_df["difference"]), 2)

    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)

    return (
        model,
        train_results_df.sort_values("date"),
        test_results_df.sort_values("date"),
        mae,
        rmse,
        r2
    )

# -------------------------------------------------
# MONTHLY / SEASONAL TABLE
# -------------------------------------------------
def create_seasonal_table(results_df):
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    seasonal_df = results_df.groupby("month_name", as_index=False).agg(
        actual_sales=("sales", "sum"),
        predicted_sales=("predicted_sales", "sum")
    )

    seasonal_df["difference"] = np.round(seasonal_df["actual_sales"] - seasonal_df["predicted_sales"], 2)
    seasonal_df["abs_difference"] = np.round(np.abs(seasonal_df["difference"]), 2)
    seasonal_df["month_num"] = seasonal_df["month_name"].map(month_order)
    seasonal_df = seasonal_df.sort_values("month_num").drop(columns=["month_num"])

    seasonal_df["actual_sales"] = seasonal_df["actual_sales"].round(2)
    seasonal_df["predicted_sales"] = seasonal_df["predicted_sales"].round(2)

    return seasonal_df

# -------------------------------------------------
# DOWNLOAD HELPER
# -------------------------------------------------
def to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")

# -------------------------------------------------
# TOP QUESTIONS
# -------------------------------------------------
def show_top_questions(df):
    st.subheader("📌 Top 10 Important Questions from Dashboard")

    highest_sale_row = df.loc[df["sales"].idxmax()]
    lowest_sale_row = df.loc[df["sales"].idxmin()]

    item_sales = df.groupby("item")["sales"].sum().sort_values(ascending=False)
    highest_item = item_sales.idxmax()
    highest_item_value = item_sales.max()
    lowest_item = item_sales.idxmin()
    lowest_item_value = item_sales.min()

    store_sales = df.groupby("store")["sales"].sum().sort_values(ascending=False)
    highest_store = store_sales.idxmax()
    highest_store_value = store_sales.max()
    lowest_store = store_sales.idxmin()
    lowest_store_value = store_sales.min()

    month_sales = df.groupby("month_name")["sales"].sum().sort_values(ascending=False)
    best_month = month_sales.idxmax()
    best_month_value = month_sales.max()
    worst_month = month_sales.idxmin()
    worst_month_value = month_sales.min()

    avg_sales = df["sales"].mean()
    total_sales = df["sales"].sum()

    questions = [
        f"1. Which date has the highest sale?  →  {highest_sale_row['date'].date()} | Store {highest_sale_row['store']} | Item {highest_sale_row['item']} | Sales {highest_sale_row['sales']}",
        f"2. Which date has the lowest sale?  →  {lowest_sale_row['date'].date()} | Store {lowest_sale_row['store']} | Item {lowest_sale_row['item']} | Sales {lowest_sale_row['sales']}",
        f"3. Which item has the highest total sales?  →  Item {highest_item} with total sales {highest_item_value}",
        f"4. Which item has the lowest total sales?  →  Item {lowest_item} with total sales {lowest_item_value}",
        f"5. Which store has the highest total sales?  →  Store {highest_store} with total sales {highest_store_value}",
        f"6. Which store has the lowest total sales?  →  Store {lowest_store} with total sales {lowest_store_value}",
        f"7. Which month has the highest sales?  →  {best_month} with total sales {best_month_value}",
        f"8. Which month has the lowest sales?  →  {worst_month} with total sales {worst_month_value}",
        f"9. What is the average sales value?  →  {avg_sales:.2f}",
        f"10. What is the total sales?  →  {total_sales:.2f}",
    ]

    for q in questions:
        st.write("✅", q)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
def dashboard():
    st.title("📊 Online Product Demand Forecasting Dashboard")
    logout_button()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = preprocess_data(df)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"{df['sales'].sum():,.0f}")
        col2.metric("Average Sales", f"{df['sales'].mean():.2f}")
        col3.metric("Total Records", len(df))

        st.subheader("📈 Original Sales Trend")
        original_sales = df.groupby("date")["sales"].sum().reset_index()
        original_sales = original_sales.set_index("date")
        st.line_chart(original_sales)

        # Train model
        model, train_results_df, results_df, mae, rmse, r2 = train_model(df)

        st.subheader("🤖 Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("R² Score", f"{r2:.2f}")

        # Training graph for close fit
        st.subheader("🎯 Training Actual vs Predicted Sales Graph (Closer View)")
        train_graph_df = train_results_df[["date", "sales", "predicted_sales"]].copy()
        train_graph_df = train_graph_df.sort_values("date")
        train_graph_df = train_graph_df.groupby("date", as_index=False).agg({
            "sales": "sum",
            "predicted_sales": "sum"
        })
        train_graph_df = train_graph_df.set_index("date")
        st.line_chart(train_graph_df)

        # Training table download
        st.subheader("📋 Training Actual vs Predicted Table")
        st.dataframe(train_results_df)

        st.download_button(
            label="⬇ Download Training Actual vs Predicted CSV",
            data=to_csv_download(train_results_df),
            file_name="training_actual_vs_predicted.csv",
            mime="text/csv"
        )

        # Test table
        st.subheader("📋 Test Actual vs Predicted Sales Table")
        st.dataframe(results_df)

        st.download_button(
            label="⬇ Download Test Actual vs Predicted CSV",
            data=to_csv_download(results_df),
            file_name="actual_vs_predicted_sales.csv",
            mime="text/csv"
        )

        # Test graph
        st.subheader("📉 Test Actual vs Predicted Sales Graph")
        graph_df = results_df[["date", "sales", "predicted_sales"]].copy()
        graph_df = graph_df.sort_values("date")
        graph_df = graph_df.groupby("date", as_index=False).agg({
            "sales": "sum",
            "predicted_sales": "sum"
        })
        graph_df = graph_df.set_index("date")
        st.line_chart(graph_df)

        # Difference table
        st.subheader("📊 Difference Table")
        difference_df = results_df[["date", "store", "item", "sales", "predicted_sales", "difference", "abs_difference"]].copy()
        st.dataframe(difference_df)

        st.download_button(
            label="⬇ Download Difference CSV",
            data=to_csv_download(difference_df),
            file_name="difference_table.csv",
            mime="text/csv"
        )

        # Seasonal / monthly actual vs predicted
        seasonal_df = create_seasonal_table(results_df)

        st.subheader("🌦 Seasonal / Monthly Actual vs Predicted Table")
        st.dataframe(seasonal_df)

        st.download_button(
            label="⬇ Download Seasonal Actual vs Predicted CSV",
            data=to_csv_download(seasonal_df),
            file_name="seasonal_actual_vs_predicted.csv",
            mime="text/csv"
        )

        st.subheader("📈 Seasonal Actual vs Predicted Graph")
        seasonal_chart = seasonal_df.set_index("month_name")[["actual_sales", "predicted_sales"]]
        st.bar_chart(seasonal_chart)

        st.subheader("📉 Seasonal Difference Graph")
        seasonal_diff_chart = seasonal_df.set_index("month_name")[["difference"]]
        st.bar_chart(seasonal_diff_chart)

        # Store-wise and Item-wise
        st.subheader("🏪 Store-wise Sales")
        store_sales = df.groupby("store")["sales"].sum()
        st.bar_chart(store_sales)

        st.subheader("📦 Item-wise Sales")
        item_sales = df.groupby("item")["sales"].sum()
        st.bar_chart(item_sales)

        # Top questions
        show_top_questions(df)

    else:
        st.info("Please upload a CSV file to view the dashboard.")

# -------------------------------------------------
# APP FLOW
# -------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()