import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, silhouette_score, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="End-to-End Data Science Use Cases", layout="wide")
st.title("?? End-to-End Data Science Use Cases Dashboard")

tabs = st.tabs([
    "Use Case 1: PCA",
    "Use Case 2: Clustering",
    "Use Case 3: RFM + LDA",
    "Use Case 4: Time Series Forecasting"
])

# -----------------------------
# USE CASE 1: PCA
# -----------------------------
with tabs[0]:
    st.header("Use Case 1: PCA")

    df = pd.read_csv("retail_sales_dataset.csv")
    customer_df = df.groupby("Customer ID").agg({
        "Quantity": "sum",
        "Price per Unit": "mean",
        "Total Amount": "sum",
        "Age": "mean"
    }).reset_index()

    X = customer_df.drop("Customer ID", axis=1)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
    ax.set_title("PCA Projection")
    st.pyplot(fig)

# -----------------------------
# USE CASE 2: CLUSTERING
# -----------------------------
with tabs[1]:
    st.header("Use Case 2: Clustering (KMeans)")

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    st.write("Silhouette Score:", sil)

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis")
    ax.set_title("KMeans Clusters")
    st.pyplot(fig)

# -----------------------------
# USE CASE 3: RFM + LDA
# -----------------------------
with tabs[2]:
    st.header("Use Case 3: RFM + LDA")

    df = pd.read_csv("online_retail_II.csv")
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["Price"]

    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot - x.max()).days,
        "Invoice": "nunique",
        "TotalAmount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    rfm["Segment"] = pd.qcut(
        rfm["Monetary"], 3, labels=["Low", "Medium", "High"]
    )

    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Segment"]

    X_scaled = StandardScaler().fit_transform(X)
    lda = LiYSr4KS8aniprJvmct1CJfoVGgKnEGSjs()
    lda.fit(X_scaled, y)

    preds = lda.predict(X_scaled)
    cm = confusion_matrix(y, preds)

    st.write("Confusion Matrix")
    st.dataframe(cm)

# -----------------------------
# USE CASE 4: TIME SERIES
# -----------------------------
with tabs[3]:
    st.header("Use Case 4: Time Series Forecasting")

    df = pd.read_csv("actual_matrix.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    ts = df.groupby("date").size().asfreq("D").fillna(0)

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    sarima = SARIMAX(
        train,
        order=(1,0,1),
        seasonal_order=(1,0,1,7)
    ).fit(disp=False)

    forecast = sarima.forecast(len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))

    st.write("SARIMAX RMSE:", rmse)

    fig, ax = plt.subplots()
    ax.plot(train.index, train, label="Train")
    ax.plot(test.index, test, label="Test")
    ax.plot(test.index, forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)
