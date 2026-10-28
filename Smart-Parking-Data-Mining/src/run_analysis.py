"""Starter pipeline for Smart Parking Pattern Mining.
The bundled parking dataset is synthetic practice data, not real-world data.
Run from the project root: python src/run_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "parking_data.csv"
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
MINING = RESULTS / "mining_results"
for folder in (CHARTS, MINING):
    folder.mkdir(parents=True, exist_ok=True)

def load_and_clean():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Entry_Time"] = pd.to_datetime(df["Entry_Time"], format="%H:%M", errors="coerce")
    df["Hour"] = df["Entry_Time"].dt.hour
    df["Day"] = df["Date"].dt.day_name()
    df["Day_Type"] = np.where(df["Date"].dt.dayofweek >= 5, "Weekend", "Working Day")
    for col in ["Parking_Duration_Min", "Parking_Fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Hour", "Vehicle_Type", "Slot_ID",
                           "Parking_Duration_Min", "Parking_Fee"])
    df = df[df["Parking_Duration_Min"] > 0].copy()
    df["Duration_Category"] = pd.cut(
        df["Parking_Duration_Min"], bins=[0, 60, 180, np.inf],
        labels=["Short", "Medium", "Long"], include_lowest=True
    ).astype(str)
    df.to_csv(RESULTS / "cleaned_parking_data.csv", index=False)
    return df

def create_eda(df):
    hourly = df.groupby("Hour").size().rename("Parking_Sessions").reset_index()
    hourly.to_csv(RESULTS / "sessions_by_hour.csv", index=False)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("Day").size().reindex(day_order, fill_value=0).rename("Parking_Sessions").reset_index()
    daily.to_csv(RESULTS / "sessions_by_day.csv", index=False)
    df.groupby("Slot_ID").size().rename("Sessions").sort_values(ascending=False).reset_index().to_csv(
        RESULTS / "slot_usage.csv", index=False
    )
    charts = [
        ("sessions_by_hour.png", lambda: sns.barplot(data=hourly, x="Hour", y="Parking_Sessions"),
         "Parking Sessions by Entry Hour (Synthetic Sample)"),
        ("sessions_by_day.png", lambda: sns.barplot(data=daily, x="Day", y="Parking_Sessions"),
         "Parking Sessions by Day (Synthetic Sample)"),
        ("vehicle_type_distribution.png", lambda: sns.countplot(data=df, x="Vehicle_Type"),
         "Vehicle Type Distribution (Synthetic Sample)")
    ]
    for filename, plotter, title in charts:
        plt.figure(figsize=(9, 5))
        plotter()
        plt.title(title)
        if filename == "sessions_by_day.png":
            plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(CHARTS / filename, dpi=150)
        plt.close()

def run_clustering(df):
    features = df[["Parking_Duration_Min", "Hour", "Parking_Fee"]]
    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    out = df.copy()
    out["Cluster"] = model.fit_predict(X)
    out.to_csv(MINING / "clustering_results.csv", index=False)
    out.groupby("Cluster")[["Parking_Duration_Min", "Hour", "Parking_Fee"]].mean().round(2).to_csv(
        MINING / "cluster_summary.csv"
    )

def run_association_rules(df):
    transactions = pd.get_dummies(df[["Day_Type", "Vehicle_Type", "Duration_Category"]]).astype(bool)
    frequent = apriori(transactions, min_support=0.05, use_colnames=True)
    if frequent.empty:
        pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]).to_csv(
            MINING / "association_rules.csv", index=False)
        return
    rules = association_rules(frequent, metric="confidence", min_threshold=0.4)
    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules.to_csv(MINING / "association_rules.csv", index=False)

def run_classification(df):
    # Demonstration only: hourly count-derived labels are not a real-world forecast.
    hourly_counts = df.groupby("Hour").size()
    low_cut, high_cut = hourly_counts.quantile([1/3, 2/3]).tolist()
    labels = {}
    for hour, count in hourly_counts.items():
        labels[hour] = "Low" if count <= low_cut else ("Medium" if count <= high_cut else "High")
    model_df = df.copy()
    model_df["Demand_Category"] = model_df["Hour"].map(labels)
    X = pd.get_dummies(model_df[["Day_Type", "Vehicle_Type"]], drop_first=False)
    y = model_df["Demand_Category"]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        model_df[["Record_ID", "Hour", "Demand_Category"]].to_csv(MINING / "classification_results.csv", index=False)
        (MINING / "classification_report.txt").write_text("Insufficient examples for a meaningful train/test split.\n")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0)
    (MINING / "classification_report.txt").write_text(
        "Synthetic-data teaching example; not a real-world forecast.\n\n"
        + f"Accuracy: {accuracy_score(y_test, pred):.3f}\n\n" + report)
    model_df["Predicted_Demand_Category"] = model.predict(X)
    model_df[["Record_ID", "Hour", "Demand_Category", "Predicted_Demand_Category"]].to_csv(
        MINING / "classification_results.csv", index=False)

def run_outlier_detection(df):
    X = df[["Parking_Duration_Min", "Parking_Fee"]]
    detector = IsolationForest(contamination=0.03, random_state=42)
    out = df.copy()
    out["Outlier_Flag"] = detector.fit_predict(X)
    out["Outlier_Status"] = np.where(out["Outlier_Flag"] == -1, "Potential outlier", "Typical")
    out[out["Outlier_Status"] == "Potential outlier"].to_csv(MINING / "potential_outliers.csv", index=False)

def main():
    print("Loading and cleaning synthetic sample data...")
    df = load_and_clean()
    print(f"Records after cleaning: {len(df)}")
    print("Creating EDA summaries and charts...")
    create_eda(df)
    print("Running K-Means clustering...")
    run_clustering(df)
    print("Running Apriori association rule mining...")
    run_association_rules(df)
    print("Running Decision Tree classification demonstration...")
    run_classification(df)
    print("Detecting potential outliers...")
    run_outlier_detection(df)
    print("\nFinished. Check the results/ folder.")
    print("Reminder: the supplied dataset is synthetic practice data.")
if __name__ == "__main__":
    main()
