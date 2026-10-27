"""
Restaurant Customer Preference Analysis
Course-End Project - Exploratory Data Analysis and Visualization (EDAV 22ADC31N)

Run:  python restaurant_analysis.py

What it does
  1. Creates the simulated dataset (NumPy, seed 42) and saves it as CSV
  2. Q1  Load, inspect, index
  3. Q2  Clean (missing values, duplicates, types, IQR outliers, min-max scaling)
  4. Q3  Transform, bin, sort, rank
  5. Q4  MultiIndex, group by, merge, reshape, time series
  6. Q5  Matplotlib / Seaborn charts + NumPy operations
  7. Saves cleaned data, summary tables and all figures to disk

Outputs
  data/restaurant_data_raw.csv      raw data (1,015 rows, messy)
  data/restaurant_data_clean.csv    cleaned data (999 rows)
  data/city_info.csv                lookup table used for the merge
  tables/*.csv                      summary tables
  figures/*.png                     all charts
"""

import os

import matplotlib
matplotlib.use("Agg")          # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs("data", exist_ok=True)
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Setup and data creation
# ----------------------------------------------------------------------
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "age": np.random.randint(15, 70, n),
    "city": np.random.choice(["Hyderabad", "Bengaluru", "Chennai", "Mumbai"], n),
    "cuisine": np.random.choice(
        ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food"],
        n, p=[0.28, 0.25, 0.17, 0.12, 0.18]),
    "order_mode": np.random.choice(["Dine-in", "Takeaway", "Delivery"], n, p=[0.5, 0.2, 0.3]),
    "party_size": np.random.randint(1, 9, n),
    "rating": np.round(np.random.normal(4.0, 0.7, n).clip(1, 5), 1),
    "visit_date": pd.date_range("2025-01-01", periods=n, freq="8h"),
    "payment_mode": np.random.choice(["Cash", "Card", "UPI"], n, p=[0.2, 0.3, 0.5]),
    "loyalty_member": np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
})
df["bill_amount"] = np.round(150 * df["party_size"] + np.random.normal(100, 80, n), 2).clip(80)

# Inject messy data so that cleaning is meaningful
df.loc[np.random.choice(n, 40, replace=False), "rating"] = np.nan
df.loc[np.random.choice(n, 25, replace=False), "payment_mode"] = np.nan
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)   # duplicates
df.loc[5, "bill_amount"] = 9000                                          # outlier

# Save the raw data, then load it back (this is the "load from CSV" step)
df.to_csv("data/restaurant_data_raw.csv", index=False)
df = pd.read_csv("data/restaurant_data_raw.csv", parse_dates=["visit_date"])

# ----------------------------------------------------------------------
# Q1 [CO1, BL2]: Loading, indexing, slicing and basic operations
# ----------------------------------------------------------------------
print("=" * 70, "\nQ1: Loading, indexing, slicing\n", "=" * 70, sep="")

s = pd.Series([250, 480, 320], index=["Mon", "Tue", "Wed"])
print(s)

print(df.head())
print(df.shape)
print(df.dtypes)
df.info()
print(df.describe())

print(df["cuisine"].head())
print(df[["cuisine", "bill_amount"]].head())
print(df.loc[0:4, ["age", "city"]])
print(df.iloc[0:5, 0:3])
print(df[df["bill_amount"] > 1000].head())

print(df.columns, df.index, df.size, df.ndim)
print(df["bill_amount"].mean(), df["bill_amount"].max())

raw_mean_bill = df["bill_amount"].mean()
raw_max_bill = df["bill_amount"].max()
raw_mean_rating = df["rating"].mean()

# ----------------------------------------------------------------------
# Q2 [CO2, BL3]: Cleaning and preprocessing
# ----------------------------------------------------------------------
print("=" * 70, "\nQ2: Cleaning\n", "=" * 70, sep="")

print(df.isnull().sum())

df["rating"] = df["rating"].fillna(df["rating"].median())
df["payment_mode"] = df["payment_mode"].fillna(df["payment_mode"].mode()[0])
# df.dropna(subset=["age"], inplace=True)   # alternative for critical columns

print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

df["loyalty_member"] = df["loyalty_member"].replace({"Yes": 1, "No": 0}).astype(int)
df["age"] = df["age"].astype(int)
df["visit_date"] = pd.to_datetime(df["visit_date"])

# Outlier treatment (IQR rule)
q1, q3 = df["bill_amount"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[(df["bill_amount"] >= low) & (df["bill_amount"] <= high)]

# Min-Max normalization
b = df["bill_amount"]
df["bill_scaled"] = (b - b.min()) / (b.max() - b.min())
print(df.isnull().sum().sum(), df.shape)

# ----------------------------------------------------------------------
# Q3 [CO3, BL4]: Transformation, categorisation and ranking
# ----------------------------------------------------------------------
print("=" * 70, "\nQ3: Transformation and ranking\n", "=" * 70, sep="")

df["payment_mode"] = df["payment_mode"].map(
    {"Cash": "Cash", "Card": "Card", "UPI": "Digital (UPI)"})
df = df.rename(columns={"bill_amount": "bill_inr"})

df["age_group"] = pd.cut(df["age"], bins=[0, 19, 34, 54, 100],
                         labels=["Teen", "Young Adult", "Adult", "Senior"])
df["spend_level"] = pd.cut(df["bill_inr"], bins=[0, 400, 900, np.inf],
                           labels=["Low", "Medium", "High"])

print(df.sort_values("bill_inr", ascending=False).head(10))
df["bill_rank"] = df["bill_inr"].rank(ascending=False)
print(df[(df["rating"] >= 4.5) & (df["cuisine"] == "Italian")].head())

pref = df["cuisine"].value_counts()
print(pref)
print(pref.rank(ascending=False))
print(df["age_group"].value_counts())

# Save the cleaned dataset
df.to_csv("data/restaurant_data_clean.csv", index=False)

# ----------------------------------------------------------------------
# Q4 [CO4, BL4]: MultiIndex, merge, reshape, group by, time series
# ----------------------------------------------------------------------
print("=" * 70, "\nQ4: Hierarchical index, group by, time series\n", "=" * 70, sep="")

multi = df.set_index(["city", "cuisine"]).sort_index()
print(multi.loc["Hyderabad"].head())
print(multi.loc[("Hyderabad", "Chinese")].head())

print(df.groupby("cuisine")["bill_inr"].agg(["count", "mean", "sum"]))
print(df.groupby(["city", "order_mode"])["bill_inr"].sum())

city_info = pd.DataFrame({"city": ["Hyderabad", "Bengaluru", "Chennai", "Mumbai"],
                          "region": ["South", "South", "South", "West"]})
city_info.to_csv("data/city_info.csv", index=False)
merged = pd.merge(df, city_info, on="city", how="left")
part1, part2 = df.iloc[:500], df.iloc[500:]
print(pd.concat([part1, part2]).shape)

cc = df.groupby(["city", "cuisine"])["bill_inr"].sum()
wide = cc.unstack()
print(wide)
print(wide.stack().head())
pivot = df.pivot_table(index="city", columns="order_mode", values="bill_inr", aggfunc="mean")
print(pivot)

ts = df.set_index("visit_date").sort_index()
print(ts.loc["2025-03"].head())
monthly = ts["bill_inr"].resample("ME").sum()
weekly = ts["bill_inr"].resample("W").mean()
print(monthly)
ts["weekday"] = ts.index.day_name()
print(ts.groupby("weekday")["bill_inr"].mean())

# ----------------------------------------------------------------------
# Q5 [CO5, BL5]: Visualisation and NumPy operations
# ----------------------------------------------------------------------
print("=" * 70, "\nQ5: Visualisation and NumPy\n", "=" * 70, sep="")

sns.set_theme(style="whitegrid", palette="Set2")

# Fig 7.2 Line plot
monthly.plot(marker="o", figsize=(8, 4))
plt.title("Monthly Revenue"); plt.xlabel("Month"); plt.ylabel("Revenue (INR)")
save_fig("fig_7_2_monthly_revenue")

# Fig 7.1 Bar chart
df["cuisine"].value_counts().plot(kind="bar", color="teal", figsize=(7, 4))
plt.title("Orders by Cuisine"); plt.ylabel("Orders")
save_fig("fig_7_1_orders_by_cuisine")

# Fig 6.1 Histogram
plt.hist(df["bill_inr"], bins=25, color="orange", edgecolor="black", label="Bill")
plt.title("Bill Amount Distribution"); plt.xlabel("Bill (INR)"); plt.legend()
save_fig("fig_6_1_bill_histogram")

# Fig 7.3 Scatter plot
plt.scatter(df["party_size"], df["bill_inr"], alpha=0.5, c="purple")
plt.title("Bill vs Party Size"); plt.xlabel("Party Size"); plt.ylabel("Bill (INR)")
save_fig("fig_7_3_bill_vs_party_size")

# Fig 7.4 Box plot, Fig 7.5 Violin plot
sns.boxplot(x="order_mode", y="rating", data=df)
plt.title("Rating by Order Mode")
save_fig("fig_7_4_rating_by_order_mode")

sns.violinplot(x="cuisine", y="bill_inr", data=df)
plt.xticks(rotation=30); plt.title("Bill by Cuisine")
save_fig("fig_7_5_bill_by_cuisine")

# Fig 6.2 Heatmap, Fig 7.6 Pair plot
corr = df[["age", "party_size", "bill_inr", "rating"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
save_fig("fig_6_2_correlation_heatmap")

sns.pairplot(df[["age", "party_size", "bill_inr", "rating"]])
plt.savefig("figures/fig_7_6_pairplot.png", dpi=120)
plt.close()

# Fig 7.7 Pie chart
df["payment_mode"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(5, 5))
plt.title("Payment Mode Share"); plt.ylabel("")
save_fig("fig_7_7_payment_share")

# ---- Appendix A: additional charts (Fig 12.1 - 12.10) ----
pv = df.pivot_table(index="city", columns="cuisine", values="bill_inr", aggfunc="count")
sns.heatmap(pv, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Orders by City and Cuisine")
save_fig("fig_12_1_city_cuisine_heatmap")

df.groupby(["city", "order_mode"]).size().unstack().plot(kind="bar", stacked=True, figsize=(7, 4))
plt.title("Order Mode by City"); plt.ylabel("Orders"); plt.xticks(rotation=0)
save_fig("fig_12_2_order_mode_by_city")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ts.groupby("weekday")["bill_inr"].mean().reindex(weekday_order).plot(
    kind="bar", color="slateblue", figsize=(7, 4))
plt.title("Average Bill by Weekday"); plt.ylabel("Average bill (INR)")
save_fig("fig_12_3_avg_bill_by_weekday")

sns.boxplot(x="age_group", y="bill_inr", data=df)
plt.title("Bill Amount by Age Group")
save_fig("fig_12_4_bill_by_age_group")

sns.histplot(df["rating"], bins=15, kde=True, color="tomato")
plt.title("Distribution of Ratings")
save_fig("fig_12_5_rating_distribution")

weekly.plot(figsize=(8, 4), color="green")
plt.title("Weekly Average Bill"); plt.xlabel("Week"); plt.ylabel("Average bill (INR)")
save_fig("fig_12_6_weekly_avg_bill")

sns.countplot(x="cuisine", hue="loyalty_member", data=df)
plt.xticks(rotation=20); plt.title("Loyalty Membership by Cuisine")
save_fig("fig_12_7_loyalty_by_cuisine")

ts["bill_inr"].cumsum().plot(figsize=(8, 4), color="darkred")
plt.title("Cumulative Revenue"); plt.ylabel("Revenue (INR)")
save_fig("fig_12_8_cumulative_revenue")

sns.countplot(x="age_group", hue="spend_level", data=df)
plt.title("Spend Level by Age Group")
save_fig("fig_12_9_spend_by_age_group")

df.groupby("cuisine")["bill_inr"].mean().sort_values().plot(kind="barh", color="teal", figsize=(7, 4))
plt.title("Average Bill by Cuisine"); plt.xlabel("Average bill (INR)")
save_fig("fig_12_10_avg_bill_by_cuisine")

# ---- NumPy fundamentals and advanced operations ----
a1 = np.array([10, 20, 30, 40])                       # 1D
a2 = np.array([[1, 2, 3], [4, 5, 6]])                 # 2D
a3 = np.arange(24).reshape(2, 3, 4)                   # 3D
print(a2[1, 2], a2[:, 1], a1[1:3])                    # indexing, slicing
print(a2 + np.array([10, 20, 30]))                    # broadcasting
print(a2 * 2)                                         # arithmetic
print(a2 @ a2.T)                                      # matrix product
print(a2.T)                                           # transpose
print(a3.reshape(4, 6).shape)                         # reshape

bills = df["bill_inr"].to_numpy()
print(np.mean(bills), np.median(bills), np.std(bills), np.percentile(bills, 90))
grid = np.random.randint(100, 1000, (4, 3))           # random: 4 cities x 3 modes
print(grid.sum(axis=0), grid.mean(axis=1), grid.max())
print(np.random.choice(bills, 5))                     # random sample

# ----------------------------------------------------------------------
# Summary tables for Chapter 9 (saved as CSV)
# ----------------------------------------------------------------------
cuisine_tbl = df.groupby("cuisine").agg(
    orders=("bill_inr", "count"), avg_bill=("bill_inr", "mean"),
    avg_rating=("rating", "mean"), revenue=("bill_inr", "sum")
).sort_values("orders", ascending=False).round(2)

city_tbl = df.groupby("city").agg(
    orders=("bill_inr", "count"), avg_bill=("bill_inr", "mean"),
    avg_party_size=("party_size", "mean"), revenue=("bill_inr", "sum")
).sort_values("revenue", ascending=False).round(2)

age_tbl = df.groupby("age_group", observed=True).agg(
    orders=("bill_inr", "count"), avg_bill=("bill_inr", "mean"),
    avg_rating=("rating", "mean")).round(2)

weekday_tbl = ts.groupby("weekday")["bill_inr"].agg(orders="count", avg_bill="mean").reindex(
    weekday_order).round(2)

cuisine_tbl.to_csv("tables/table_9_2_cuisine_summary.csv")
city_tbl.to_csv("tables/table_9_3_city_summary.csv")
age_tbl.to_csv("tables/table_9_4_age_group_summary.csv")
weekday_tbl.to_csv("tables/table_9_5_weekday_summary.csv")
df.describe().round(2).to_csv("tables/table_6_1_summary_statistics.csv")

print("=" * 70, "\nKey results\n", "=" * 70, sep="")
print(cuisine_tbl, city_tbl, age_tbl, weekday_tbl, sep="\n\n")
print("\nCorrelation party_size vs bill:", round(df["party_size"].corr(df["bill_inr"]), 2))
print("Correlation bill vs rating    :", round(df["bill_inr"].corr(df["rating"]), 2))
print(f"Mean bill before/after cleaning: {raw_mean_bill:.2f} / {df['bill_inr'].mean():.2f}")
print(f"Max bill before/after cleaning : {raw_max_bill:.2f} / {df['bill_inr'].max():.2f}")
print(f"Mean rating before/after       : {raw_mean_rating:.2f} / {df['rating'].mean():.2f}")
print("\nDone. See data/, tables/ and figures/.")
