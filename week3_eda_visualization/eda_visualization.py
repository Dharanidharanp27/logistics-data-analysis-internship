"""Week 3 - Exploratory data analysis and visualization of a simulated logistics dataset.

Run:  python eda_visualization.py
Output: figs/*.png (8 charts) and console statistics
"""
import os
os.makedirs("figs", exist_ok=True)



# ---- Simulate the dataset ----
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 8000

# --- Dates: 12 months, heavier volume in the festival window (15 Oct - 20 Nov)
days = np.arange(365)
w = np.where((days >= 14) & (days <= 50), 2.2, 1.0)
day = rng.choice(days, n, p=w / w.sum())
order_date = pd.Timestamp("2025-10-01") + pd.to_timedelta(day, unit="D")
festival = ((day >= 14) & (day <= 50)).astype(int)
weekend = (order_date.dayofweek >= 5).astype(int)

# --- Destination district and road distance from the central warehouse (km)
dist_base = {"Trichy": 10, "Pudukkottai": 55, "Perambalur": 55, "Thanjavur": 60,
             "Ariyalur": 70, "Karur": 80, "Madurai": 130, "Salem": 140}
district = rng.choice(list(dist_base), n, p=[.28, .10, .08, .14, .07, .10, .12, .11])
distance_km = np.array([dist_base[d] for d in district]) + rng.normal(0, 6, n)
distance_km = distance_km.clip(3)

# --- Shipment attributes
mode = rng.choice(["Standard", "Express", "Priority"], n, p=[.60, .30, .10])
sched = pd.Series(mode).map({"Standard": 3, "Express": 2, "Priority": 1}).values
weight_kg = rng.lognormal(3.0, 0.7, n).round(1)
units = rng.poisson(weight_kg / 4).clip(1)
van = rng.choice([f"V{i:02d}" for i in range(1, 13)], n)
dispatch_delay_hr = (rng.lognormal(1.0, 0.6, n) + 2 * festival).round(1)

# --- Delivery time: driven by distance, warehouse delay, festival load, weekends, weak vans
van_penalty = np.where(np.isin(van, ["V07", "V11"]), 0.7, 0.0)
delay = (-1.3 + 0.006 * distance_km + 0.05 * dispatch_delay_hr + 0.5 * festival
         + 0.2 * weekend + van_penalty + rng.normal(0, 0.45, n))
actual_days = np.maximum(1, sched + np.round(delay)).astype(int)

# --- Transport cost (INR): distance x mode rate, weight effect, festival surcharge
rate = pd.Series(mode).map({"Standard": 5.5, "Express": 7.5, "Priority": 10}).values
cost = 60 + distance_km * rate * (1 + weight_kg / 200) + 0.8 * weight_kg + 40 * festival + rng.normal(0, 40, n)

df = pd.DataFrame({
    "shipment_id": np.arange(1, n + 1), "order_date": order_date, "district": district,
    "distance_km": distance_km.round(1), "shipping_mode": mode, "weight_kg": weight_kg,
    "units": units, "van_id": van, "dispatch_delay_hr": dispatch_delay_hr,
    "scheduled_days": sched, "actual_days": actual_days, "transport_cost": cost.clip(50).round(2),
    "festival_season": festival, "weekend_order": weekend})

df["delay_days"] = df["actual_days"] - df["scheduled_days"]
df["is_late"] = (df["delay_days"] > 0).astype(int)
df["cost_per_km"] = df["transport_cost"] / df["distance_km"]
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["weekday"] = df["order_date"].dt.day_name()
print(df.shape); print(df.head(3).T)


# ---- Central tendency and spread ----
key = ["distance_km", "weight_kg", "units", "dispatch_delay_hr",
       "actual_days", "delay_days", "transport_cost", "cost_per_km"]

summary = df[key].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
print(summary)

print("Overall on-time rate: %.1f%%" % ((1 - df["is_late"].mean()) * 100))
print("Mode of delivery days:", df["actual_days"].mode()[0])
print(df.groupby("shipping_mode")[["transport_cost", "cost_per_km", "delay_days"]].mean().round(2))


# ---- Fig 1: monthly trend ----
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.0)
BLUE, ORANGE, RED = "#1F4E79", "#E07B00", "#C0392B"

# Figure 1: monthly volume (bars) and on-time rate (line) - trend over time
monthly = df.groupby("month").agg(shipments=("shipment_id", "count"),
                                  ontime=("is_late", lambda s: (1 - s.mean()) * 100))
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(monthly.index, monthly["shipments"], color=BLUE, alpha=.8)
ax1.set_ylabel("Shipments"); ax1.tick_params(axis="x", rotation=45)
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["ontime"], color=ORANGE, marker="o", lw=2)
ax2.set_ylabel("On-time rate (%)"); ax2.grid(False)
ax1.set_title("Monthly Shipment Volume vs On-Time Delivery Rate")
plt.tight_layout(); plt.savefig("figs/fig1_trend.png", dpi=150); plt.close()


# ---- Fig 2-3: delay distribution and cost by mode ----
# Figure 2: distribution of delivery delay (days beyond promise)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["delay_days"], discrete=True, color=BLUE, ax=ax)
ax.axvline(0, color=RED, ls="--", label="Promised date")
ax.set_xlabel("Delay (actual - scheduled days)"); ax.set_ylabel("Shipments"); ax.legend()
ax.set_title("Distribution of Delivery Delay")
plt.tight_layout(); plt.savefig("figs/fig2_delay_dist.png", dpi=150); plt.close()

# Figure 3: transport cost by shipping mode (box plot shows spread and outliers)
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="shipping_mode", y="transport_cost",
            order=["Standard", "Express", "Priority"], palette=[BLUE, ORANGE, RED], ax=ax)
ax.set_xlabel("Shipping mode"); ax.set_ylabel("Transport cost (INR)")
ax.set_title("Transport Cost by Shipping Mode")
plt.tight_layout(); plt.savefig("figs/fig3_cost_mode.png", dpi=150); plt.close()


# ---- Fig 4-5: correlation and distance vs cost ----
# Figure 4: correlation heatmap of numeric variables
num = ["distance_km", "weight_kg", "units", "dispatch_delay_hr",
       "scheduled_days", "actual_days", "delay_days", "transport_cost"]
corr = df[num].corr().round(2)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix of Shipment Variables")
plt.tight_layout(); plt.savefig("figs/fig4_corr.png", dpi=150); plt.close()

# Figure 5: distance vs cost, coloured by mode, with fitted lines
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(data=df.sample(1500, random_state=1), x="distance_km", y="transport_cost",
                hue="shipping_mode", hue_order=["Standard", "Express", "Priority"],
                palette=[BLUE, ORANGE, RED], alpha=.5, s=18, ax=ax)
for m, col in zip(["Standard", "Express", "Priority"], [BLUE, ORANGE, RED]):
    d = df[df["shipping_mode"] == m]
    k, b0 = np.polyfit(d["distance_km"], d["transport_cost"], 1)
    xs = np.linspace(0, 150, 50); ax.plot(xs, k * xs + b0, color=col, lw=2)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Transport cost (INR)")
ax.set_title("Transport Cost vs Distance by Shipping Mode")
plt.tight_layout(); plt.savefig("figs/fig5_dist_cost.png", dpi=150); plt.close()


# ---- Fig 6-7: late rate by district and van ----
# Figure 6: late-delivery rate by district (ranked bar chart)
dist_late = (df.groupby("district")["is_late"].mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=dist_late.values, y=dist_late.index, color=BLUE, ax=ax)
ax.axvline(df["is_late"].mean() * 100, color=RED, ls="--", label="Network average")
ax.set_xlabel("Late deliveries (%)"); ax.set_ylabel(""); ax.legend()
ax.set_title("Late-Delivery Rate by District")
plt.tight_layout(); plt.savefig("figs/fig6_district_late.png", dpi=150); plt.close()

# Figure 7: late-delivery rate by van (bottleneck check)
van_late = (df.groupby("van_id")["is_late"].mean() * 100)
colors = [RED if v > van_late.mean() + 8 else BLUE for v in van_late]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(van_late.index, van_late.values, color=colors)
ax.axhline(df["is_late"].mean() * 100, color="black", ls="--", lw=1, label="Network average")
ax.set_ylabel("Late deliveries (%)"); ax.set_xlabel("Van"); ax.legend()
ax.set_title("Late-Delivery Rate by Van")
plt.tight_layout(); plt.savefig("figs/fig7_van_late.png", dpi=150); plt.close()


# ---- Fig 8: dispatch delay x season ----
# Figure 8: warehouse dispatch delay vs lateness, split by festival season
df["dispatch_bin"] = pd.cut(df["dispatch_delay_hr"], [0, 3, 4.5, 6, 100],
                            labels=["<3h", "3-4.5h", "4.5-6h", ">6h"])
pivot = (df.pivot_table(index="dispatch_bin", columns="festival_season",
                        values="is_late", aggfunc="mean", observed=True) * 100).round(1)
pivot.columns = ["Normal season", "Festival season"]
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Late (%)"}, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Warehouse dispatch delay")
ax.set_title("Late-Delivery Rate: Dispatch Delay x Season")
plt.tight_layout(); plt.savefig("figs/fig8_dispatch.png", dpi=150); plt.close()


# ---- Cost drivers (regression) and late-delivery summary ----
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Cost drivers: standardised regression coefficients (comparable across variables)
X = df[["distance_km", "weight_kg", "festival_season"]].copy()
X["express"] = (df["shipping_mode"] == "Express").astype(int)
X["priority"] = (df["shipping_mode"] == "Priority").astype(int)
y = df["transport_cost"]

Xs = StandardScaler().fit_transform(X)
lr = LinearRegression().fit(Xs, y)
drivers = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False).round(1)
print(drivers); print("R-squared: %.2f" % lr.score(Xs, y))

# Late-delivery drivers: share of late orders by condition
print(df.groupby("festival_season")["is_late"].mean().round(3))
print(df.groupby("weekend_order")["is_late"].mean().round(3))
print(df.groupby(df["van_id"].isin(["V07", "V11"]))["is_late"].mean().round(3))
