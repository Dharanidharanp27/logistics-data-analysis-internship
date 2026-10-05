"""Week 1 - Strategic planning: illustrative Python snippets for the proposed logistics analysis.

These are the code illustrations from the Week 1 report. File and column names
(orders.csv, products.csv ...) are placeholders; the full, runnable implementations
are in the Week 2-4 folders.
"""


# ======================================================================
# 1. Data loading and cleaning
# ======================================================================
import pandas as pd
import numpy as np

orders   = pd.read_csv("orders.csv", parse_dates=["order_date", "promised_date", "delivered_date"])
products = pd.read_csv("products.csv")

# Remove duplicates and rows with no delivery date
orders = orders.drop_duplicates(subset="order_id")
orders = orders.dropna(subset=["delivered_date"])

# Fill missing quantities with the median of that product
orders["quantity"] = orders.groupby("product_id")["quantity"] \
                           .transform(lambda s: s.fillna(s.median()))

# Merge and create delivery features
df = orders.merge(products, on="product_id", how="left")
df["delay_days"] = (df["delivered_date"] - df["promised_date"]).dt.days
df["is_late"]    = (df["delay_days"] > 0).astype(int)


# ======================================================================
# 2. KPI calculation
# ======================================================================
# On-Time Delivery Rate
otd = (1 - df["is_late"].mean()) * 100

# Cost per delivery
cost_per_delivery = df["delivery_cost"].sum() / df["order_id"].nunique()

# Inventory turnover
inventory_turnover = df["cogs"].sum() / df["avg_inventory_value"].mean()

print(f"OTD: {otd:.1f}%  |  Cost/Delivery: Rs {cost_per_delivery:.2f}  |  Turnover: {inventory_turnover:.2f}")


# ======================================================================
# 3. Exploratory data analysis
# ======================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Monthly demand trend
monthly = df.set_index("order_date").resample("M")["quantity"].sum()
monthly.plot(title="Monthly Demand")
plt.show()

# Late-delivery rate by district
late_by_district = df.groupby("district")["is_late"].mean().sort_values(ascending=False)
sns.barplot(x=late_by_district.values[:10], y=late_by_district.index[:10])
plt.title("Top 10 Districts by Late-Delivery Rate")
plt.show()


# ======================================================================
# 4. Demand forecasting (regression)
# ======================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

weekly = (df.groupby(["category", pd.Grouper(key="order_date", freq="W")])["quantity"]
            .sum().reset_index())
weekly = weekly.sort_values(["category", "order_date"])

# Lag and rolling features (per category)
weekly["lag_1"] = weekly.groupby("category")["quantity"].shift(1)
weekly["lag_4"] = weekly.groupby("category")["quantity"].shift(4)
weekly["roll_4"] = weekly.groupby("category")["quantity"].transform(
                       lambda s: s.shift(1).rolling(4).mean())
weekly["week_no"] = weekly["order_date"].dt.isocalendar().week.astype(int)
weekly = weekly.dropna()

features = ["lag_1", "lag_4", "roll_4", "week_no"]

# Time-based split: train on the past, test on the most recent weeks
cutoff = weekly["order_date"].quantile(0.8)
train, test = weekly[weekly.order_date <= cutoff], weekly[weekly.order_date > cutoff]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(train[features], train["quantity"])
pred = model.predict(test[features])

print("MAE :", mean_absolute_error(test["quantity"], pred))
print("MAPE:", mean_absolute_percentage_error(test["quantity"], pred) * 100, "%")


# ======================================================================
# 5. Zone clustering (K-Means)
# ======================================================================
from sklearn.cluster import KMeans

coords = df[["customer_lat", "customer_lng"]].drop_duplicates()
km = KMeans(n_clusters=12, n_init=10, random_state=42)   # one zone per van
coords["zone"] = km.fit_predict(coords)

# Check a sensible number of clusters with the elbow method
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords[["customer_lat","customer_lng"]]).inertia_
           for k in range(2, 20)]


# ======================================================================
# 6. Route optimisation (OR-Tools vehicle routing)
# ======================================================================
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_vrp(dist_matrix, demands, vehicle_capacity, num_vehicles, depot=0):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        return dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]
    dem = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(dem, 0, [vehicle_capacity] * num_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(params)
    return routing, manager, solution   # read routes from 'solution'


# ======================================================================
# 7. Reorder point (inventory rule)
# ======================================================================
# Reorder point = demand during lead time + safety stock
z = 1.65                       # about 95% service level
daily_mean, daily_std = 40, 12 # from historical data (example values)
lead_time_days = 5

safety_stock  = z * daily_std * np.sqrt(lead_time_days)
reorder_point = daily_mean * lead_time_days + safety_stock
print(round(reorder_point))
