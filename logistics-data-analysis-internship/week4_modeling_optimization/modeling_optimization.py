"""Week 4 - Predictive modeling and optimization for a logistics network.

Run:  python modeling_optimization.py
Output: figs/*.png (6 charts) and console results
"""
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("figs", exist_ok=True)



# ---- Simulate the dataset (same as Week 3) ----
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


# ---- Data preparation and time-based split ----
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = df.sort_values("order_date").reset_index(drop=True)

# Features known at dispatch time (no actual_days, no cost -> no leakage)
num_features = ["distance_km", "weight_kg", "dispatch_delay_hr",
                "scheduled_days", "festival_season", "weekend_order"]
features = num_features + ["van_id"]
target = "delay_days"

# Time-based split: train on the past, test on the most recent 3 months
cutoff = pd.Timestamp("2026-07-01")
train, test = df[df.order_date < cutoff], df[df.order_date >= cutoff]
X_train, y_train = train[features], train[target]
X_test,  y_test  = test[features],  test[target]
print(len(train), len(test))

# One-hot encode the van; numeric columns pass straight through
prep = ColumnTransformer([("van", OneHotEncoder(handle_unknown="ignore"), ["van_id"])],
                         remainder="passthrough")


# ---- Train and cross-validate candidate models ----
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Baseline (mean)":   DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                               random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

tscv = TimeSeriesSplit(n_splits=5)          # each fold trains on the past only
rows = []
for name, m in models.items():
    pipe = Pipeline([("prep", prep), ("model", m)])
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=tscv,
                               scoring="neg_root_mean_squared_error").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({"model": name, "CV_RMSE": cv_rmse,
                 "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                 "R2": r2_score(y_test, pred)})
results = pd.DataFrame(rows).set_index("model").round(3)
print(results)


# ---- Hyperparameter tuning ----
# Hyperparameter tuning: grid search with time-aware cross-validation
cv = TimeSeriesSplit(n_splits=5)
searches = {
  "Random Forest (tuned)": (RandomForestRegressor(random_state=42, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [4, 6, 8, None],
         "model__min_samples_leaf": [5, 20, 50]}),
  "Gradient Boosting (tuned)": (GradientBoostingRegressor(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0]})}

tuned, fitted = {}, {}
for name, (est, grid) in searches.items():
    gs = GridSearchCV(Pipeline([("prep", prep), ("model", est)]), grid, cv=cv,
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_train, y_train)
    p = gs.predict(X_test)
    tuned[name] = {"CV_RMSE": -gs.best_score_, "MAE": mean_absolute_error(y_test, p),
                   "RMSE": np.sqrt(mean_squared_error(y_test, p)), "R2": r2_score(y_test, p),
                   "params": gs.best_params_}
    fitted[name] = gs
    print(name, gs.best_params_, "CV RMSE %.3f" % -gs.best_score_)

# Select the final model on cross-validation error (not on the test set)
best_name = min(tuned, key=lambda k: tuned[k]["CV_RMSE"])
search = fitted[best_name]; best = search.best_estimator_
pred_best = best.predict(X_test)
print("Final model:", best_name, {k: round(v, 3) for k, v in tuned[best_name].items() if k != "params"})


# ---- Evaluation charts ----
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
BLUE, ORANGE, RED, GREEN = "#1F4E79", "#E07B00", "#C0392B", "#2E8B57"

# Figure 1: model comparison on the held-out test set
comp = results[["MAE", "RMSE"]].copy()
for k in tuned: comp.loc[k] = [tuned[k]["MAE"], tuned[k]["RMSE"]]
ax = comp.plot(kind="barh", figsize=(8, 4.6), color=[BLUE, ORANGE])
ax.set_xlabel("Error (days) - lower is better"); ax.set_ylabel("")
ax.set_title("Model Comparison on Test Set"); plt.tight_layout()
plt.savefig("figs/fig1_models.png", dpi=150); plt.close()

# Figure 2: predicted vs actual delay (mean prediction for each actual value)
cmp_df = pd.DataFrame({"actual": y_test.values, "pred": pred_best})
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=cmp_df, x="actual", y="pred", color=BLUE, ax=ax)
ax.set_xlabel("Actual delay (days)"); ax.set_ylabel("Predicted delay (days)")
ax.set_title("Predicted vs Actual Delay (Test Set)"); plt.tight_layout()
plt.savefig("figs/fig2_pred_actual.png", dpi=150); plt.close()

# Figure 3: feature importance of the tuned forest
final = best.named_steps["model"]
names = best.named_steps["prep"].get_feature_names_out()
names = [n.replace("van__van_id_", "van ").replace("remainder__", "") for n in names]
imp = pd.Series(final.feature_importances_, index=names).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(8, 4.2))
imp.plot(kind="barh", color=BLUE, ax=ax)
ax.set_xlabel("Importance"); ax.set_title("Top 10 Feature Importances (Final Delay Model)")
plt.tight_layout(); plt.savefig("figs/fig3_importance.png", dpi=150); plt.close()


# ---- Late-shipment risk classifier ----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

y_tr_c, y_te_c = train["is_late"], test["is_late"]
prep_c = ColumnTransformer([("van", OneHotEncoder(handle_unknown="ignore"), ["van_id"]),
                            ("num", StandardScaler(), num_features)])
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                            class_weight="balanced_subsample",
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)}

# Choose the alert threshold on a validation slice of the TRAINING period
split = int(len(train) * 0.8)
fit_part, val_part = train.iloc[:split], train.iloc[split:]

clf_rows, probs, thresholds = [], {}, {}
for name, m in clf_models.items():
    pc = Pipeline([("prep", prep_c), ("model", m)])
    pc.fit(fit_part[features], fit_part["is_late"])
    pv = pc.predict_proba(val_part[features])[:, 1]
    # highest threshold that still catches at least 70% of late shipments
    thr = max(t for t in np.linspace(0.05, 0.95, 91)
              if recall_score(val_part["is_late"], pv >= t) >= 0.70)
    pc.fit(X_train, y_tr_c)                              # refit on full training period
    p = pc.predict_proba(X_test)[:, 1]
    yhat = (p >= thr).astype(int)
    probs[name], thresholds[name] = (pc, p), thr
    clf_rows.append({"model": name, "ROC_AUC": roc_auc_score(y_te_c, p),
                     "PR_AUC": average_precision_score(y_te_c, p), "threshold": thr,
                     "precision": precision_score(y_te_c, yhat), "recall": recall_score(y_te_c, yhat),
                     "F1": f1_score(y_te_c, yhat)})
clf_results = pd.DataFrame(clf_rows).set_index("model").round(3)
print(clf_results)
print("Late rate in test set: %.3f" % y_te_c.mean())

# Figure 4: ROC curves
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for name, (pc, p) in probs.items():
    fpr, tpr, _ = roc_curve(y_te_c, p); ax.plot(fpr, tpr, lw=2, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC Curve: Late-Delivery Classifier"); ax.legend()
plt.tight_layout(); plt.savefig("figs/fig4_roc.png", dpi=150); plt.close()

# Practical value: if we only inspect the riskiest shipments, how many late ones do we catch?
p_gb = probs["Gradient Boosting"][1]
rank = pd.DataFrame({"p": p_gb, "late": y_te_c.values}).sort_values("p", ascending=False)
capture = {f"top {q}%": rank.head(int(len(rank) * q / 100))["late"].sum() / rank["late"].sum()
           for q in (10, 20, 30)}
print({k: round(v, 3) for k, v in capture.items()})


# ---- Optimization A: risk-aware van allocation (LP) ----
from scipy.optimize import linprog

# Optimization A: risk-aware allocation of shipments to vans (linear programming)
clf, _ = probs["Gradient Boosting"]      # unweighted -> usable probabilities
vans = sorted(df["van_id"].unique())
districts = sorted(df["district"].unique())

# p[v, d] = predicted late probability if van v serves district d
P = np.zeros((len(vans), len(districts)))
for j, d in enumerate(districts):
    sub = df[df["district"] == d][features].copy()
    for i, v in enumerate(vans):
        sub["van_id"] = v
        P[i, j] = clf.predict_proba(sub)[:, 1].mean()

demand = df.groupby("district").size().reindex(districts).values     # shipments per district
total = demand.sum()
cap = 1.15 * total / len(vans)                  # no van carries >15% above an even share
PENALTY = 150                                    # assumed INR cost of one late delivery

# Decision variables x[v, d], flattened row by row; minimise expected late cost
c_vec = (P * PENALTY).flatten()
A_eq = np.zeros((len(districts), P.size)); b_eq = demand.astype(float)
for j in range(len(districts)):
    A_eq[j, j::len(districts)] = 1               # all demand of district j is served
A_ub = np.zeros((len(vans), P.size)); b_ub = np.full(len(vans), cap)
for i in range(len(vans)):
    A_ub[i, i * len(districts):(i + 1) * len(districts)] = 1   # van capacity

res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
X = res.x.reshape(P.shape)

# Compare with today's roughly even split of every district across all vans
x_now = np.tile(demand / len(vans), (len(vans), 1))
late_now, late_opt = (P * x_now).sum(), (P * X).sum()
load_now, load_opt = x_now.sum(axis=1), X.sum(axis=1)
print("Model-implied late rate today: %.1f%% (actual %.1f%%)" % (100 * late_now / total, 100 * df["is_late"].mean()))
print("Expected late shipments: %.0f -> %.0f" % (late_now, late_opt))
print("Expected saving (INR): %.0f" % ((late_now - late_opt) * PENALTY))

# Figure 5: share of volume per van before vs after
fig, ax = plt.subplots(figsize=(8, 4)); idx = np.arange(len(vans))
ax.bar(idx - .2, load_now / total * 100, .4, color="#9AA5B1", label="Current (even split)")
ax.bar(idx + .2, load_opt / total * 100, .4, color=BLUE, label="Optimised")
ax.set_xticks(idx); ax.set_xticklabels(vans); ax.set_ylabel("Share of shipments (%)")
ax.set_ylim(0, 12); ax.set_title("Shipment Allocation by Van: Current vs Risk-Aware"); ax.legend(loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig("figs/fig5_allocation.png", dpi=150); plt.close()


# ---- Optimization B: route sequencing (nearest neighbour + 2-opt) ----
# Optimization B: route sequencing for one delivery run (nearest neighbour + 2-opt)
rng2 = np.random.default_rng(7)
depot = np.array([[0.0, 0.0]])
stops = rng2.normal([100, 60], 14, size=(15, 2))        # 15 stops in one district (km)
pts = np.vstack([depot, stops])
D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) * 1.3   # 1.3 = road-winding factor

def route_len(r): return sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))

def nearest_neighbour(D):
    left, route = set(range(1, len(D))), [0]
    while left:
        nxt = min(left, key=lambda j: D[route[-1], j]); route.append(nxt); left.remove(nxt)
    return route + [0]

def two_opt(route):
    best, improved = route, True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if route_len(cand) < route_len(best) - 1e-9:
                    best, improved = cand, True
    return best

naive = list(range(len(D))) + [0]                       # visit in order received
opt = two_opt(nearest_neighbour(D))
len_naive, len_opt = route_len(naive), route_len(opt)
COST_KM = 18                                            # assumed INR per km operating cost
print("Route: %.0f km -> %.0f km (%.1f%% shorter)" % (len_naive, len_opt, 100 * (1 - len_opt / len_naive)))

# Figure 6: before vs after
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for ax, r, ttl, col in [(axs[0], naive, f"Order received: {len_naive:.0f} km", RED),
                        (axs[1], opt, f"Optimised: {len_opt:.0f} km", GREEN)]:
    ax.plot(pts[r, 0], pts[r, 1], "-o", color=col, ms=4, lw=1.2)
    ax.plot(*pts[0], "k*", ms=14); ax.set_title(ttl); ax.set_xlabel("km east")
axs[0].set_ylabel("km north"); plt.tight_layout()
plt.savefig("figs/fig6_route.png", dpi=150); plt.close()


# ---- Optimization C: festival capacity planning ----
# Optimization C: festival-season capacity planning
weekly = df.set_index("order_date").resample("W")["shipment_id"].count()
normal_week = weekly[weekly.index > "2025-12-14"].mean()      # average non-festival week
festival_week = weekly[(weekly.index >= "2025-10-19") & (weekly.index <= "2025-11-23")].mean()
peak_week = weekly.max()

VANS, UTIL = 12, 0.80                         # fleet size, normal average utilisation
per_van_max = normal_week / VANS / UTIL       # shipments per van-week at 100% utilisation
max_capacity = VANS * per_van_max             # one shift, fully loaded
second_shift = 2 * max_capacity               # a second shift on the same vans
extra_vans = int(np.ceil(max(0, peak_week - max_capacity) / per_van_max))

print("Normal %.0f | festival avg %.0f | peak %.0f shipments/week" % (normal_week, festival_week, peak_week))
print("Capacity: one shift %.0f | two shifts %.0f | extra vans for single shift: %d" % (max_capacity, second_shift, extra_vans))
