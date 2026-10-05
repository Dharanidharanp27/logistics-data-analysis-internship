"""Simulates a raw DataCo-style shipping dataset with injected data-quality problems."""
import numpy as np, pandas as pd
def make_data(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    modes = np.array(["Standard Class","Second Class","First Class","Same Day"])
    sched_map = {"Standard Class":4,"Second Class":2,"First Class":1,"Same Day":0}
    mode = rng.choice(modes, n, p=[.6,.2,.15,.05])
    sched = np.array([sched_map[m] for m in mode])
    real = np.clip(sched + rng.integers(-1,4,n), 0, None)
    odate = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0,1000,n), unit="D")
    sales = np.round(rng.lognormal(5,0.6,n),2)
    sales[rng.choice(n,int(.01*n),replace=False)] *= 20
    qty = rng.integers(1,6,n).astype(float)
    profit = np.round(sales*rng.uniform(-.2,.4,n),2)
    df = pd.DataFrame({
      "Order Id": np.arange(1000,1000+n),
      "order date (DateOrders)": odate, "shipping date (DateOrders)": odate + pd.to_timedelta(real, unit="D"),
      "Shipping Mode": mode, "Order Region": rng.choice(["South Asia","West Europe","Central America","Oceania","US Center","Eastern Asia"], n),
      "Customer Zipcode": rng.integers(10000,99999,n).astype(float),
      "Order Item Quantity": qty, "Sales": sales, "Order Profit Per Order": profit,
      "Days for shipping (real)": real.astype(float), "Days for shipment (scheduled)": sched.astype(float),
      "Delivery Status": np.where(real>sched,"Late delivery","Shipping on time"), "Product Description": np.nan})
    for c in ["order date (DateOrders)","shipping date (DateOrders)"]:
        df[c] = df[c].dt.strftime("%m/%d/%Y %H:%M")
    idx = lambda f: rng.choice(n,int(f*n),replace=False)
    df.loc[idx(.03),"Sales"]=np.nan; df.loc[idx(.02),"Customer Zipcode"]=np.nan
    df.loc[idx(.04),"Order Region"]=np.nan; df.loc[idx(.015),"Days for shipping (real)"]=np.nan
    df.loc[idx(.01),"Delivery Status"]=np.nan
    df.loc[idx(.005),"Order Item Quantity"]=-1; df.loc[idx(.003),"Sales"]=-50
    df.loc[idx(.01),"order date (DateOrders)"]="N/A"
    b = idx(.02); df.loc[b,"shipping date (DateOrders)"] = (pd.to_datetime(df.loc[b,"order date (DateOrders)"],errors="coerce")-pd.Timedelta(days=2)).dt.strftime("%m/%d/%Y %H:%M")
    v = idx(.08); df.loc[v,"Shipping Mode"] = [m.lower()+" " if i%2 else m.upper() for i,m in enumerate(df.loc[v,"Shipping Mode"])]
    df = pd.concat([df, df.sample(int(.015*n), random_state=1)], ignore_index=True)
    return df
