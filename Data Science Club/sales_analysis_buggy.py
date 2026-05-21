"""
Data Science Club — Bug Hunt Exercise
======================================
This script loads and analyzes a sales dataset.
It should clean the data, compute summary statistics,
and produce a few key insights.

There are 6 bugs hidden in this file.
Your job: find them, explain what's wrong, and fix them.

Tips:
  - Read carefully — some bugs are logic errors, not syntax errors.
  - Try running the code section by section.
  - Think about what each line is *supposed* to do.
"""

import pandas as pd
import numpy as np

# ── 1. Load & inspect ────────────────────────────────────────────────────────

data = {
    "order_id":   [101, 102, 103, 104, 105, 106, 107, 108],
    "region":     ["North", "South", "East", "West", "North", "East", "South", "West"],
    "product":    ["Widget", "Gadget", "Widget", "Gizmo", "Gizmo", "Widget", "Gadget", "Gizmo"],
    "quantity":   [3, 1, 4, 2, 5, 1, 3, 2],
    "unit_price": [19.99, 45.00, 19.99, 30.00, 30.00, 19.99, 45.00, 30.00],
    "discount":   [0.1, 0.0, 0.2, 0.0, 0.15, 0.0, 0.1, 0.05],
    "returned":   [False, False, True, False, False, False, True, False],
}

df = pd.DataFrame(data)

print("Shape:", df.shape)
print(df.head())


# ── 2. Compute revenue ───────────────────────────────────────────────────────

# Revenue = quantity * unit_price * (1 - discount)
df["revenue"] = df["quantity"] * df["unit_price"] * (1 + df["discount"])


# ── 3. Filter out returned orders ────────────────────────────────────────────

clean_df = df[df["returned"] == False and df["revenue"] > 0]


# ── 4. Region summary ────────────────────────────────────────────────────────

region_summary = clean_df.groupby("region")["revenue"].sum()
print("\nRevenue by region:")
print(region_summary)


# ── 5. Top product by revenue ────────────────────────────────────────────────

product_revenue = clean_df.groupby("product")["revenue"].sum()

top_product = product_revenue.idxmax
print(f"\nTop product: {top_product}")


# ── 6. Add a margin column ───────────────────────────────────────────────────

# Assume a flat 40% cost ratio (cost = 40% of unit_price)
df["cost"] = df["quantity"] * df["unit_price"] * 0.4
clean_df["margin"] = clean_df["revenue"] - df["cost"]


# ── 7. Flag high-value orders ────────────────────────────────────────────────

# Orders are "high value" if revenue is in the top 25% of all orders.
threshold = clean_df["revenue"].quantile(0.75)

clean_df.loc[clean_df["revenue"] < threshold, "high_value"] = True
clean_df.loc[clean_df["revenue"] >= threshold, "high_value"] = False


# ── 8. Monthly summary (if we had a date column) ─────────────────────────────

# Simulate adding a date column
clean_df = clean_df.copy()
clean_df["order_date"] = pd.to_datetime([
    "2024-01-15", "2024-01-22", "2024-02-05",
    "2024-02-18", "2024-03-03", "2024-03-20",
])

clean_df["month"] = clean_df["order_date"].dt.month_name

monthly = clean_df.groupby("month")["revenue"].sum()
print("\nRevenue by month:")
print(monthly)


# ── 9. Export summary ────────────────────────────────────────────────────────

summary = {
    "total_revenue": clean_df["revenue"].sum().round(2),
    "avg_order_value": clean_df["revenue"].mean().round(2),
    "top_product": top_product,
    "top_region": region_summary.idxmax(),
}

print("\n── Final summary ──")
for k, v in summary.items():
    print(f"  {k}: {v}")
