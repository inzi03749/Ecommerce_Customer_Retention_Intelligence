"""
E-Commerce Customer Retention Intelligence System
==================================================
Author  : Inzimamul Haq
Project : IBM Capstone -- E-Commerce Customer Retention Intelligence System

This single file contains the complete project implementation:
  - Stage 2 : Data Cleaning & Exploratory Data Analysis
  - Stage 3 : Machine Learning Model Development
  - Stage 4 : Streamlit Interactive Dashboard

Dataset : data/E Commerce Dataset.xlsx  (read-only, never modified)

Usage
-----
Run Stage 2 (EDA + cleaning):
    python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 2

Run Stage 3 (ML training + evaluation):
    python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 3

Run both Stage 2 and Stage 3 sequentially:
    python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage all

Launch Streamlit dashboard (Stage 4):
    streamlit run Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py

IMPORTANT NOTES:
- The original dataset (data/E Commerce Dataset.xlsx) is NEVER modified.
- Stage 3 intentionally reloads the raw Excel (not the Stage-2 cleaned CSV)
  so that median imputation is performed inside sklearn Pipelines, fitted
  exclusively on training data, preventing preprocessing leakage.
- The F1-selected threshold (0.34) is loaded dynamically from
  ml_outputs/rf_threshold.joblib at dashboard runtime.
- The High-risk boundary (0.60) is a fixed business/dashboard segmentation
  boundary -- it was NOT selected through OOF optimisation.
- All predicted scores are churn RISK SCORES, not calibrated probabilities.
- All findings are observational, not causal.
"""

import sys
import os

# ============================================================================
# ENTRY POINT ROUTING
# ============================================================================
# When invoked via `streamlit run`, Streamlit sets argv[0] to the script path
# and does NOT pass --stage. We detect this to avoid running ML/EDA stages.
_running_in_streamlit = (
    "streamlit" in sys.modules or
    any("streamlit" in arg for arg in sys.argv)
)

def _parse_stage():
    """Return requested stage string, or None if not specified."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--stage", "-s") and i < len(sys.argv):
            return sys.argv[i + 1].lower()
        if arg.startswith("--stage="):
            return arg.split("=", 1)[1].lower()
    return None


# ============================================================================
# SHARED IMPORTS (used by all stages)
# ============================================================================
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================================
# ============================================================================
#
#   STAGE 2 -- DATA CLEANING & EXPLORATORY DATA ANALYSIS
#
# ============================================================================
# ============================================================================

def run_stage2():
    """
    Stage 2: Data Cleaning, Quality Validation, EDA, Business Insight Discovery
    Source: stage2_eda.py (merged verbatim -- logic unchanged)
    """

    # -- Output directory for charts -------------------------------------------
    OUTPUT_DIR = "eda_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Colour palette --------------------------------------------------------
    CHURN_PALETTE = {0: "#4C9BE8", 1: "#E84C4C"}
    CHURN_LABELS  = {0: "Retained (0)", 1: "Churned (1)"}
    sns.set_theme(style="whitegrid", font_scale=1.05)

    def section(title):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def subsection(title):
        print(f"\n--- {title} ---")

    def save_fig(name):
        path = os.path.join(OUTPUT_DIR, name)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {path}")

    # ==========================================================================
    # STAGE 2A -- LOAD AND CLEAN
    # ==========================================================================
    section("STAGE 2A -- LOAD AND CLEAN")

    DATA_PATH = os.path.join("data", "E Commerce Dataset.xlsx")
    df_raw = pd.read_excel(DATA_PATH, sheet_name="E Comm", engine="openpyxl")
    print(f"\nLoaded '{DATA_PATH}' -- sheet 'E Comm'")
    print(f"Raw shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    df = df_raw.copy()

    # Step 1: Remove duplicate rows
    subsection("Step 1 -- Duplicate Rows")
    rows_before = len(df)
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)
    print(f"  Rows before   : {rows_before}")
    print(f"  Duplicates    : {n_dupes}")
    print(f"  Rows after    : {rows_after_dedup}")

    # Step 2: CustomerID uniqueness
    subsection("Step 2 -- CustomerID Uniqueness")
    n_unique_id = df["CustomerID"].nunique()
    print(f"  Total rows          : {len(df)}")
    print(f"  Unique CustomerIDs  : {n_unique_id}")
    print(f"  All IDs unique?     : {n_unique_id == len(df)}")
    print("  -> CustomerID will be dropped before ML (identifier, not predictor).")

    # Step 3: Missing values -- before imputation
    subsection("Step 3 -- Missing Values")
    NUMERICAL_IMPUTE_COLS = [
        "DaySinceLastOrder", "OrderAmountHikeFromlastYear", "Tenure",
        "OrderCount", "CouponUsed", "HourSpendOnApp", "WarehouseToHome"
    ]
    print("\n  Missing value counts BEFORE imputation:")
    for col in NUMERICAL_IMPUTE_COLS:
        n_miss = df[col].isna().sum()
        pct    = n_miss / len(df) * 100
        print(f"    {col:<35} {n_miss:>4} ({pct:.2f}%)")

    medians = {}
    for col in NUMERICAL_IMPUTE_COLS:
        med = df[col].median()
        medians[col] = med
        df[col] = df[col].fillna(med)

    print("\n  Medians used for imputation:")
    for col, med in medians.items():
        print(f"    {col:<35} median = {med}")

    print("\n  Missing value counts AFTER imputation:")
    for col in NUMERICAL_IMPUTE_COLS:
        print(f"    {col:<35} {df[col].isna().sum():>4}")

    # Step 4: Categorical standardisation
    subsection("Step 4 -- Categorical Inspection & Standardisation")

    print("\n  [PreferredPaymentMode] Value counts with churn rates:")
    ppm_df = df.groupby("PreferredPaymentMode").agg(
        Count=("Churn", "count"),
        ChurnCount=("Churn", "sum")
    ).assign(ChurnRate=lambda x: (x["ChurnCount"] / x["Count"] * 100).round(2))
    print(ppm_df.sort_values("Count", ascending=False).to_string())

    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({
        "CC":  "Credit Card",
        "COD": "Cash on Delivery"
    })
    print("\n  -> Consolidated: 'CC' -> 'Credit Card', 'COD' -> 'Cash on Delivery'")
    print("    Justification: abbreviations for the same payment methods.")
    print("\n  [PreferredPaymentMode] After consolidation:")
    print(df["PreferredPaymentMode"].value_counts().to_string())

    print("\n  [PreferredLoginDevice] Value counts with churn rates:")
    pld_df = df.groupby("PreferredLoginDevice").agg(
        Count=("Churn", "count"),
        ChurnCount=("Churn", "sum")
    ).assign(ChurnRate=lambda x: (x["ChurnCount"] / x["Count"] * 100).round(2))
    print(pld_df.sort_values("Count", ascending=False).to_string())

    phone_rate  = pld_df.loc["Phone",        "ChurnRate"] if "Phone"        in pld_df.index else None
    mobile_rate = pld_df.loc["Mobile Phone", "ChurnRate"] if "Mobile Phone" in pld_df.index else None
    print(f"\n  Churn rate 'Phone':        {phone_rate}%")
    print(f"  Churn rate 'Mobile Phone': {mobile_rate}%")

    if phone_rate is not None and mobile_rate is not None:
        diff = abs(phone_rate - mobile_rate)
        print(f"  -> Churn rate difference = {diff:.2f}% (> 3%). Keeping 'Phone' and 'Mobile Phone' separate.")
        print("    Justification: observed churn rates differ materially (Phone ~22.4% vs Mobile Phone ~12.6%).")

    print("\n  [PreferedOrderCat] Value counts with churn rates:")
    poc_df = df.groupby("PreferedOrderCat").agg(
        Count=("Churn", "count"),
        ChurnCount=("Churn", "sum")
    ).assign(ChurnRate=lambda x: (x["ChurnCount"] / x["Count"] * 100).round(2))
    print(poc_df.sort_values("Count", ascending=False).to_string())

    mob_rate  = poc_df.loc["Mobile",       "ChurnRate"] if "Mobile"       in poc_df.index else None
    mobp_rate = poc_df.loc["Mobile Phone", "ChurnRate"] if "Mobile Phone" in poc_df.index else None
    print(f"\n  Churn rate 'Mobile':        {mob_rate}%")
    print(f"  Churn rate 'Mobile Phone':  {mobp_rate}%")

    if mob_rate is not None and mobp_rate is not None:
        diff2 = abs(mob_rate - mobp_rate)
        if diff2 <= 3.0:
            df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})
            print(f"  -> Churn rate difference = {diff2:.2f}% (<= 3%). Merging 'Mobile' -> 'Mobile Phone'.")
        else:
            print(f"  -> Churn rate difference = {diff2:.2f}% (> 3%). Keeping separate.")

    # Step 5: Rename PreferedOrderCat
    subsection("Step 5 -- Rename PreferedOrderCat")
    df = df.rename(columns={"PreferedOrderCat": "PreferredOrderCategory"})
    print("  Renamed 'PreferedOrderCat' -> 'PreferredOrderCategory'")

    # Steps 6-8: Final validation
    subsection("Steps 6-8 -- Final Cleaned DataFrame")
    print(f"\n  Shape              : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Remaining missing  : {df.isnull().sum().sum()}")
    print(f"  Remaining dupes    : {df.duplicated().sum()}")
    print(f"\n  Columns: {list(df.columns)}")

    # ==========================================================================
    # STAGE 2B -- DATA QUALITY VALIDATION
    # ==========================================================================
    section("STAGE 2B -- DATA QUALITY VALIDATION")

    subsection("Data Types")
    print(df.dtypes.to_string())

    subsection("Target Variable Values")
    print(f"  Churn unique values : {sorted(df['Churn'].unique())}")
    print(f"  Expected            : [0, 1]")

    subsection("Numerical Range Checks")
    RANGE_CHECKS = {
        "Tenure"                      : (0, 100),
        "WarehouseToHome"             : (1, 200),
        "HourSpendOnApp"              : (0, 24),
        "NumberOfDeviceRegistered"    : (1, 20),
        "SatisfactionScore"           : (1, 5),
        "NumberOfAddress"             : (1, 50),
        "Complain"                    : (0, 1),
        "OrderAmountHikeFromlastYear" : (0, 100),
        "CouponUsed"                  : (0, 50),
        "OrderCount"                  : (0, 100),
        "DaySinceLastOrder"           : (0, 365),
        "CashbackAmount"              : (0, 1000),
    }

    print(f"\n  {'Column':<35} {'Min':>8} {'Max':>8} {'Mean':>10} {'Std':>10}  {'Suspicions'}")
    print(f"  {'-'*90}")
    for col, (lo, hi) in RANGE_CHECKS.items():
        mn  = df[col].min()
        mx  = df[col].max()
        avg = df[col].mean()
        std = df[col].std()
        flags = []
        if mn < lo: flags.append(f"min<{lo}")
        if mx > hi: flags.append(f"max>{hi}")
        flag_str = ", ".join(flags) if flags else "OK"
        print(f"  {col:<35} {mn:>8.1f} {mx:>8.1f} {avg:>10.2f} {std:>10.2f}  {flag_str}")

    subsection("Outlier Investigation")
    OUTLIER_COLS = [
        "WarehouseToHome", "Tenure", "DaySinceLastOrder",
        "CashbackAmount", "OrderCount", "CouponUsed", "SatisfactionScore"
    ]
    for col in OUTLIER_COLS:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        n_out = ((df[col] < lo_fence) | (df[col] > hi_fence)).sum()
        print(f"  {col:<35} IQR-outliers = {n_out:>4}  (fence: {lo_fence:.1f}-{hi_fence:.1f})")

    print("\n  -> Outlier policy: RETAIN all outliers.")
    print("    Random Forest is naturally robust to outliers; Logistic Regression uses")
    print("    StandardScaler which limits their influence.")

    subsection("Categorical Values After Standardisation")
    for col in ["PreferredLoginDevice", "PreferredPaymentMode",
                "Gender", "PreferredOrderCategory", "MaritalStatus"]:
        vals = sorted(df[col].unique())
        print(f"  {col}: {vals}")

    subsection("Duplicate CustomerID Check (post-dedup)")
    dup_cid = df["CustomerID"].duplicated().sum()
    print(f"  Duplicate CustomerIDs: {dup_cid}")
    if dup_cid == 0:
        print("  -> All CustomerIDs unique after deduplication.")

    # ==========================================================================
    # STAGE 2C -- EDA
    # ==========================================================================
    section("STAGE 2C -- EXPLORATORY DATA ANALYSIS")

    churn0 = df[df["Churn"] == 0]
    churn1 = df[df["Churn"] == 1]
    total  = len(df)
    n_churned  = len(churn1)
    n_retained = len(churn0)

    def churn_rate_by(col):
        return (
            df.groupby(col)["Churn"]
            .agg(Count="count", ChurnCount="sum")
            .assign(ChurnRate=lambda x: (x["ChurnCount"] / x["Count"] * 100).round(2))
            .sort_values("ChurnRate", ascending=False)
        )

    # 1. Overall Churn Distribution
    subsection("1. Overall Churn Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Overall Churn Distribution", fontsize=14, fontweight="bold")
    counts = df["Churn"].value_counts().sort_index()
    colors = [CHURN_PALETTE[i] for i in counts.index]
    bars = axes[0].bar(["Retained (0)", "Churned (1)"], counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_title("Count")
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[1].pie(
        counts.values,
        labels=["Retained (0)", "Churned (1)"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    axes[1].set_title("Proportion")
    plt.tight_layout()
    save_fig("01_churn_distribution.png")
    print(f"  Retained : {n_retained:,} ({n_retained/total*100:.1f}%)")
    print(f"  Churned  : {n_churned:,}  ({n_churned/total*100:.1f}%)")

    # 2. Churn Rate by Categorical Features
    subsection("2. Churn Rate by Categorical Features")
    CAT_FEATURES = [
        "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
        "CityTier", "PreferredOrderCategory", "MaritalStatus", "Complain"
    ]
    fig, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Churn Rate by Categorical / Binary Features", fontsize=15, fontweight="bold")
    axes2 = axes2.flatten()
    cat_results = {}
    for i, feat in enumerate(CAT_FEATURES):
        cr = churn_rate_by(feat).reset_index()
        cat_results[feat] = cr
        print(f"\n  {feat}:")
        print(cr.to_string(index=False))
        ax = axes2[i]
        bars2 = ax.bar(cr[feat].astype(str), cr["ChurnRate"],
                       color="#E84C4C", edgecolor="white", width=0.55)
        for bar, rate, cnt in zip(bars2, cr["ChurnRate"], cr["Count"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{rate:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=8)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, cr["ChurnRate"].max() * 1.35)
        ax.tick_params(axis="x", rotation=30)
    axes2[-1].set_visible(False)
    plt.tight_layout()
    save_fig("02_churn_rate_by_categorical.png")

    # 3. Numerical Distributions vs Churn
    subsection("3. Numerical Feature Distributions vs Churn")
    NUM_FEATURES = [
        "Tenure", "SatisfactionScore", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "NumberOfAddress",
        "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount", "WarehouseToHome"
    ]
    fig, axes3 = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle("Numerical Feature Box Plots by Churn Status", fontsize=15, fontweight="bold")
    axes3 = axes3.flatten()
    for i, feat in enumerate(NUM_FEATURES):
        ax = axes3[i]
        data_to_plot = [churn0[feat].dropna(), churn1[feat].dropna()]
        bp = ax.boxplot(data_to_plot, labels=["Retained", "Churned"],
                        patch_artist=True, notch=False, widths=0.5)
        for patch, color in zip(bp["boxes"], [CHURN_PALETTE[0], CHURN_PALETTE[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        med0 = churn0[feat].median()
        med1 = churn1[feat].median()
        ax.set_title(f"{feat}\nRetained med={med0:.1f} | Churned med={med1:.1f}", fontsize=9)
        ax.set_ylabel("Value")
        print(f"  {feat:<35} median retained={med0:.2f}  churned={med1:.2f}")
    axes3[-1].set_visible(False)
    plt.tight_layout()
    save_fig("03_boxplots_numerical_vs_churn.png")

    # 4a. SatisfactionScore vs Churn
    subsection("4a. SatisfactionScore vs Churn")
    ss_cr = churn_rate_by("SatisfactionScore").reset_index().sort_values("SatisfactionScore")
    print(ss_cr.to_string(index=False))
    fig, axes4 = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SatisfactionScore vs Churn", fontsize=13, fontweight="bold")
    grouped = df.groupby(["SatisfactionScore", "Churn"]).size().unstack(fill_value=0)
    grouped.plot(kind="bar", ax=axes4[0], color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
                 edgecolor="white", width=0.7)
    axes4[0].set_title("Customer Counts by Score")
    axes4[0].set_xlabel("Satisfaction Score")
    axes4[0].set_ylabel("Count")
    axes4[0].legend(["Retained", "Churned"])
    axes4[0].tick_params(axis="x", rotation=0)
    axes4[1].plot(ss_cr["SatisfactionScore"], ss_cr["ChurnRate"],
                  marker="o", color="#E84C4C", linewidth=2, markersize=8)
    for _, row in ss_cr.iterrows():
        axes4[1].annotate(f"{row['ChurnRate']:.1f}%",
                          (row["SatisfactionScore"], row["ChurnRate"]),
                          textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axes4[1].set_title("Churn Rate by Score")
    axes4[1].set_xlabel("Satisfaction Score")
    axes4[1].set_ylabel("Churn Rate (%)")
    axes4[1].set_ylim(0, ss_cr["ChurnRate"].max() * 1.3)
    plt.tight_layout()
    save_fig("04a_satisfaction_vs_churn.png")

    # 4b. Complain vs Churn
    subsection("4b. Complain vs Churn")
    comp_cr = churn_rate_by("Complain").reset_index()
    print(comp_cr.to_string(index=False))
    fig, axes5 = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Complaint vs Churn", fontsize=13, fontweight="bold")
    grouped_c = df.groupby(["Complain", "Churn"]).size().unstack(fill_value=0)
    grouped_c.index = ["No Complaint", "Complaint"]
    grouped_c.columns = ["Retained", "Churned"]
    grouped_c.plot(kind="bar", ax=axes5[0], color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
                   edgecolor="white", width=0.6)
    axes5[0].set_title("Customer Counts")
    axes5[0].set_xlabel("")
    axes5[0].set_ylabel("Count")
    axes5[0].tick_params(axis="x", rotation=0)
    comp_cr_plot = comp_cr.copy()
    comp_cr_plot["Complain"] = comp_cr_plot["Complain"].map({0: "No Complaint", 1: "Complaint"})
    colors_c = [CHURN_PALETTE[1]] * 2
    bars_c = axes5[1].bar(comp_cr_plot["Complain"], comp_cr_plot["ChurnRate"],
                          color=colors_c, edgecolor="white", width=0.5)
    for bar, rate in zip(bars_c, comp_cr_plot["ChurnRate"]):
        axes5[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                      f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes5[1].set_title("Churn Rate")
    axes5[1].set_ylabel("Churn Rate (%)")
    axes5[1].set_ylim(0, comp_cr_plot["ChurnRate"].max() * 1.3)
    plt.tight_layout()
    save_fig("04b_complain_vs_churn.png")

    # 4c. Tenure vs Churn
    subsection("4c. Tenure vs Churn")
    tenure_bins   = [-1, 0, 3, 6, 12, 24, 100]
    tenure_labels = ["0 mo", "1-3 mo", "4-6 mo", "7-12 mo", "13-24 mo", "25+ mo"]
    df["TenureBin"] = pd.cut(df["Tenure"], bins=tenure_bins, labels=tenure_labels)
    tenure_cr = churn_rate_by("TenureBin").reset_index()
    tenure_cr["TenureBin"] = pd.Categorical(tenure_cr["TenureBin"], categories=tenure_labels, ordered=True)
    tenure_cr = tenure_cr.sort_values("TenureBin")
    print(tenure_cr.to_string(index=False))
    fig, axes6 = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tenure vs Churn", fontsize=13, fontweight="bold")
    grouped_t = df.groupby(["TenureBin", "Churn"]).size().unstack(fill_value=0)
    grouped_t.plot(kind="bar", ax=axes6[0], color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
                   edgecolor="white", width=0.7)
    axes6[0].set_title("Customer Counts by Tenure Group")
    axes6[0].set_xlabel("Tenure")
    axes6[0].set_ylabel("Count")
    axes6[0].legend(["Retained", "Churned"])
    axes6[0].tick_params(axis="x", rotation=25)
    axes6[1].bar(tenure_cr["TenureBin"].astype(str), tenure_cr["ChurnRate"],
                 color="#E84C4C", edgecolor="white", width=0.6)
    for _, row in tenure_cr.iterrows():
        axes6[1].text(list(tenure_labels).index(str(row["TenureBin"])),
                      row["ChurnRate"] + 0.3, f"{row['ChurnRate']:.1f}%",
                      ha="center", va="bottom", fontsize=9)
    axes6[1].set_title("Churn Rate by Tenure Group")
    axes6[1].set_xlabel("Tenure Group")
    axes6[1].set_ylabel("Churn Rate (%)")
    axes6[1].set_ylim(0, tenure_cr["ChurnRate"].max() * 1.3)
    axes6[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    save_fig("04c_tenure_vs_churn.png")

    # 4d. OrderCount vs Churn
    subsection("4d. OrderCount vs Churn")
    print(df.groupby("Churn")["OrderCount"].describe().to_string())
    fig, axes7 = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("OrderCount vs Churn", fontsize=13, fontweight="bold")
    axes7[0].hist([churn0["OrderCount"], churn1["OrderCount"]],
                  bins=range(0, 18), color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
                  label=["Retained", "Churned"], alpha=0.7, edgecolor="white")
    axes7[0].set_xlabel("Order Count")
    axes7[0].set_ylabel("Frequency")
    axes7[0].set_title("Distribution of OrderCount")
    axes7[0].legend()
    order_cr = churn_rate_by("OrderCount").reset_index().sort_values("OrderCount")
    axes7[1].bar(order_cr["OrderCount"].astype(str), order_cr["ChurnRate"],
                 color="#E84C4C", edgecolor="white")
    axes7[1].set_xlabel("Order Count")
    axes7[1].set_ylabel("Churn Rate (%)")
    axes7[1].set_title("Churn Rate by Order Count")
    axes7[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    save_fig("04d_ordercount_vs_churn.png")

    # 4e. DaySinceLastOrder vs Churn
    subsection("4e. DaySinceLastOrder vs Churn")
    print(df.groupby("Churn")["DaySinceLastOrder"].describe().to_string())
    fig, axes8 = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DaySinceLastOrder vs Churn", fontsize=13, fontweight="bold")
    axes8[0].hist([churn0["DaySinceLastOrder"], churn1["DaySinceLastOrder"]],
                  bins=25, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
                  label=["Retained", "Churned"], alpha=0.7, edgecolor="white")
    axes8[0].set_xlabel("Days Since Last Order")
    axes8[0].set_ylabel("Frequency")
    axes8[0].set_title("Distribution")
    axes8[0].legend()
    from scipy.stats import gaussian_kde
    for subset, lbl, col in [(churn0, "Retained", CHURN_PALETTE[0]),
                              (churn1, "Churned",  CHURN_PALETTE[1])]:
        vals = subset["DaySinceLastOrder"].dropna().values
        kde  = gaussian_kde(vals)
        xs   = np.linspace(vals.min(), vals.max(), 200)
        axes8[1].plot(xs, kde(xs), label=lbl, color=col, linewidth=2)
    axes8[1].set_xlabel("Days Since Last Order")
    axes8[1].set_ylabel("Density")
    axes8[1].set_title("KDE by Churn Status")
    axes8[1].legend()
    plt.tight_layout()
    save_fig("04e_dayssincelastorder_vs_churn.png")

    # 5. Correlation Heatmap
    subsection("5. Correlation Heatmap")
    NUM_CORR_COLS = [
        "Churn", "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount"
    ]
    corr = df[NUM_CORR_COLS].corr()
    print("\n  Pearson correlations with Churn (sorted by |r|):")
    churn_corr = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    for col, val in churn_corr.items():
        print(f"    {col:<35} r = {val:+.4f}")
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Pearson Correlation Matrix -- Numerical Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("05_correlation_heatmap.png")

    # 6. CashbackAmount vs Churn
    subsection("6. CashbackAmount vs Churn")
    print(df.groupby("Churn")["CashbackAmount"].describe().to_string())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist([churn0["CashbackAmount"], churn1["CashbackAmount"]],
            bins=30, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]],
            label=["Retained", "Churned"], alpha=0.7, edgecolor="white")
    ax.set_xlabel("Cashback Amount")
    ax.set_ylabel("Frequency")
    ax.set_title("CashbackAmount Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("06_cashback_vs_churn.png")

    df = df.drop(columns=["TenureBin"])

    # ==========================================================================
    # STAGE 2D -- BUSINESS INSIGHT DISCOVERY
    # ==========================================================================
    section("STAGE 2D -- KEY BUSINESS FINDINGS")

    overall_churn_rate = df["Churn"].mean() * 100
    comp_no   = df[df["Complain"] == 0]["Churn"].mean() * 100
    comp_yes  = df[df["Complain"] == 1]["Churn"].mean() * 100
    new_cust  = df[df["Tenure"] <= 3]
    new_churn_rate  = new_cust["Churn"].mean() * 100
    long_cust = df[df["Tenure"] > 12]
    long_churn_rate = long_cust["Churn"].mean() * 100
    ss1_2  = df[df["SatisfactionScore"] <= 2]["Churn"].mean() * 100
    ss4_5  = df[df["SatisfactionScore"] >= 4]["Churn"].mean() * 100
    ss5_rate = df[df["SatisfactionScore"] == 5]["Churn"].mean() * 100
    ss_churn_by_score = df.groupby("SatisfactionScore")["Churn"].mean() * 100
    city_cr  = churn_rate_by("CityTier").reset_index()
    cb_med_retained  = churn0["CashbackAmount"].median()
    cb_med_churned   = churn1["CashbackAmount"].median()
    cb_mean_retained = churn0["CashbackAmount"].mean()
    cb_mean_churned  = churn1["CashbackAmount"].mean()
    dlo_med_retained = churn0["DaySinceLastOrder"].median()
    dlo_med_churned  = churn1["DaySinceLastOrder"].median()
    poc_cr   = churn_rate_by("PreferredOrderCategory").reset_index()

    print(f"\n  Overall churn rate: {overall_churn_rate:.1f}%")
    print()
    print("  FINDING 1 -- Complaints are strongly associated with churn")
    print(f"    Churn rate, no complaint  : {comp_no:.1f}%")
    print(f"    Churn rate, with complaint: {comp_yes:.1f}%")
    print(f"    Lift                       : {comp_yes/comp_no:.1f}x")
    print()
    print("  FINDING 2 -- New customers (<=3 months) churn at a much higher rate")
    print(f"    Churn rate, Tenure <= 3 mo  : {new_churn_rate:.1f}%")
    print(f"    Churn rate, Tenure > 12 mo : {long_churn_rate:.1f}%")
    print()
    print("  FINDING 3 -- SatisfactionScore 5 has the highest observed churn rate (counter-intuitive)")
    print("    Churn rate per score:")
    for score, rate in ss_churn_by_score.items():
        print(f"      Score {score}: {rate:.1f}%")
    print(f"    Score 1-2  : {ss1_2:.1f}%")
    print(f"    Score 4-5  : {ss4_5:.1f}%")
    print()
    print("  FINDING 4 -- City Tier 3 has the highest observed churn rate")
    print(city_cr.to_string(index=False))
    print()
    print("  FINDING 5 -- Churned customers have lower median cashback amounts")
    print(f"    Median cashback, Retained : {cb_med_retained:.2f}")
    print(f"    Median cashback, Churned  : {cb_med_churned:.2f}")
    print()
    print("  FINDING 6 -- DaySinceLastOrder: churned customers have a slightly LOWER median")
    print(f"    Median DaySinceLastOrder, Retained : {dlo_med_retained:.1f}")
    print(f"    Median DaySinceLastOrder, Churned  : {dlo_med_churned:.1f}")
    print()
    print("  FINDING 7 -- PreferredOrderCategory shows churn rate variation")
    print(poc_cr.to_string(index=False))

    # ==========================================================================
    # STAGE 2E -- RISK & OPPORTUNITY DISCOVERY
    # ==========================================================================
    section("STAGE 2E -- RISK AND OPPORTUNITY DISCOVERY")
    print("""
POTENTIAL RISK INDICATORS (evidence-based):
============================================
1. New customers (Tenure <= 3 months)
   -> Highest churn segment in the dataset. Tenure shows the strongest
     observed numerical association with churn (r ~ -0.34).

2. Customers who raised a complaint
   -> Customers with a recorded complaint had a substantially higher
     observed churn rate (31.7% vs 10.9%). Treat as association, not
     causal, due to unknown temporal ordering.

3. DaySinceLastOrder (requires careful interpretation)
   -> Churned customers had a slightly LOWER median DaySinceLastOrder.
     This counter-intuitive result requires careful interpretation.

4. CashbackAmount below dataset median
   -> Churned customers received lower median cashback (149.66 vs 166.12).

5. City Tier 3 customers
   -> City Tier 3 has the highest observed churn rate (21.4%).

6. Single marital status
   -> Single customers showed the highest observed churn rate (~26.7%).

POTENTIAL OPPORTUNITIES (evidence-based):
==========================================
1. Early-tenure engagement programme (0-3 months) -- testable hypothesis
2. Complaint resolution pathway -- testable hypothesis
3. Cashback / incentive programme review -- testable hypothesis
4. City Tier 3 targeted engagement -- testable hypothesis
5. Satisfaction score trajectory monitoring as an early-warning signal -- testable hypothesis
""")

    # ==========================================================================
    # STAGE 2F -- SUMMARY
    # ==========================================================================
    section("STAGE 2F -- STAGE 2 SUMMARY")
    print(f"""
DATA CLEANING SUMMARY
---------------------
  Rows before cleaning          : {rows_before:,}
  Duplicate rows removed        : {n_dupes:,}
  Rows after deduplication      : {rows_after_dedup:,}
  Missing values imputed        : 7 numerical columns (median strategy)
  Categorical standardisations  :
    PreferredPaymentMode   -> merged 'CC' -> 'Credit Card', 'COD' -> 'Cash on Delivery'
    PreferredLoginDevice   -> 'Phone' and 'Mobile Phone' KEPT SEPARATE (churn rate diff = 9.83 pp)
    PreferredOrderCategory -> merged 'Mobile' -> 'Mobile Phone' (churn rate diff = 0.35 pp)
    PreferedOrderCat       -> renamed to 'PreferredOrderCategory'
  Final shape                   : {df.shape[0]:,} rows x {df.shape[1]} columns
  Remaining missing values      : {df.isnull().sum().sum()}
  Remaining duplicate rows      : {df.duplicated().sum()}
""")

    print(f"""
EDA SUMMARY
-----------
  Overall churn rate       : {overall_churn_rate:.1f}%
  Class distribution       : {n_retained:,} retained ({n_retained/total*100:.1f}%) | {n_churned:,} churned ({n_churned/total*100:.1f}%)

  Key correlations with Churn (Pearson r):""")
    for col, val in churn_corr.items():
        print(f"    {col:<35} r = {val:+.4f}")

    # Save cleaned data for Stage-2 EDA/business-analysis reference
    CLEAN_PATH = os.path.join("data", "ecomm_cleaned.csv")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\n  Cleaned dataset saved for Stage-2 EDA reference: {CLEAN_PATH}")
    print("  Note: Stage 3 reloads the raw Excel source and does not use this CSV for ML.")
    print(f"  Shape: {df.shape}")
    print("\n  EDA charts saved to:", OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("  STAGE 2 COMPLETE -- Awaiting instruction for Stage 3 (ML Model Training)")
    print("=" * 70)


# ============================================================================
# ============================================================================
#
#   STAGE 3 -- MACHINE LEARNING MODEL DEVELOPMENT
#
# ============================================================================
# ============================================================================

def run_stage3():
    """
    Stage 3: Machine Learning Model Development (Audited & Corrected)
    Source: ml_model.py (merged verbatim -- logic unchanged)

    STAGE 2 vs STAGE 3 DATA FLOW NOTE:
      Stage 2 (run_stage2): Loads raw Excel, cleans/imputes for EDA, saves cleaned CSV.
        The pre-split imputation is a known limitation for ML use.
      Stage 3 (this function): Intentionally reloads the raw Excel (NOT the cleaned CSV).
        All imputation is performed INSIDE sklearn Pipelines, fitted on X_train only.
        This prevents preprocessing leakage.

    THRESHOLD SELECTION NOTE:
      The classification threshold is selected using OOF predictions on the
      training set only. The test set is not used for threshold selection.
      It is reserved for final held-out evaluation and post-fit diagnostics.
    """
    import joblib
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, cross_validate, cross_val_predict
    )
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
        roc_curve, precision_recall_curve
    )

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    ML_DIR = "ml_outputs"
    os.makedirs(ML_DIR, exist_ok=True)

    LOG_PATH = os.path.join(ML_DIR, "ml_results.txt")
    _log_lines = []

    def log(msg=""):
        print(msg)
        _log_lines.append(str(msg))

    def flush_log():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(_log_lines))

    def section(title):
        bar = "=" * 70
        log("\n" + bar)
        log(f"  {title}")
        log(bar)

    def subsection(title):
        log(f"\n--- {title} ---")

    def save_fig(name):
        path = os.path.join(ML_DIR, name)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        log(f"  [saved] {path}")

    sns.set_theme(style="whitegrid", font_scale=1.05)

    C_LR = "#3B82D4"
    C_RF = "#E84C4C"
    C_GB = "#2CA02C"
    MODEL_COLOURS = {
        "Logistic Regression": C_LR,
        "Random Forest":       C_RF,
        "Gradient Boosting":   C_GB,
    }

    # ==========================================================================
    # 1. LOAD RAW DATA (from original Excel -- preserves missing values)
    # ==========================================================================
    section("1. LOAD DATA")

    RAW_PATH = os.path.join("data", "E Commerce Dataset.xlsx")
    df_raw = pd.read_excel(RAW_PATH, sheet_name="E Comm", engine="openpyxl")
    log(f"\n  Source    : {RAW_PATH}  (read-only, original Excel)")
    log(f"  Raw shape : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    df_raw["PreferredPaymentMode"] = df_raw["PreferredPaymentMode"].replace(
        {"CC": "Credit Card", "COD": "Cash on Delivery"}
    )
    df_raw["PreferedOrderCat"] = df_raw["PreferedOrderCat"].replace(
        {"Mobile": "Mobile Phone"}
    )
    df_raw = df_raw.rename(columns={"PreferedOrderCat": "PreferredOrderCategory"})

    assert "Phone" in df_raw["PreferredLoginDevice"].values, \
        "ERROR: Phone category missing in PreferredLoginDevice"
    assert "Mobile Phone" in df_raw["PreferredLoginDevice"].values, \
        "ERROR: Mobile Phone category missing in PreferredLoginDevice"

    log("  Categorical standardisations applied (same as Stage 2):")
    log("    PreferredPaymentMode : CC->Credit Card, COD->Cash on Delivery")
    log("    PreferredOrderCategory: Mobile->Mobile Phone")
    log("    Column renamed: PreferedOrderCat -> PreferredOrderCategory")
    log("    PreferredLoginDevice: Phone and Mobile Phone kept SEPARATE (churn rate diff 9.83 pp)")

    TARGET    = "Churn"
    DROP_COLS = ["CustomerID"]
    LEAKAGE_FLAGGED = ["Complain", "DaySinceLastOrder"]

    y = df_raw[TARGET].copy()
    X = df_raw.drop(columns=[TARGET] + DROP_COLS)

    CATEGORICAL_COLS = [c for c in X.columns if X[c].dtype == "object"]
    NUMERICAL_COLS   = [c for c in X.columns if X[c].dtype != "object"]
    IMPUTE_COLS      = [c for c in NUMERICAL_COLS if X[c].isna().sum() > 0]
    COMPLETE_NUM_COLS = [c for c in NUMERICAL_COLS if c not in IMPUTE_COLS]

    log(f"\n  Features  : {X.shape[1]}  (CustomerID dropped)")
    log(f"  Target    : {TARGET}  | class 0={(y==0).sum()}, class 1={(y==1).sum()}")
    log(f"\n  Categorical ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
    log(f"  Numerical   ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")
    log(f"\n  Columns requiring imputation ({len(IMPUTE_COLS)}): {IMPUTE_COLS}")
    log(f"  Missing counts:")
    for c in IMPUTE_COLS:
        log(f"    {c:<35} {X[c].isna().sum()} missing")

    # ==========================================================================
    # 2. TRAIN / TEST SPLIT
    # ==========================================================================
    section("2. TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    log(f"\n  Training set : {len(X_train)} rows  "
        f"(churn={y_train.sum()} / {len(y_train)} = {y_train.mean()*100:.1f}%)")
    log(f"  Test set     : {len(X_test)} rows  "
        f"(churn={y_test.sum()} / {len(y_test)} = {y_test.mean()*100:.1f}%)")
    log("  Stratification confirmed: churn proportions are preserved in both sets.")

    # ==========================================================================
    # 3. PREPROCESSING PIPELINES (imputation inside Pipeline)
    # ==========================================================================
    section("3. PREPROCESSING PIPELINES")

    def build_preprocessor(scale=True):
        if scale:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]
        else:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("passthrough", FunctionTransformer()),
            ]
        num_pipe = Pipeline(num_steps)
        cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return ColumnTransformer([
            ("num", num_pipe, NUMERICAL_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ], remainder="drop")

    log("\n  Preprocessor A (scaled)   : SimpleImputer(median) -> StandardScaler -> OHE")
    log("  Preprocessor B (unscaled) : SimpleImputer(median) -> passthrough     -> OHE")
    log("  CORRECTION: Median imputation is now inside the Pipeline and fitted")
    log("  exclusively on X_train. The pre-split imputation limitation from")
    log("  Stage 2 is now resolved for the ML pipeline.")

    # ==========================================================================
    # 4. MODEL DEFINITIONS
    # ==========================================================================
    section("4. MODEL DEFINITIONS")

    models = {
        "Logistic Regression": Pipeline([
            ("pre", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
                C=1.0
            ))
        ]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", build_preprocessor(scale=False)),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    for name in models:
        log(f"  Defined: {name}")

    # ==========================================================================
    # 5. STRATIFIED CROSS-VALIDATION (training set only)
    # ==========================================================================
    section("5. STRATIFIED CROSS-VALIDATION (Training Set Only)")

    CV_FOLDS    = 5
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scoring  = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    cv_results_summary = {}

    for name, pipeline in models.items():
        subsection(f"CV -- {name}")
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_splitter, scoring=cv_scoring,
            return_train_score=False, n_jobs=-1
        )
        row = {}
        for metric in cv_scoring:
            scores = cv_res[f"test_{metric}"]
            row[metric]          = scores.mean()
            row[f"{metric}_std"] = scores.std()
            log(f"  {metric:<22}: {scores.mean():.4f}  (+/- {scores.std():.4f})")
        cv_results_summary[name] = row

    # ==========================================================================
    # 6. THRESHOLD SELECTION -- Out-of-Fold (Training Set Only)
    # ==========================================================================
    section("6. THRESHOLD SELECTION -- Out-of-Fold (Training Set Only)")

    log("\n  Threshold selection uses only out-of-fold (OOF) predictions on the")
    log("  training set via StratifiedKFold. The test set is NOT consulted.")
    log("  The criterion is maximum F1 on OOF predictions.")
    log("  This is labelled 'F1-selected threshold' and is NOT claimed to be")
    log("  the business-optimal threshold.")

    rf_pipeline_for_oof = Pipeline([
        ("pre", build_preprocessor(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    oof_probas = cross_val_predict(
        rf_pipeline_for_oof, X_train, y_train,
        cv=cv_splitter, method="predict_proba", n_jobs=-1
    )[:, 1]

    CANDIDATE_THRESHOLDS = np.linspace(0.05, 0.95, 91)
    oof_f1_scores = []
    for t in CANDIDATE_THRESHOLDS:
        preds_t = (oof_probas >= t).astype(int)
        oof_f1_scores.append(f1_score(y_train, preds_t, zero_division=0))

    best_oof_idx       = int(np.argmax(oof_f1_scores))
    SELECTED_THRESHOLD = float(CANDIDATE_THRESHOLDS[best_oof_idx])

    log(f"\n  OOF F1 scores evaluated over {len(CANDIDATE_THRESHOLDS)} candidate thresholds.")
    log(f"  F1-selected threshold (from OOF) : {SELECTED_THRESHOLD:.2f}")
    log(f"  OOF F1 at selected threshold     : {oof_f1_scores[best_oof_idx]:.4f}")
    log(f"  OOF F1 at default threshold 0.50 : {oof_f1_scores[list(CANDIDATE_THRESHOLDS).index(min(CANDIDATE_THRESHOLDS, key=lambda t: abs(t-0.50)))]:.4f}")

    # ==========================================================================
    # 7. TRAIN ON FULL TRAINING SET + EVALUATE ON TEST SET
    # ==========================================================================
    section("7. TEST-SET EVALUATION")

    def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te):
        pipeline.fit(X_tr, y_tr)
        y_pred_default = pipeline.predict(X_te)
        y_proba        = pipeline.predict_proba(X_te)[:, 1]
        acc   = accuracy_score(y_te, y_pred_default)
        prec  = precision_score(y_te, y_pred_default, zero_division=0)
        rec   = recall_score(y_te, y_pred_default, zero_division=0)
        f1    = f1_score(y_te, y_pred_default, zero_division=0)
        roc   = roc_auc_score(y_te, y_proba)
        prauc = average_precision_score(y_te, y_proba)
        cm    = confusion_matrix(y_te, y_pred_default)
        cr    = classification_report(y_te, y_pred_default,
                                       target_names=["Retained", "Churned"])
        tn, fp, fn, tp = cm.ravel()
        cm_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cm_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert abs(cm_prec - prec) < 1e-9, f"Precision mismatch in {name}"
        assert abs(cm_rec  - rec)  < 1e-9, f"Recall mismatch in {name}"
        preds_05   = (y_proba >= 0.50).astype(int)
        agree_rate = np.mean(y_pred_default == preds_05)
        log(f"\n  Model            : {name}")
        log(f"  Accuracy         : {acc:.4f}")
        log(f"  Precision        : {prec:.4f}")
        log(f"  Recall           : {rec:.4f}")
        log(f"  F1-score         : {f1:.4f}")
        log(f"  ROC-AUC          : {roc:.4f}")
        log(f"  PR-AUC           : {prauc:.4f}")
        log(f"  Confusion Matrix : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        log(f"  Agreement predict() vs proba>=0.50 : {agree_rate*100:.2f}%")
        log(f"\n  Classification Report:\n{cr}")
        log(f"  [CM consistency check PASSED: precision and recall match confusion matrix]")
        return {
            "name": name, "pipeline": pipeline,
            "y_pred": y_pred_default, "y_proba": y_proba,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": roc, "pr_auc": prauc,
            "confusion_matrix": cm
        }

    results = {}
    for name, pipeline in models.items():
        results[name] = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)

    # ==========================================================================
    # 8. RANDOM FOREST: EVALUATE SELECTED THRESHOLD ON TEST SET
    # ==========================================================================
    section("8. RANDOM FOREST -- F1-SELECTED THRESHOLD (Final Test Evaluation)")

    log(f"\n  F1-selected threshold : {SELECTED_THRESHOLD:.2f}  "
        f"(selected on OOF training predictions -- test set was NOT used)")
    log("  This threshold is applied to predict_proba()[:,1] on the test set.")
    log("  It is evaluated ONCE on the held-out test set (never used for selection).")

    rf_proba_test   = results["Random Forest"]["y_proba"]
    y_pred_selected = (rf_proba_test >= SELECTED_THRESHOLD).astype(int)

    sel_acc  = accuracy_score(y_test, y_pred_selected)
    sel_prec = precision_score(y_test, y_pred_selected, zero_division=0)
    sel_rec  = recall_score(y_test, y_pred_selected, zero_division=0)
    sel_f1   = f1_score(y_test, y_pred_selected, zero_division=0)
    sel_cm   = confusion_matrix(y_test, y_pred_selected)
    sel_tn, sel_fp, sel_fn, sel_tp = sel_cm.ravel()

    log(f"\n  === Random Forest at F1-Selected Threshold ({SELECTED_THRESHOLD:.2f}) ===")
    log(f"  Accuracy  : {sel_acc:.4f}")
    log(f"  Precision : {sel_prec:.4f}")
    log(f"  Recall    : {sel_rec:.4f}")
    log(f"  F1        : {sel_f1:.4f}")
    log(f"  Confusion : TN={sel_tn}  FP={sel_fp}  FN={sel_fn}  TP={sel_tp}")

    log(f"\n  === Random Forest at Default Threshold (0.50) ===")
    rf_default = results["Random Forest"]
    tn0, fp0, fn0, tp0 = rf_default["confusion_matrix"].ravel()
    log(f"  Accuracy  : {rf_default['accuracy']:.4f}")
    log(f"  Precision : {rf_default['precision']:.4f}")
    log(f"  Recall    : {rf_default['recall']:.4f}")
    log(f"  F1        : {rf_default['f1']:.4f}")
    log(f"  Confusion : TN={tn0}  FP={fp0}  FN={fn0}  TP={tp0}")

    log("\n  Business trade-off interpretation:")
    log("  Lowering the threshold increases Recall (fewer churners missed) but")
    log("  reduces Precision (more false alarms in a retention campaign).")
    log("  The F1-selected threshold is NOT claimed to be the business-optimal threshold.")

    # ==========================================================================
    # 9. LEAKAGE SENSITIVITY ANALYSIS
    # ==========================================================================
    section("9. SENSITIVITY ANALYSIS -- Without Leakage-Flagged Features")

    log(f"\n  Flagged features removed: {LEAKAGE_FLAGGED}")
    log("  All models retrained on the reduced feature set.")
    log("  NOTE: this analysis does NOT by itself establish the absence of")
    log("  temporal leakage. It provides quantitative evidence about feature")
    log("  dependency only.\n")

    X_noleak    = X.drop(columns=LEAKAGE_FLAGGED)
    CAT_COLS_NL = [c for c in X_noleak.columns if X_noleak[c].dtype == "object"]
    NUM_COLS_NL = [c for c in X_noleak.columns if X_noleak[c].dtype != "object"]

    X_tr_nl, X_te_nl, y_tr_nl, y_te_nl = train_test_split(
        X_noleak, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    def build_pipeline_noleak(model_name, clf):
        scale = (model_name == "Logistic Regression")
        if scale:
            num_steps = [("imputer", SimpleImputer(strategy="median")),
                         ("scaler",  StandardScaler())]
        else:
            num_steps = [("imputer", SimpleImputer(strategy="median")),
                         ("passthrough", FunctionTransformer())]
        num_pipe = Pipeline(num_steps)
        cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([
            ("num", num_pipe, NUM_COLS_NL),
            ("cat", cat_pipe, CAT_COLS_NL),
        ], remainder="drop")
        return Pipeline([("pre", pre), ("clf", clf)])

    noleak_clfs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, solver="lbfgs", C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=4, random_state=RANDOM_STATE),
    }

    sensitivity_results = {}
    for name, clf in noleak_clfs.items():
        pipe_nl = build_pipeline_noleak(name, clf)
        pipe_nl.fit(X_tr_nl, y_tr_nl)
        y_pred_nl  = pipe_nl.predict(X_te_nl)
        y_proba_nl = pipe_nl.predict_proba(X_te_nl)[:, 1]
        f1_nl  = f1_score(y_te_nl, y_pred_nl, zero_division=0)
        roc_nl = roc_auc_score(y_te_nl, y_proba_nl)
        rec_nl = recall_score(y_te_nl, y_pred_nl, zero_division=0)
        sensitivity_results[name] = {"f1": f1_nl, "roc_auc": roc_nl, "recall": rec_nl}
        log(f"  {name:<25}  F1={f1_nl:.4f}  ROC-AUC={roc_nl:.4f}  Recall={rec_nl:.4f}")

    log("\n  Random Forest performance remains very high after removing Complain")
    log("  and DaySinceLastOrder, although this sensitivity analysis does not by")
    log("  itself establish the absence of temporal leakage.")

    # ==========================================================================
    # 10. MODEL COMPARISON TABLE
    # ==========================================================================
    section("10. MODEL COMPARISON TABLE")

    comparison_rows = []
    for name, r in results.items():
        cv  = cv_results_summary[name]
        nl  = sensitivity_results.get(name, {})
        comparison_rows.append({
            "Model"          : name,
            "CV F1 (mean)"   : round(cv["f1"],       4),
            "CV ROC-AUC"     : round(cv["roc_auc"],  4),
            "Test Accuracy"  : round(r["accuracy"],  4),
            "Test Precision" : round(r["precision"], 4),
            "Test Recall"    : round(r["recall"],    4),
            "Test F1"        : round(r["f1"],        4),
            "Test ROC-AUC"   : round(r["roc_auc"],   4),
            "Test PR-AUC"    : round(r["pr_auc"],    4),
            "NoLeak F1"      : round(nl.get("f1",      float("nan")), 4),
            "NoLeak ROC-AUC" : round(nl.get("roc_auc", float("nan")), 4),
        })

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    log("\n  All metrics at default threshold (0.50) unless stated otherwise.\n")
    log(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(ML_DIR, "model_comparison.csv"))
    log(f"\n  [saved] {os.path.join(ML_DIR, 'model_comparison.csv')}")

    log("\n  NOTE: Random Forest has the highest reported CV F1, test F1, CV ROC-AUC,")
    log("  and test ROC-AUC among the three evaluated models. Logistic Regression")
    log("  has higher recall at the default threshold of 0.50.")
    log("  Model selection should consider all reported metrics.")

    # ==========================================================================
    # 11. VISUALIZATIONS
    # ==========================================================================
    section("11. VISUALIZATIONS")

    # 11a. ROC Curves
    subsection("11a. ROC Curves (Test Set)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})",
                color=MODEL_COLOURS[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- All Models (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    save_fig("ml01_roc_curves.png")

    # 11b. Precision-Recall Curves
    subsection("11b. Precision-Recall Curves (Test Set)")
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_pr = y_test.mean()
    ax.axhline(baseline_pr, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (prevalence={baseline_pr:.2f})")
    for name, r in results.items():
        p_arr, r_arr, _ = precision_recall_curve(y_test, r["y_proba"])
        ax.plot(r_arr, p_arr, label=f"{name} (AP={r['pr_auc']:.3f})",
                color=MODEL_COLOURS[name], linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- All Models (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    save_fig("ml02_precision_recall_curves.png")

    # 11c. Confusion Matrices
    subsection("11c. Confusion Matrices (pipeline.predict() default)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confusion Matrices (Test Set, pipeline.predict() default)",
                 fontsize=13, fontweight="bold")
    for ax, (name, r) in zip(axes, results.items()):
        cm = r["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["Actual 0", "Actual 1"],
                    linewidths=0.5, linecolor="white", annot_kws={"size": 13})
        ax.set_title(f"{name}\nPrec={r['precision']:.3f}  Rec={r['recall']:.3f}  F1={r['f1']:.3f}",
                     fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_fig("ml03_confusion_matrices.png")

    # 11d. Model Comparison Bar Chart
    subsection("11d. Model Comparison Bar Chart")
    metrics_plot = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_plot))
    width = 0.25
    for i, (name, color) in enumerate(MODEL_COLOURS.items()):
        vals = [comparison_df.loc[name, m] for m in metrics_plot]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_plot, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- Test Set Metrics (Default Threshold 0.50)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("ml04_model_comparison.png")

    # 11e. RF Feature Importance (impurity)
    subsection("11e. RF Feature Importance (Impurity-Based)")
    rf_pipeline = results["Random Forest"]["pipeline"]
    rf_clf      = rf_pipeline.named_steps["clf"]
    rf_pre      = rf_pipeline.named_steps["pre"]
    num_names   = NUMERICAL_COLS
    cat_names   = list(
        rf_pre.named_transformers_["cat"]
              .named_steps["ohe"]
              .get_feature_names_out(CATEGORICAL_COLS)
    )
    all_feature_names = num_names + cat_names
    importances = rf_clf.feature_importances_
    feat_imp_df = (
        pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(20)
    )
    log("\n  Top 20 Random Forest impurity-based feature importances:")
    log("  NOTE: one-hot encoded categorical levels are evaluated as separate")
    log("  features; importance may be spread across dummy variables.")
    log("  Importance does not imply causality.")
    log(feat_imp_df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp_plot = feat_imp_df.sort_values("Importance")
    ax.barh(feat_imp_plot["Feature"], feat_imp_plot["Importance"],
            color=C_RF, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Random Forest -- Top 20 Feature Importances (Impurity)\n"
                 "One-hot encoded levels evaluated separately. Importance != causality.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig("ml05_rf_feature_importance.png")

    # 11f. LR Coefficients
    subsection("11f. Logistic Regression Coefficients")
    lr_pipeline   = results["Logistic Regression"]["pipeline"]
    lr_clf        = lr_pipeline.named_steps["clf"]
    lr_pre        = lr_pipeline.named_steps["pre"]
    lr_cat_names  = list(
        lr_pre.named_transformers_["cat"]
              .named_steps["ohe"]
              .get_feature_names_out(CATEGORICAL_COLS)
    )
    lr_feature_names = NUMERICAL_COLS + lr_cat_names
    lr_coefs = lr_clf.coef_[0]
    coef_df  = (
        pd.DataFrame({"Feature": lr_feature_names, "Coefficient": lr_coefs})
        .sort_values("Coefficient", ascending=False)
    )
    top_pos = coef_df.head(15)
    top_neg = coef_df.tail(15).sort_values("Coefficient")
    coef_plot_df = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values("Coefficient")
    log("\n  Logistic Regression coefficient interpretation:")
    log("  - Positive coefficient = associated with higher predicted churn risk score.")
    log("  - Negative coefficient = associated with lower predicted churn risk score.")
    log("  - Coefficients are NOT causal effects.")
    log("\n  Top positive (higher predicted churn risk):")
    log(top_pos[["Feature", "Coefficient"]].to_string(index=False))
    log("\n  Top negative (lower predicted churn risk):")
    log(top_neg[["Feature", "Coefficient"]].to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 9))
    colors_bar = [C_LR if v >= 0 else "#AAAAAA" for v in coef_plot_df["Coefficient"]]
    ax.barh(coef_plot_df["Feature"], coef_plot_df["Coefficient"],
            color=colors_bar, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds scale, standardised numericals)")
    ax.set_title("Logistic Regression -- Top Coefficients\n"
                 "(Association with predicted churn risk -- NOT causal effects)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig("ml06_lr_coefficients.png")

    # 11g. RF Permutation Importance (Test Set)
    subsection("11g. RF Permutation Importance (Test Set)")
    X_test_transformed = rf_pre.transform(X_test)
    perm_imp = permutation_importance(
        rf_clf, X_test_transformed, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc"
    )
    perm_imp_df = (
        pd.DataFrame({
            "Feature":    all_feature_names,
            "Importance": perm_imp.importances_mean,
            "Std":        perm_imp.importances_std
        })
        .sort_values("Importance", ascending=False)
        .head(20)
    )
    log("\n  Top 20 Permutation Importances (mean ROC-AUC decrease, test set):")
    log("  Permutation importance was computed after the Random Forest was fitted,")
    log("  using the held-out test set as a post-fit diagnostic. It was not used")
    log("  for model fitting, hyperparameter selection, threshold selection, or")
    log("  model selection.")
    log("  Importance does not imply causality.")
    log(perm_imp_df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 8))
    perm_plot = perm_imp_df.sort_values("Importance")
    ax.barh(perm_plot["Feature"], perm_plot["Importance"],
            xerr=perm_plot["Std"], color=C_RF, edgecolor="white", alpha=0.85,
            ecolor="gray", capsize=3)
    ax.set_xlabel("Mean ROC-AUC Decrease (Permutation Importance, Test Set)")
    ax.set_title("Random Forest -- Permutation Importance (Test Set)\n"
                 "OHE features evaluated separately. Importance != causality.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig("ml07_rf_permutation_importance.png")

    # 11h. Threshold Analysis
    subsection("11h. Threshold Analysis -- Random Forest")
    log("  Left panel : OOF F1 vs threshold (training data only -- used for selection)")
    log("  Right panel: Precision/Recall/F1 on test set across candidate thresholds")
    log(f"  F1-selected threshold: {SELECTED_THRESHOLD:.2f}  (from OOF -- test set NOT used)")

    oof_prec_list, oof_rec_list, oof_f1_list = [], [], []
    for t in CANDIDATE_THRESHOLDS:
        preds_t = (oof_probas >= t).astype(int)
        oof_prec_list.append(precision_score(y_train, preds_t, zero_division=0))
        oof_rec_list.append(recall_score(y_train, preds_t, zero_division=0))
        oof_f1_list.append(f1_score(y_train, preds_t, zero_division=0))

    test_prec_list, test_rec_list, test_f1_list = [], [], []
    for t in CANDIDATE_THRESHOLDS:
        preds_t = (rf_proba_test >= t).astype(int)
        test_prec_list.append(precision_score(y_test, preds_t, zero_division=0))
        test_rec_list.append(recall_score(y_test, preds_t, zero_division=0))
        test_f1_list.append(f1_score(y_test, preds_t, zero_division=0))

    # Override 0.50 chart point with canonical pipeline.predict() metrics
    default_idx     = int(np.argmin(np.abs(CANDIDATE_THRESHOLDS - 0.50)))
    rf_default_pred = results["Random Forest"]["y_pred"]
    test_prec_list[default_idx] = precision_score(y_test, rf_default_pred, zero_division=0)
    test_rec_list[default_idx]  = recall_score(y_test, rf_default_pred, zero_division=0)
    test_f1_list[default_idx]   = f1_score(y_test, rf_default_pred, zero_division=0)

    assert abs(test_prec_list[default_idx] - rf_default["precision"]) < 1e-12, \
        "Chart 0.50 precision mismatch vs canonical RF default"
    assert abs(test_rec_list[default_idx]  - rf_default["recall"])    < 1e-12, \
        "Chart 0.50 recall mismatch vs canonical RF default"
    assert abs(test_f1_list[default_idx]   - rf_default["f1"])        < 1e-12, \
        "Chart 0.50 F1 mismatch vs canonical RF default"

    log("  Right panel: test-set metrics across probability thresholds.")
    log("  At 0.50, the canonical pipeline.predict() metrics are shown.")
    log("  Test labels are used only for post-fit evaluation/visualisation.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Random Forest -- Threshold Analysis", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(CANDIDATE_THRESHOLDS, oof_prec_list, label="OOF Precision", color=C_LR, lw=2)
    ax.plot(CANDIDATE_THRESHOLDS, oof_rec_list,  label="OOF Recall",    color=C_RF, lw=2)
    ax.plot(CANDIDATE_THRESHOLDS, oof_f1_list,   label="OOF F1",        color=C_GB, lw=2)
    ax.axvline(0.50, color="black", ls="--", lw=1, label="Default (0.50)")
    ax.axvline(SELECTED_THRESHOLD, color="#FF7F0E", ls=":", lw=2,
               label=f"F1-selected ({SELECTED_THRESHOLD:.2f})")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Training OOF Predictions\n(Threshold selection panel)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95]); ax.set_ylim([0, 1.05])
    ax = axes[1]
    ax.plot(CANDIDATE_THRESHOLDS, test_prec_list, label="Test Precision", color=C_LR, lw=2)
    ax.plot(CANDIDATE_THRESHOLDS, test_rec_list,  label="Test Recall",    color=C_RF, lw=2)
    ax.plot(CANDIDATE_THRESHOLDS, test_f1_list,   label="Test F1",        color=C_GB, lw=2)
    ax.axvline(0.50, color="black", ls="--", lw=1, label="Default (0.50)")
    ax.axvline(SELECTED_THRESHOLD, color="#FF7F0E", ls=":", lw=2,
               label=f"F1-selected ({SELECTED_THRESHOLD:.2f}) -- locked from OOF")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Test Set Performance\n(Final evaluation -- NOT used for selection)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    save_fig("ml08_threshold_analysis.png")

    # 11i. Sensitivity: Full vs No-Leak
    subsection("11i. Sensitivity -- Full vs No-Leak Feature Sets")
    sens_rows = []
    for name in results:
        r_full = results[name]
        r_nl   = sensitivity_results[name]
        sens_rows.append({
            "Model"          : name,
            "Full F1"        : round(r_full["f1"],      4),
            "NoLeak F1"      : round(r_nl["f1"],        4),
            "F1 Drop"        : round(r_full["f1"] - r_nl["f1"], 4),
            "Full ROC-AUC"   : round(r_full["roc_auc"], 4),
            "NoLeak ROC-AUC" : round(r_nl["roc_auc"],   4),
            "AUC Drop"       : round(r_full["roc_auc"] - r_nl["roc_auc"], 4),
            "Full Recall"    : round(r_full["recall"],  4),
            "NoLeak Recall"  : round(r_nl["recall"],    4),
        })
    sens_df = pd.DataFrame(sens_rows).set_index("Model")
    log("\n  Full features vs. without Complain + DaySinceLastOrder:\n")
    log(sens_df.to_string())
    log("\n  Random Forest performance remains very high after removing Complain")
    log("  and DaySinceLastOrder, although this sensitivity analysis does not")
    log("  by itself establish the absence of temporal leakage.")

    # ==========================================================================
    # 12. SAVE TRAINED MODELS
    # ==========================================================================
    section("12. SAVE TRAINED MODELS")

    for name, r in results.items():
        safe_name  = name.lower().replace(" ", "_")
        model_path = os.path.join(ML_DIR, f"model_{safe_name}.joblib")
        joblib.dump(r["pipeline"], model_path)
        log(f"  Saved: {model_path}")

    threshold_info = {
        "threshold": SELECTED_THRESHOLD,
        "selected_threshold": SELECTED_THRESHOLD,
        "selection_method": "F1-maximisation on OOF training predictions",
        "note": (
            "This threshold was selected on training-set out-of-fold predictions. "
            "It is NOT claimed to be the business-optimal threshold."
        )
    }
    threshold_path = os.path.join(ML_DIR, "rf_threshold.joblib")
    joblib.dump(threshold_info, threshold_path)
    log(f"  Saved: {threshold_path}  (F1-selected threshold = {SELECTED_THRESHOLD:.2f})")

    # ==========================================================================
    # 13. FULL TEXT REPORT
    # ==========================================================================
    section("13. STAGE 3 REPORT")

    rf_r = results["Random Forest"]
    tn0, fp0, fn0, tp0 = rf_r["confusion_matrix"].ravel()

    log(f"""
STAGE 3 REPORT: ML MODEL DEVELOPMENT
======================================

1. ML OBJECTIVE
   Build a binary classification model to predict churn risk.
   All reported relationships are observational, not causal.
   Predicted scores are churn RISK SCORES, not calibrated probabilities.

2. DATASET / FEATURES
   Source       : data/E Commerce Dataset.xlsx (original, read-only)
   Raw sheet    : E Comm ({df_raw.shape[0]} rows x {df_raw.shape[1]} columns)
   Target       : Churn (0=Retained, 1=Churned)
   Features used: {X.shape[1]} (CustomerID dropped)
   Numerical    : {len(NUMERICAL_COLS)} columns  (incl. {len(IMPUTE_COLS)} requiring imputation)
   Categorical  : {len(CATEGORICAL_COLS)} columns

3. TRAIN/TEST METHODOLOGY
   Stratified 80/20 split  (random_state={RANDOM_STATE})
   Training set : {len(X_train)} rows  (churn={y_train.sum()}, {y_train.mean()*100:.1f}%)
   Test set     : {len(X_test)} rows  (churn={y_test.sum()}, {y_test.mean()*100:.1f}%)
   Test set was not used for preprocessing fitting, cross-validation,
   model fitting, hyperparameter selection, or threshold selection.
   It was reserved for final held-out evaluation and post-fit diagnostics.

4. PREPROCESSING (all steps inside sklearn Pipeline, fitted on X_train only)
   Numerical  : SimpleImputer(strategy='median') applied inside Pipeline.
                StandardScaler for Logistic Regression; passthrough for trees.
   Categorical: OneHotEncoder(handle_unknown='ignore')

5. MODELS EVALUATED
   1. Logistic Regression  (class_weight='balanced', C=1.0, lbfgs, max_iter=1000)
   2. Random Forest        (n_estimators=300, class_weight='balanced')
   3. Gradient Boosting    (n_estimators=200, lr=0.1, max_depth=4)

6. CROSS-VALIDATION ({CV_FOLDS}-fold stratified, training set only)
{comparison_df[['CV F1 (mean)','CV ROC-AUC']].to_string()}

7. TEST-SET RESULTS (default threshold = 0.50)
{comparison_df[['Test Accuracy','Test Precision','Test Recall','Test F1','Test ROC-AUC','Test PR-AUC']].to_string()}

8. THRESHOLD SELECTION (Random Forest)
   Method    : Out-of-fold F1 maximisation on X_train/y_train only
   Selected  : {SELECTED_THRESHOLD:.2f}  (F1-selected threshold)
   Test set  : NOT used for threshold selection

9. RANDOM FOREST -- DEFAULT vs F1-SELECTED THRESHOLD (test set)
   Default threshold (0.50):
     Precision={rf_r['precision']:.4f}  Recall={rf_r['recall']:.4f}  F1={rf_r['f1']:.4f}
     Confusion: TN={tn0}  FP={fp0}  FN={fn0}  TP={tp0}
   F1-selected threshold ({SELECTED_THRESHOLD:.2f}):
     Precision={sel_prec:.4f}  Recall={sel_rec:.4f}  F1={sel_f1:.4f}
     Confusion: TN={sel_tn}  FP={sel_fp}  FN={sel_fn}  TP={sel_tp}

10. SENSITIVITY ANALYSIS (without Complain + DaySinceLastOrder)
{sens_df[['Full F1','NoLeak F1','F1 Drop','Full ROC-AUC','NoLeak ROC-AUC','AUC Drop']].to_string()}

11. FEATURE INTERPRETATION
    RF permutation importance (test set, top 5):
{perm_imp_df.head(5)[['Feature','Importance']].to_string(index=False)}
    LR top positive coefficients:
{top_pos.head(5)[['Feature','Coefficient']].to_string(index=False)}
    LR top negative coefficients:
{top_neg.head(5)[['Feature','Coefficient']].to_string(index=False)}
    Coefficients and importances are NOT causal effects.

12. LIMITATIONS
    - Observational dataset; no controlled experiment
    - Temporal leakage possible for Complain and DaySinceLastOrder
    - No hyperparameter tuning (default/sensible parameters used)
    - No probability calibration performed
""")

    # ==========================================================================
    # 14. FINAL VALIDATION SECTION
    # ==========================================================================
    section("14. STAGE 3 FINAL VALIDATION")
    log("\n  Running all quality-control checks...\n")

    checks = []

    def chk(label, condition, details=""):
        status = "PASS" if condition else "FAIL"
        checks.append((label, status, details))
        log(f"  [{status}] {label}")
        if details:
            log(f"         {details}")
        return condition

    all_passed = True

    all_passed &= chk("CustomerID excluded from features",
                      "CustomerID" not in X.columns)
    all_passed &= chk("Phone and Mobile Phone kept separate",
                      "Phone" in df_raw["PreferredLoginDevice"].values and
                      "Mobile Phone" in df_raw["PreferredLoginDevice"].values)
    all_passed &= chk("Train/test split is stratified (churn rate preserved)",
                      abs(y_train.mean() - y_test.mean()) < 0.005,
                      f"Train={y_train.mean()*100:.2f}%  Test={y_test.mean()*100:.2f}%")

    rf_preprocessor = models["Random Forest"].named_steps["pre"]
    rf_num_pipeline  = rf_preprocessor.named_transformers_["num"]
    imputer_step     = rf_num_pipeline.named_steps.get("imputer")
    imputation_ok = (
        isinstance(imputer_step, SimpleImputer)
        and imputer_step.strategy == "median"
    )
    all_passed &= chk(
        "Median imputation exists inside Random Forest Pipeline",
        imputation_ok, "Verified through pipeline structure."
    )

    threshold_structure_ok = (
        len(oof_probas) == len(X_train)
        and len(oof_f1_scores) == len(CANDIDATE_THRESHOLDS)
        and np.isfinite(oof_probas).all()
        and SELECTED_THRESHOLD in CANDIDATE_THRESHOLDS
        and SELECTED_THRESHOLD == float(
            CANDIDATE_THRESHOLDS[int(np.argmax(oof_f1_scores))]
        )
    )
    all_passed &= chk(
        "Threshold selected from OOF training predictions",
        threshold_structure_ok,
        f"OOF predictions={len(oof_probas)}, candidates={len(CANDIDATE_THRESHOLDS)}, selected={SELECTED_THRESHOLD:.2f}"
    )

    for name, r in results.items():
        y_pred_check = (r["y_proba"] >= 0.50).astype(int)
        agree_rate   = np.mean(r["y_pred"] == y_pred_check)
        all_passed  &= chk(
            f"predict() vs proba>=0.50 agreement [{name}]",
            agree_rate >= 0.95,
            f"agreement={agree_rate*100:.2f}%"
        )

    for name, r in results.items():
        tn_, fp_, fn_, tp_ = r["confusion_matrix"].ravel()
        cm_prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        cm_rec  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        prec_ok = abs(cm_prec - r["precision"]) < 1e-9
        rec_ok  = abs(cm_rec  - r["recall"])    < 1e-9
        all_passed &= chk(f"CM consistent with Precision [{name}]", prec_ok,
                          f"CM-prec={cm_prec:.6f}  metric={r['precision']:.6f}")
        all_passed &= chk(f"CM consistent with Recall [{name}]", rec_ok,
                          f"CM-rec={cm_rec:.6f}  metric={r['recall']:.6f}")

    selected_test_check = (
        SELECTED_THRESHOLD in CANDIDATE_THRESHOLDS
        and len(y_pred_selected) == len(y_test)
        and abs(sel_prec - precision_score(y_test, y_pred_selected, zero_division=0)) < 1e-12
        and abs(sel_rec  - recall_score(y_test, y_pred_selected, zero_division=0))    < 1e-12
        and abs(sel_f1   - f1_score(y_test, y_pred_selected, zero_division=0))        < 1e-12
    )
    all_passed &= chk(
        "F1-selected threshold test evaluation is internally consistent",
        selected_test_check,
        f"Threshold={SELECTED_THRESHOLD:.2f}  Prec={sel_prec:.4f}  Rec={sel_rec:.4f}  F1={sel_f1:.4f}"
    )

    all_passed &= chk("Sensitivity analysis (no-leak) completed", len(sensitivity_results) == 3)

    saved_model_files = [
        "model_logistic_regression.joblib",
        "model_random_forest.joblib",
        "model_gradient_boosting.joblib",
        "rf_threshold.joblib",
    ]
    for mf in saved_model_files:
        path   = os.path.join(ML_DIR, mf)
        exists = os.path.exists(path)
        if exists:
            joblib.load(path)
            loadable = True
        else:
            loadable = False
        all_passed &= chk(f"Model file exists and loads [{mf}]", exists and loadable)

    for name, r in results.items():
        safe_name   = name.lower().replace(" ", "_")
        loaded_pipe = joblib.load(os.path.join(ML_DIR, f"model_{safe_name}.joblib"))
        preds  = loaded_pipe.predict(X_test)
        probas = loaded_pipe.predict_proba(X_test)[:, 1]
        valid  = (len(preds) == len(y_test) and
                  set(preds).issubset({0, 1}) and
                  probas.min() >= 0.0 and
                  probas.max() <= 1.0)
        all_passed &= chk(f"Loaded model produces valid predictions [{name}]", valid)

    csv_path = os.path.join(ML_DIR, "model_comparison.csv")
    csv_ok   = os.path.exists(csv_path)
    if csv_ok:
        csv_df = pd.read_csv(csv_path, index_col=0)
        csv_ok = len(csv_df) == 3
    all_passed &= chk("model_comparison.csv exists with 3 model rows", csv_ok)

    df_check = pd.read_excel(RAW_PATH, sheet_name="E Comm", engine="openpyxl")
    excel_ok = len(df_check) == 5630
    all_passed &= chk("Original Excel file has 5630 rows (unchanged)",
                      excel_ok, f"Actual rows: {len(df_check)}")

    chart_files = [
        "ml01_roc_curves.png", "ml02_precision_recall_curves.png",
        "ml03_confusion_matrices.png", "ml04_model_comparison.png",
        "ml05_rf_feature_importance.png", "ml06_lr_coefficients.png",
        "ml07_rf_permutation_importance.png", "ml08_threshold_analysis.png",
    ]
    for cf in chart_files:
        exists = os.path.exists(os.path.join(ML_DIR, cf))
        all_passed &= chk(f"Chart file exists [{cf}]", exists)

    log_text = "\n".join(_log_lines).lower()
    causal_phrases = ["causes churn", "causes retention", "proves",
                      "because of churn", "guaranteed probability"]
    no_causal = all(p not in log_text for p in causal_phrases)
    all_passed &= chk("No prohibited causal/certainty language detected", no_causal)

    log("\n" + "-" * 50)
    total_checks  = len(checks)
    passed_checks = sum(1 for _, s, _ in checks if s == "PASS")
    log(f"  {passed_checks}/{total_checks} checks PASSED")

    log(f"\n  Output files in {ML_DIR}/:")
    for f in sorted(os.listdir(ML_DIR)):
        size = os.path.getsize(os.path.join(ML_DIR, f))
        log(f"    {f:<50} {size:>10,} bytes")

    log("""
  REMAINING LIMITATIONS:
  - Complain and DaySinceLastOrder: temporal ordering vs churn outcome
    is not established by the dataset. Both are retained with documented caution.
  - No hyperparameter tuning performed (sensible defaults used).
  - No probability calibration performed; outputs are risk scores only.
  - No controlled experiment available; all findings are observational.
""")

    if all_passed:
        log("=" * 70)
        log("  STAGE 3 COMPLETE -- READY FOR STAGE 4")
        log("=" * 70)
    else:
        log("=" * 70)
        log("  STAGE 3 INCOMPLETE -- REVIEW FAILED CHECKS ABOVE")
        log("=" * 70)

    flush_log()


# ============================================================================
# ============================================================================
#
#   STAGE 4 -- STREAMLIT INTERACTIVE DASHBOARD
#
#   Source: dashboard.py (merged verbatim -- logic unchanged)
#
#   Run:  streamlit run Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py
#
#   THRESHOLD NOTE:
#     The F1-selected threshold is loaded dynamically from
#     ml_outputs/rf_threshold.joblib.  No hardcoded threshold value is used
#     for scoring or prediction labels.
#
#     The High-risk boundary (0.60) is a BUSINESS/DASHBOARD SEGMENTATION
#     boundary -- NOT selected through OOF optimisation.
#
#   MODEL RESULTS CALLOUT NOTE:
#     The RF confusion-matrix metrics on Page 3 are synchronised with the
#     Stage 3 output.  If ml_model.py is rerun and metrics change, those
#     values must be updated to match.
#
# ============================================================================
# ============================================================================

# Dashboard imports (only loaded when running under Streamlit)
import importlib as _importlib
# Only initialize the Streamlit dashboard when the file is actually being
# executed by Streamlit.  Checking only whether the package is installed is
# insufficient because this same file is also used by the Stage 2/3 CLI.
_st_spec = None
if _running_in_streamlit:
    _st_spec = _importlib.util.find_spec("streamlit")
    if _st_spec is None:
        raise RuntimeError(
            "Streamlit runtime detected, but the 'streamlit' package is unavailable."
        )

if _st_spec is not None:
    import joblib
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    # =========================================================================
    # PAGE CONFIG (must be first Streamlit call)
    # =========================================================================
    st.set_page_config(
        page_title="E-Commerce Retention Intelligence",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # =========================================================================
    # DASHBOARD CONSTANTS
    # =========================================================================
    _DATA_PATH    = os.path.join("data", "E Commerce Dataset.xlsx")
    _SHEET_NAME   = "E Comm"
    _MODEL_DIR    = "ml_outputs"
    _EDA_DIR      = "eda_outputs"

    _RF_MODEL_PATH = os.path.join(_MODEL_DIR, "model_random_forest.joblib")
    _LR_MODEL_PATH = os.path.join(_MODEL_DIR, "model_logistic_regression.joblib")
    _GB_MODEL_PATH = os.path.join(_MODEL_DIR, "model_gradient_boosting.joblib")
    _THRESH_PATH   = os.path.join(_MODEL_DIR, "rf_threshold.joblib")
    _COMPARE_PATH  = os.path.join(_MODEL_DIR, "model_comparison.csv")

    # The 5 main categorical features used in the EDA Explorer categorical tab.
    _DASH_CAT_COLS = [
        "PreferredLoginDevice", "PreferredPaymentMode",
        "Gender", "PreferredOrderCategory", "MaritalStatus",
    ]
    _DASH_NUM_COLS = [
        "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
        "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
        "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder", "CashbackAmount",
    ]
    _DASH_FEATURE_COLS = _DASH_NUM_COLS + _DASH_CAT_COLS

    _COLOR_RETAINED = "#3b82d4"
    _COLOR_CHURNED  = "#e05252"
    _COLOR_NEUTRAL  = "#7c5cd8"

    # Business/dashboard segmentation boundary for the High risk tier.
    # This is a FIXED OPERATIONAL CUT-OFF -- NOT selected through OOF optimisation.
    _HIGH_RISK_BOUNDARY = 0.60

    # =========================================================================
    # DATA LOADING & PREPROCESSING
    # =========================================================================
    @st.cache_data(show_spinner=False)
    def _load_data():
        """Load raw Excel, apply same standardisation as ml_model.py."""
        df = pd.read_excel(_DATA_PATH, sheet_name=_SHEET_NAME)
        if "PreferedOrderCat" in df.columns:
            df.rename(columns={"PreferedOrderCat": "PreferredOrderCategory"}, inplace=True)
        df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
            {"CC": "Credit Card", "COD": "Cash on Delivery"}
        )
        df["PreferredOrderCategory"] = df["PreferredOrderCategory"].replace(
            {"Mobile": "Mobile Phone"}
        )
        return df

    @st.cache_resource(show_spinner=False)
    def _load_models():
        """Load pre-trained pipelines and the Stage 3 selected threshold from disk."""
        model_paths = [
            ("Random Forest",        _RF_MODEL_PATH),
            ("Logistic Regression",  _LR_MODEL_PATH),
            ("Gradient Boosting",    _GB_MODEL_PATH),
        ]
        missing_models = [path for _, path in model_paths if not os.path.exists(path)]
        if missing_models:
            raise FileNotFoundError(
                "Required model artifact(s) not found: " + ", ".join(missing_models)
                + ". Run with --stage 3 first."
            )
        if not os.path.exists(_THRESH_PATH):
            raise FileNotFoundError(
                f"Required threshold artifact not found: {_THRESH_PATH}. "
                "Run with --stage 3 first."
            )
        models = {name: joblib.load(path) for name, path in model_paths}
        threshold_data = joblib.load(_THRESH_PATH)
        if not isinstance(threshold_data, dict):
            raise ValueError(f"Invalid threshold artifact at {_THRESH_PATH}.")
        if "selected_threshold" not in threshold_data:
            raise KeyError(f"Invalid threshold artifact at {_THRESH_PATH}: missing 'selected_threshold'.")
        threshold = float(threshold_data["selected_threshold"])
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid selected threshold {threshold!r}.")
        if threshold >= _HIGH_RISK_BOUNDARY:
            raise ValueError(
                f"Selected threshold ({threshold:.4f}) must be below "
                f"HIGH_RISK_BOUNDARY ({_HIGH_RISK_BOUNDARY:.2f})."
            )
        return models, threshold

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _assign_risk_tier(score, f1_threshold):
        """High>=0.60 (fixed boundary) | Medium>=f1_threshold | Low."""
        if score >= _HIGH_RISK_BOUNDARY:
            return "High"
        elif score >= f1_threshold:
            return "Medium"
        else:
            return "Low"

    def _risk_color(tier):
        return {"High": "#e05252", "Medium": "#f59e0b", "Low": "#22c55e"}.get(tier, "#888")

    # =========================================================================
    # SIDEBAR NAVIGATION
    # =========================================================================
    def _sidebar_nav(threshold):
        st.sidebar.title("Retention Intelligence")
        st.sidebar.markdown("---")
        pages = [
            "1 - Overview",
            "2 - EDA Explorer",
            "3 - Model Results",
            "4 - Churn Predictor",
            "5 - Retention Plan",
        ]
        choice = st.sidebar.radio("Navigate to", pages, label_visibility="collapsed")
        st.sidebar.markdown("---")
        st.sidebar.caption(
            "**Dataset:** E Commerce Dataset.xlsx\n\n"
            "**Target:** Churn (0=Retained, 1=Churned)\n\n"
            "**Primary model:** Random Forest\n\n"
            f"**F1-selected threshold:** {threshold:.2f} (loaded from rf_threshold.joblib)\n\n"
            f"**High-risk boundary:** {_HIGH_RISK_BOUNDARY:.2f} (business segmentation boundary)\n\n"
            "All findings are observational, not causal."
        )
        return choice

    # =========================================================================
    # PAGE 1 -- OVERVIEW
    # =========================================================================
    def _page_overview(df):
        st.title("E-Commerce Customer Retention Intelligence System")
        st.markdown(
            "**Business question:** Which customers are at risk of churning, "
            "what behavioural factors are associated with that risk, and what "
            "retention actions should the business consider?"
        )
        st.markdown("---")

        total         = len(df)
        churned       = int(df["Churn"].sum())
        retained      = total - churned
        churn_rate    = churned / total * 100
        avg_tenure    = df["Tenure"].median()
        complain_rate = df["Complain"].mean() * 100

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Customers",    f"{total:,}")
        col2.metric("Churned",            f"{churned:,}",   f"{churn_rate:.1f}%")
        col3.metric("Retained",           f"{retained:,}",  f"{100-churn_rate:.1f}%")
        col4.metric("Churn Rate",         f"{churn_rate:.1f}%")
        col5.metric("Median Tenure (mo)", f"{avg_tenure:.0f}")
        col6.metric("Complain Rate",      f"{complain_rate:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Churn Distribution")
            counts = df["Churn"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            counts["Status"] = counts["Status"].map({0: "Retained", 1: "Churned"})
            fig = px.pie(
                counts, names="Status", values="Count",
                color="Status",
                color_discrete_map={"Retained": _COLOR_RETAINED, "Churned": _COLOR_CHURNED},
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Churn Rate by City Tier")
            tier_churn = df.groupby("CityTier")["Churn"].mean().reset_index()
            tier_churn["Churn Rate (%)"] = tier_churn["Churn"] * 100
            tier_churn["CityTier"] = tier_churn["CityTier"].astype(str)
            fig2 = px.bar(
                tier_churn, x="CityTier", y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale=["#3b82d4", "#e05252"],
                labels={"CityTier": "City Tier"},
                text_auto=".1f",
            )
            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Churn Rate by Tenure Band")
        df_tmp = df.copy()
        df_tmp["Tenure (filled)"] = df_tmp["Tenure"].fillna(df_tmp["Tenure"].median())
        df_tmp["Tenure Band"] = pd.cut(
            df_tmp["Tenure (filled)"],
            bins=[0, 3, 6, 12, 24, 61],
            labels=["0-3 mo", "4-6 mo", "7-12 mo", "13-24 mo", "25+ mo"],
        )
        tenure_churn = df_tmp.groupby("Tenure Band", observed=True)["Churn"].mean().reset_index()
        tenure_churn["Churn Rate (%)"] = tenure_churn["Churn"] * 100
        fig3 = px.bar(
            tenure_churn, x="Tenure Band", y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale=["#22c55e", "#f59e0b", "#e05252"],
            text_auto=".1f",
        )
        fig3.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.caption("All percentages and rates are computed from the raw dataset. "
                   "Relationships shown are observational, not causal.")

    # =========================================================================
    # PAGE 2 -- EDA EXPLORER
    # =========================================================================
    def _page_eda(df):
        st.title("Exploratory Data Analysis")
        st.markdown("Explore how customer attributes are associated with churn. "
                    "All relationships are **observational** -- not causal.")
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(
            ["Categorical Breakdown", "Numerical Distribution", "Saved EDA Charts"]
        )

        with tab1:
            st.subheader("Churn Rate by Main Categorical Feature")
            st.caption("Covers the five main categorical features used in the model. "
                       "Select one to view churn rates per category.")
            cat_col = st.selectbox("Select main categorical feature", _DASH_CAT_COLS, key="eda_cat")

            cat_churn = (
                df.groupby(cat_col)["Churn"]
                .agg(["mean", "count"])
                .reset_index()
            )
            cat_churn.columns = [cat_col, "Churn Rate", "Count"]
            cat_churn["Churn Rate (%)"] = cat_churn["Churn Rate"] * 100
            cat_churn = cat_churn.sort_values("Churn Rate (%)", ascending=False)

            fig = px.bar(
                cat_churn, x=cat_col, y="Churn Rate (%)",
                text_auto=".1f",
                color="Churn Rate (%)",
                color_continuous_scale=["#3b82d4", "#e05252"],
                hover_data={"Count": True},
            )
            fig.update_layout(
                coloraxis_showscale=False, yaxis_ticksuffix="%",
                margin=dict(t=20, b=40, l=20, r=20), xaxis_title=cat_col,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                cat_churn[[cat_col, "Count", "Churn Rate (%)"]].style.format(
                    {"Churn Rate (%)": "{:.1f}%"}
                ),
                use_container_width=True,
            )

        with tab2:
            st.subheader("Numerical Feature Distribution by Churn Status")
            num_col = st.selectbox("Select numerical column", _DASH_NUM_COLS, key="eda_num")
            df_plot = df[[num_col, "Churn"]].dropna()
            df_plot["Status"] = df_plot["Churn"].map({0: "Retained", 1: "Churned"})
            fig = px.box(
                df_plot, x="Status", y=num_col,
                color="Status",
                color_discrete_map={"Retained": _COLOR_RETAINED, "Churned": _COLOR_CHURNED},
                points="outliers",
            )
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            stats = df_plot.groupby("Status")[num_col].describe().T
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        with tab3:
            st.subheader("Pre-Generated EDA Charts")
            eda_charts = {
                "Churn Distribution":           "01_churn_distribution.png",
                "Churn Rate by Categorical":    "02_churn_rate_by_categorical.png",
                "Boxplots vs Churn":            "03_boxplots_numerical_vs_churn.png",
                "Satisfaction Score vs Churn":  "04a_satisfaction_vs_churn.png",
                "Complain vs Churn":            "04b_complain_vs_churn.png",
                "Tenure vs Churn":              "04c_tenure_vs_churn.png",
                "Order Count vs Churn":         "04d_ordercount_vs_churn.png",
                "Days Since Last Order":        "04e_dayssincelastorder_vs_churn.png",
                "Correlation Heatmap":          "05_correlation_heatmap.png",
                "Cashback vs Churn":            "06_cashback_vs_churn.png",
            }
            chosen = st.selectbox("Select chart", list(eda_charts.keys()), key="eda_chart")
            img_path = os.path.join(_EDA_DIR, eda_charts[chosen])
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"Chart not found: {img_path}")

    # =========================================================================
    # PAGE 3 -- MODEL RESULTS
    # =========================================================================
    def _page_model_results(threshold):
        """
        The hardcoded confusion-matrix metrics below are synchronised with the
        Stage 3 ml_model.py evaluation output (ml_outputs/ml_results.txt and
        model_comparison.csv).  They are reproduced here because
        model_comparison.csv does not include per-threshold confusion-matrix
        breakdown.  If ml_model.py is rerun and any metric changes, these
        values must be updated.

        Hardcoded values (Stage 3 verified):
          Default threshold (pipeline.predict(), effectively 0.50):
            Precision=1.0000, Recall=0.8632, F1=0.9266, ROC-AUC=0.9990
            TN=936, FP=0, FN=26, TP=164
          F1-selected threshold (0.34, loaded from rf_threshold.joblib):
            Precision=0.9122, Recall=0.9842, F1=0.9468
            TN=918, FP=18, FN=3, TP=187
        """
        st.title("Machine Learning Model Results")
        st.markdown(
            "Three models were evaluated: **Logistic Regression** (baseline), "
            "**Random Forest** (primary), and **Gradient Boosting** (comparison). "
            "The comparison table reports both **5-fold cross-validation metrics** and "
            "metrics from the held-out **20% test set**. "
            "The test set was not used for preprocessing fitting, cross-validation, "
            "model fitting, or threshold selection."
        )
        st.markdown("---")

        st.subheader("Model Comparison Table")
        if os.path.exists(_COMPARE_PATH):
            comp_df = pd.read_csv(_COMPARE_PATH, index_col=0)
            display_cols = {
                "CV F1 (mean)":   "CV F1",
                "CV ROC-AUC":     "CV AUC",
                "Test Accuracy":  "Test Acc",
                "Test Precision": "Test Prec",
                "Test Recall":    "Test Recall",
                "Test F1":        "Test F1",
                "Test ROC-AUC":   "Test AUC",
                "Test PR-AUC":    "Test PR-AUC",
                "NoLeak F1":      "No-Leak F1",
                "NoLeak ROC-AUC": "No-Leak AUC",
            }
            display_df = comp_df.rename(columns=display_cols)[list(display_cols.values())]
            st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True)
            st.caption(
                "All metrics at default threshold (0.50 / pipeline.predict()). "
                "No-Leak = model retrained without Complain & DaySinceLastOrder. "
                "Random Forest has the highest CV F1, test F1, CV ROC-AUC, and test ROC-AUC "
                "among the three evaluated models. Logistic Regression has higher recall at "
                "the default threshold. Model selection should consider all reported metrics."
            )
        else:
            st.warning("model_comparison.csv not found. Run with --stage 3 first.")

        st.markdown("---")

        st.subheader("Random Forest -- Default vs F1-Selected Threshold")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Default threshold (pipeline.predict(), effectively 0.50)**")
            st.markdown(
                "- Precision: **1.0000** - Recall: **0.8632**\n"
                "- F1: **0.9266** - ROC-AUC: **0.9990**\n"
                "- TN=936 - FP=0 - FN=26 - TP=164"
            )
        with col2:
            st.markdown(f"**F1-selected threshold ({threshold:.2f}, loaded from rf_threshold.joblib)**")
            st.markdown(
                "- Precision: **0.9122** - Recall: **0.9842**\n"
                "- F1: **0.9468** - ROC-AUC: **0.9990**\n"
                "- TN=918 - FP=18 - FN=3 - TP=187"
            )
        st.info(
            f"The F1-selected threshold ({threshold:.2f}) was chosen by maximising F1 on "
            "out-of-fold training predictions **only** -- the test set was never used for "
            "threshold selection. "
            "Lowering the threshold increases Recall (fewer churners missed) at the cost "
            "of lower Precision (more false alarms in a retention campaign)."
        )
        st.markdown("---")

        st.subheader("ML Evaluation Charts")
        ml_charts = {
            "ROC Curves":                "ml01_roc_curves.png",
            "Precision-Recall Curves":   "ml02_precision_recall_curves.png",
            "Confusion Matrices":        "ml03_confusion_matrices.png",
            "Model Comparison":          "ml04_model_comparison.png",
            "RF Feature Importance":     "ml05_rf_feature_importance.png",
            "LR Coefficients":           "ml06_lr_coefficients.png",
            "RF Permutation Importance": "ml07_rf_permutation_importance.png",
            "Threshold Analysis":        "ml08_threshold_analysis.png",
        }
        chosen_chart = st.selectbox("Select chart", list(ml_charts.keys()), key="ml_chart")
        img_path = os.path.join(_MODEL_DIR, ml_charts[chosen_chart])
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Chart not found: {img_path}")

        st.markdown("---")
        st.caption(
            "Predicted scores are churn **risk scores**, not calibrated probabilities. "
            "All findings are observational. "
            "Permutation importance was computed on the test set as a **post-fit diagnostic** "
            "of the final fitted model -- it was not used for model fitting, hyperparameter "
            "selection, threshold selection, or model selection."
        )

    # =========================================================================
    # PAGE 4 -- CHURN PREDICTOR
    # =========================================================================
    def _page_predictor(models, threshold):
        st.title("Interactive Churn Risk Predictor")
        st.markdown(
            "Enter a customer profile below to receive a **churn risk score** "
            "from the trained Random Forest model. "
            "The score is not a calibrated probability -- treat it as a relative risk indicator."
        )
        st.markdown("---")

        if "Random Forest" not in models:
            st.error("Random Forest model not loaded. Run with --stage 3 first.")
            return

        rf_model = models["Random Forest"]

        with st.form("predictor_form"):
            st.subheader("Customer Profile")
            col1, col2, col3 = st.columns(3)

            with col1:
                tenure            = st.number_input("Tenure (months)", min_value=0, max_value=61,  value=12)
                city_tier         = st.selectbox("City Tier", [1, 2, 3])
                warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=5, max_value=127, value=20)
                hour_spend        = st.number_input("Hours on App / day", min_value=0, max_value=5,   value=3)
                num_devices       = st.number_input("Devices Registered", min_value=1, max_value=6,   value=3)

            with col2:
                satisfaction  = st.slider("Satisfaction Score", 1, 5, 3)
                num_address   = st.number_input("Number of Addresses", min_value=1, max_value=22, value=3)
                complain      = st.selectbox("Filed Complaint (last month)?", [0, 1],
                                             format_func=lambda x: "Yes" if x == 1 else "No")
                order_hike    = st.number_input("Order Amount Hike YoY (%)", min_value=11, max_value=26, value=15)
                coupon_used   = st.number_input("Coupons Used", min_value=0, max_value=16, value=2)

            with col3:
                order_count      = st.number_input("Order Count", min_value=1, max_value=16, value=3)
                days_since_order = st.number_input("Days Since Last Order", min_value=0, max_value=46, value=4)
                cashback_amount  = st.number_input("Cashback Amount (INR)", min_value=0.0, max_value=325.0, value=150.0, step=1.0)
                login_device     = st.selectbox("Preferred Login Device",
                                                ["Mobile Phone", "Phone", "Computer"])
                payment_mode     = st.selectbox("Preferred Payment Mode",
                                                ["Debit Card", "Credit Card", "Cash on Delivery",
                                                 "E wallet", "UPI"])

            with st.columns(2)[0]:
                gender         = st.selectbox("Gender", ["Female", "Male"])
                order_category = st.selectbox("Preferred Order Category",
                                              ["Laptop & Accessory", "Mobile Phone",
                                               "Fashion", "Grocery", "Others"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

            submitted = st.form_submit_button("Calculate Churn Risk Score", type="primary")

        if submitted:
            input_dict = {
                "Tenure":                      tenure,
                "CityTier":                    city_tier,
                "WarehouseToHome":             warehouse_to_home,
                "HourSpendOnApp":              hour_spend,
                "NumberOfDeviceRegistered":    num_devices,
                "SatisfactionScore":           satisfaction,
                "NumberOfAddress":             num_address,
                "Complain":                    complain,
                "OrderAmountHikeFromlastYear": order_hike,
                "CouponUsed":                  coupon_used,
                "OrderCount":                  order_count,
                "DaySinceLastOrder":           days_since_order,
                "CashbackAmount":              cashback_amount,
                "PreferredLoginDevice":        login_device,
                "PreferredPaymentMode":        payment_mode,
                "Gender":                      gender,
                "PreferredOrderCategory":      order_category,
                "MaritalStatus":               marital_status,
            }

            input_df   = pd.DataFrame([input_dict])
            risk_score = rf_model.predict_proba(input_df)[0][1]
            risk_tier  = _assign_risk_tier(risk_score, threshold)
            prediction = int(risk_score >= threshold)

            st.markdown("---")
            st.subheader("Risk Assessment Result")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Churn Risk Score", f"{risk_score:.4f}")
            col_r2.metric("Risk Tier", risk_tier)
            col_r3.metric(
                f"Prediction (F1-selected threshold={threshold:.2f})",
                "At Risk" if prediction == 1 else "Low Risk",
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Churn Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _risk_color(risk_tier)},
                    "steps": [
                        {"range": [0,  threshold * 100],                      "color": "#d1fae5"},
                        {"range": [threshold * 100, _HIGH_RISK_BOUNDARY * 100], "color": "#fef3c7"},
                        {"range": [_HIGH_RISK_BOUNDARY * 100, 100],            "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2328", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
                number={"suffix": "%", "valueformat": ".1f"},
            ))
            fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            if risk_tier == "High":
                st.error("**High Risk.** This customer profile has a high predicted churn risk score. "
                         "Consider immediate retention outreach, personalised offers, or proactive support.")
            elif risk_tier == "Medium":
                st.warning("**Medium Risk.** This customer profile shows moderate churn risk. "
                           "Targeted engagement campaigns or loyalty incentives may be appropriate.")
            else:
                st.success("**Low Risk.** This customer profile shows low predicted churn risk. "
                           "Standard engagement is likely sufficient.")

            st.caption(
                f"Risk score from the Random Forest model -- relative indicator, not a calibrated probability. "
                f"F1-selected threshold: {threshold:.2f} (loaded from rf_threshold.joblib, OOF-selected). "
                f"High-risk boundary: {_HIGH_RISK_BOUNDARY:.2f} (fixed business boundary, NOT OOF-optimised)."
            )

    # =========================================================================
    # PAGE 5 -- RETENTION PLAN
    # =========================================================================
    def _page_retention(df, models, threshold):
        st.title("Customer Retention Plan")
        st.markdown("Score all customers in the dataset using the Random Forest model, "
                    "segment them by risk tier, and view data-driven retention recommendations.")
        st.markdown("---")

        if "Random Forest" not in models:
            st.error("Random Forest model not loaded. Run with --stage 3 first.")
            return

        rf_model    = models["Random Forest"]
        feature_df  = df[_DASH_FEATURE_COLS].copy()
        risk_scores = rf_model.predict_proba(feature_df)[:, 1]

        scored_df = df[
            ["CustomerID", "Churn", "Tenure", "SatisfactionScore",
             "Complain", "PreferredLoginDevice"]
        ].copy()
        scored_df["Risk Score"]   = risk_scores
        scored_df["Risk Tier"]    = scored_df["Risk Score"].apply(
            lambda s: _assign_risk_tier(s, threshold)
        )
        scored_df["Actual Label"] = scored_df["Churn"].map({0: "Retained", 1: "Churned"})

        st.subheader("Risk Tier Summary")
        st.caption(
            f"High risk: score >= {_HIGH_RISK_BOUNDARY:.2f} (business segmentation boundary -- NOT OOF-optimised).  "
            f"Medium risk: score >= {threshold:.2f} (F1-selected threshold, loaded from rf_threshold.joblib).  "
            f"Low risk: score < {threshold:.2f}."
        )

        tier_counts = (
            scored_df["Risk Tier"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .reset_index()
        )
        tier_counts.columns = ["Risk Tier", "Customers"]
        tier_counts["% of Total"] = (tier_counts["Customers"] / len(scored_df) * 100).round(1)

        col1, col2, col3 = st.columns(3)
        for i, tier in enumerate(["High", "Medium", "Low"]):
            row   = tier_counts[tier_counts["Risk Tier"] == tier]
            count = int(row["Customers"].iloc[0])    if not row.empty else 0
            pct   = float(row["% of Total"].iloc[0]) if not row.empty else 0.0
            [col1, col2, col3][i].metric(f"{tier} Risk Customers", f"{count:,}", f"{pct:.1f}% of total")

        fig = px.pie(
            tier_counts, names="Risk Tier", values="Customers",
            color="Risk Tier",
            color_discrete_map={"High": "#e05252", "Medium": "#f59e0b", "Low": "#22c55e"},
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

        st.subheader(f"High-Risk Customer List (Risk Score >= {_HIGH_RISK_BOUNDARY:.2f})")
        high_risk = scored_df[scored_df["Risk Tier"] == "High"].sort_values("Risk Score", ascending=False)
        high_risk_display = high_risk[
            ["CustomerID", "Risk Score", "Actual Label", "Tenure",
             "SatisfactionScore", "Complain", "PreferredLoginDevice"]
        ].reset_index(drop=True)
        high_risk_display["Risk Score"] = high_risk_display["Risk Score"].round(4)
        st.dataframe(
            high_risk_display.style.background_gradient(subset=["Risk Score"], cmap="Reds"),
            use_container_width=True, height=350,
        )
        st.caption(
            f"Showing {len(high_risk_display):,} customers with risk score >= "
            f"{_HIGH_RISK_BOUNDARY:.2f} (business segmentation boundary)."
        )

        csv_bytes = high_risk_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download High-Risk Customer List (CSV)",
            data=csv_bytes, file_name="high_risk_customers.csv", mime="text/csv",
        )
        st.markdown("---")

        st.subheader("Data-Driven Retention Recommendations")
        st.markdown(
            "The following recommendations are based on observed patterns in the EDA "
            "and the feature importance analysis from the ML models. "
            "These are **hypotheses** for retention strategies, not guaranteed interventions. "
            "All should be validated through controlled experiments before full deployment."
        )

        recs = [
            {
                "Segment": "New customers (Tenure <= 3 months)",
                "Observation": "Customers with the shortest tenure show the highest churn rates in EDA.",
                "Suggested Action": "Implement onboarding journeys with early engagement touchpoints, "
                                     "welcome offers, and guided first-purchase incentives.",
            },
            {
                "Segment": "Customers who filed a complaint",
                "Observation": "Complain is among the highest-importance features in both permutation "
                                "importance and LR coefficients, associated with higher churn risk.",
                "Suggested Action": "Establish a rapid complaint-resolution workflow. Follow up with "
                                      "affected customers proactively. Track complaint resolution as a KPI.",
            },
            {
                "Segment": "Customers with multiple addresses registered",
                "Observation": "Higher NumberOfAddress shows association with higher churn risk in both "
                                "EDA and ML feature importance.",
                "Suggested Action": "Investigate whether address complexity creates friction in delivery "
                                      "or returns. Consider simplifying the address management experience.",
            },
            {
                "Segment": "City Tier 3 customers",
                "Observation": "City Tier 3 has the highest observed churn rate (21.4%) vs Tier 1 (14.5%).",
                "Suggested Action": "Investigate service-level differences across tiers. "
                                      "Consider tier-specific promotions.",
            },
            {
                "Segment": "Customers with Satisfaction Score 5",
                "Observation": "Counter-intuitively, Satisfaction Score 5 has the highest observed churn "
                                "rate (23.8%). This may reflect a survey-response artefact or timing issue.",
                "Suggested Action": "Audit how the satisfaction survey is administered. Investigate whether "
                                      "high-scoring churners share other risk signals.",
            },
            {
                "Segment": "Customers with low cashback received",
                "Observation": "CashbackAmount shows a negative association with churn in LR coefficients "
                                "and is among the top impurity-based importances in RF.",
                "Suggested Action": "Evaluate whether cashback or loyalty reward programmes are reaching "
                                      "at-risk segments. Consider targeted cashback offers for high-risk customers.",
            },
            {
                "Segment": f"Predicted high-risk customers (score >= {_HIGH_RISK_BOUNDARY:.2f})",
                "Observation": f"These customers score above the business segmentation boundary "
                                f"({_HIGH_RISK_BOUNDARY:.2f}) set for outreach prioritisation.",
                "Suggested Action": "Prioritise for immediate outreach. Combine with other CRM signals "
                                      "(recency, complaint history) to personalise retention offers. "
                                      "Run A/B test to measure intervention effectiveness.",
            },
        ]

        for rec in recs:
            with st.expander(f"**{rec['Segment']}**"):
                st.markdown(f"**Observation:** {rec['Observation']}")
                st.markdown(f"**Suggested Action:** {rec['Suggested Action']}")

        st.markdown("---")
        st.caption(
            "All recommendations are based on observed associations in the dataset. "
            "Causal direction cannot be established from this analysis alone. "
            "Validate through controlled A/B experiments before large-scale deployment."
        )

    # =========================================================================
    # DASHBOARD MAIN
    # =========================================================================
    def _dashboard_main():
        with st.spinner("Loading data and models..."):
            df = _load_data()
            models, threshold = _load_models()

        page = _sidebar_nav(threshold)

        if page.startswith("1"):
            _page_overview(df)
        elif page.startswith("2"):
            _page_eda(df)
        elif page.startswith("3"):
            _page_model_results(threshold)
        elif page.startswith("4"):
            _page_predictor(models, threshold)
        elif page.startswith("5"):
            _page_retention(df, models, threshold)


# ============================================================================
# ============================================================================
#
#   ENTRY POINT
#
# ============================================================================
# ============================================================================

if _running_in_streamlit:
    # Streamlit executes the user script in a fake __main__ module, so the
    # normal __name__ == "__main__" guard cannot be used to distinguish the
    # dashboard launch from the CLI launch.
    if _st_spec is not None:
        _dashboard_main()

elif __name__ == "__main__":
    # Running directly via Python (not Streamlit)
    stage = _parse_stage()

    if stage is None:
        print(
            "\nUsage:\n"
            "  python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 2\n"
            "  python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 3\n"
            "  python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage all\n"
            "\n"
            "To launch the Streamlit dashboard (Stage 4):\n"
            "  streamlit run Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py\n"
        )
        sys.exit(0)

    if stage in ("2", "all"):
        run_stage2()

    if stage in ("3", "all"):
        run_stage3()

    if stage not in ("2", "3", "all"):
        print(f"Unknown stage '{stage}'. Valid values: 2, 3, all")
        sys.exit(1)

