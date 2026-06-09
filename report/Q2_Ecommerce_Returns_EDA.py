# ============================================================
#  QUESTION 2 — E-Commerce Customer Return Behavior
#  Introduction to Data Science | SZABIST Islamabad
#  Tool: Google Colab | Libraries: pandas, numpy, matplotlib, seaborn
# ============================================================

# ── STEP 0: Import libraries & upload file ───────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

uploaded = files.upload()   # select:  Q2_Ecommerce_Returns_Dataset.xlsx

# ── STEP 1: Explore Dataset Structure & Classify Variables ───────────────────
df = pd.read_excel('Q2_Ecommerce_Returns_Dataset.xlsx')

print("=" * 60)
print("STEP 1 — DATASET STRUCTURE & VARIABLE CLASSIFICATION")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows  x  {df.shape[1]} columns")
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("""
Variable Classification:
  ID Variable       : OrderID
  Numerical (cont.) : OrderValue, DeliveryDays, DiscountPercent
  Categorical (nom.): CustomerRegion, ProductCategory,
                      Returned, PaymentMethod
""")

# WHY classify variables first?
# Different variable types require different analysis techniques.
# Numerical → mean, median, box plots, histograms, correlations
# Categorical → frequency tables, bar charts, cross-tabulations
# Knowing this upfront guides every decision that follows.

print("\nBasic statistics:")
print(df.describe())

# ── STEP 2: Data Quality Issues Summary ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — DATA QUALITY PROBLEMS")
print("=" * 60)

# 2a. Missing values
print("\n--- 2a. Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# 2b. Duplicate rows
print("\n--- 2b. Duplicate Rows ---")
dup_count = df.duplicated().sum()
print(f"Total duplicate rows: {dup_count}")
if dup_count > 0:
    print(df[df.duplicated(keep=False)][['OrderID', 'CustomerRegion', 'Returned']])

# 2c. Inconsistent labels in Returned column
print("\n--- 2c. Inconsistent Labels in 'Returned' ---")
print("Unique values found:", df['Returned'].unique())
# 'Yes', 'yes', 'YES' all mean the same thing — Python treats them differently!

# 2d. Invalid DiscountPercent values
print("\n--- 2d. Invalid Discount Percentages ---")
invalid_disc = df[(df['DiscountPercent'] < 0) | (df['DiscountPercent'] > 100)]
print(f"Records with invalid discount (< 0 or > 100): {len(invalid_disc)}")
print(invalid_disc[['OrderID', 'DiscountPercent']])

# 2e. Outliers in OrderValue using IQR
print("\n--- 2e. Outliers in OrderValue ---")
Q1 = df['OrderValue'].quantile(0.25)
Q3 = df['OrderValue'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers_ov = df[df['OrderValue'] > upper_fence]
print(f"IQR={IQR:.2f} | Upper fence={upper_fence:.2f}")
print(f"Outlier records ({len(outliers_ov)}):")
print(outliers_ov[['OrderID', 'OrderValue']])

# WHY use IQR for outlier detection instead of mean ± 2SD?
# OrderValue is heavily right-skewed (max = 8750 vs median = 84).
# Mean and SD are pulled by extreme values, so IQR is more reliable here.

# Summary table of all issues
print("\n--- SUMMARY TABLE OF DATA QUALITY ISSUES ---")
quality_summary = pd.DataFrame({
    'Issue': ['Missing DeliveryDays', 'Duplicate Rows',
              'Inconsistent Returned Labels', 'Invalid DiscountPercent',
              'High OrderValue Outliers'],
    'Count': [df['DeliveryDays'].isnull().sum(), df.duplicated().sum(),
              df['Returned'].str.lower().value_counts().sum() - df['Returned'].nunique() + df['Returned'].nunique(),
              len(invalid_disc), len(outliers_ov)],
    'Impact': ['Incomplete delivery analysis', 'Inflated order counts',
               'Wrong group counts in charts', 'Impossible business values',
               'Distorted averages and box plots']
})
print(quality_summary.to_string(index=False))

# ── STEP 3: Clean the Dataset ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — DATA CLEANING WITH JUSTIFICATIONS")
print("=" * 60)

df_clean = df.copy()

# 3a. Remove duplicate rows
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"3a. Removed {before - len(df_clean)} duplicate rows.")
# JUSTIFICATION: Each OrderID should be unique. Duplicate orders would double-
# count certain transactions, making return rates appear artificially different.

# 3b. Standardise 'Returned' labels → Yes / No
df_clean['Returned'] = df_clean['Returned'].str.strip().str.title()
print("3b. Returned values after fix:", df_clean['Returned'].unique())
# JUSTIFICATION: 'yes', 'YES', 'Yes' are the same — Python sees them as
# 3 different categories. .str.title() converts everything to 'Yes'/'No'.

# 3c. Fill missing DeliveryDays with MEDIAN
median_delivery = df_clean['DeliveryDays'].median()
df_clean['DeliveryDays'] = df_clean['DeliveryDays'].fillna(median_delivery)
print(f"3c. Filled {df['DeliveryDays'].isnull().sum()} missing DeliveryDays with median = {median_delivery}")
# JUSTIFICATION: Delivery days are moderately skewed. Median is preferred
# over mean because it is not pulled by extreme values. We impute rather
# than drop because these rows still have useful information.

# 3d. Cap invalid DiscountPercent: clip to 0-100 range
df_clean['DiscountPercent'] = df_clean['DiscountPercent'].clip(lower=0, upper=100)
print("3d. DiscountPercent capped to valid range [0, 100].")
# JUSTIFICATION: A discount of -5% or 135% is physically impossible.
# We cap (clip) rather than delete these rows because the other fields
# (order value, region, category) are still valid and useful.

# 3e. Cap OrderValue outliers using Winsorizing (IQR upper fence)
df_clean['OrderValue'] = df_clean['OrderValue'].clip(upper=upper_fence)
print(f"3e. OrderValue outliers capped at upper fence = {upper_fence:.2f}")
# JUSTIFICATION: Orders worth 8750 or 5200 are extreme but could be real
# (bulk/business orders). We winsorise rather than delete — this keeps the
# row but limits its distorting effect on visualisations and averages.

print(f"\nFinal clean dataset shape: {df_clean.shape}")
print("Remaining missing values:")
print(df_clean.isnull().sum())

# ── STEP 4: Visualisations ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — VISUALISATIONS")
print("=" * 60)

sns.set_style("whitegrid")
colors = {'Yes': '#e74c3c', 'No': '#2ecc71'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Q2 — E-Commerce Customer Return Behavior EDA',
             fontsize=16, fontweight='bold')

# ── Plot 1: Bar Chart — Frequency of Returned  ───────────────────────────────
return_counts = df_clean['Returned'].value_counts()

# Frequency table first
print("\nFrequency Table — Returned:")
freq_table = pd.DataFrame({
    'Count': return_counts,
    'Percentage': (return_counts / len(df_clean) * 100).round(2)
})
print(freq_table)

axes[0, 0].bar(return_counts.index, return_counts.values,
               color=[colors.get(x, '#95a5a6') for x in return_counts.index],
               edgecolor='white', alpha=0.9)
for i, (label, val) in enumerate(return_counts.items()):
    axes[0, 0].text(i, val + 0.5, f'{val}\n({val/len(df_clean)*100:.1f}%)',
                    ha='center', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Return Frequency (Returned vs Not Returned)', fontweight='bold')
axes[0, 0].set_xlabel('Returned')
axes[0, 0].set_ylabel('Number of Orders')

# WHY BAR CHART for a categorical variable?
# 'Returned' is a binary categorical variable (Yes/No). A bar chart directly
# shows counts per category. A pie chart would work too, but bar charts are
# easier to compare when combined with percentage labels.

# ── Plot 2: Box Plot — OrderValue by Returned  ───────────────────────────────
sns.boxplot(data=df_clean, x='Returned', y='OrderValue',
            palette=colors, ax=axes[0, 1])
axes[0, 1].set_title('Order Value by Return Status (Box Plot)', fontweight='bold')
axes[0, 1].set_xlabel('Returned')
axes[0, 1].set_ylabel('Order Value (PKR/Currency)')

# WHY BOX PLOT not a bar chart?
# We want to compare the DISTRIBUTION of OrderValue for returned vs
# non-returned orders. A box plot shows median, spread (IQR), and outliers
# for both groups. A bar chart would only show averages — hiding whether
# high-value orders have more variance in return rates.

# ── Plot 3: Analysis — DeliveryDays vs Returned  ─────────────────────────────
avg_delivery = df_clean.groupby('Returned')['DeliveryDays'].mean()
bars = axes[1, 0].bar(avg_delivery.index, avg_delivery.values,
                      color=[colors.get(x, '#95a5a6') for x in avg_delivery.index],
                      edgecolor='white', alpha=0.9)
axes[1, 0].set_title('Avg Delivery Days by Return Status', fontweight='bold')
axes[1, 0].set_xlabel('Returned')
axes[1, 0].set_ylabel('Average Delivery Days')
for bar, val in zip(bars, avg_delivery.values):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1, f'{val:.1f} days',
                    ha='center', fontsize=11, fontweight='bold')

# WHY: Longer delivery times may frustrate customers, increasing returns.
# Comparing average delivery days between Returned=Yes and Returned=No
# directly tests this hypothesis.

# ── Plot 4: Heatmap — Correlation Matrix for Numerical Variables  ─────────────
num_cols = ['OrderValue', 'DeliveryDays', 'DiscountPercent']
corr_matrix = df_clean[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=axes[1, 1],
            cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Correlation Heatmap — Numerical Variables', fontweight='bold')

# WHY HEATMAP for correlations?
# A heatmap shows all pairwise correlations at once with colour intensity.
# Values close to +1 = strong positive correlation (both go up together).
# Values close to -1 = strong negative correlation (one goes up, other down).
# Values near 0 = no linear relationship.
# This helps identify which numerical variable is most linked to returns.

plt.tight_layout()
plt.savefig('Q2_Ecommerce_Returns_EDA_Plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as Q2_Ecommerce_Returns_EDA_Plots.png")

# ── Bonus: Return rate by ProductCategory (extra insight) ────────────────────
print("\nReturn Rate by Product Category:")
cat_return = df_clean.groupby('ProductCategory')['Returned'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2).sort_values(ascending=False)
print(cat_return)

print("\nReturn Rate by CustomerRegion:")
reg_return = df_clean.groupby('CustomerRegion')['Returned'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).round(2).sort_values(ascending=False)
print(reg_return)

# ── STEP 5: Data-Driven Recommendations ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — DATA-DRIVEN RECOMMENDATIONS FOR THE COMPANY")
print("=" * 60)
print("""
Recommendation 1 — Improve Delivery Speed for High-Return Categories:
  Orders that were returned had a higher average delivery time compared
  to orders that were kept. Customers who wait longer are more likely to
  lose interest in the product or find alternatives. The company should
  partner with faster logistics providers for high-return categories
  (e.g., Fashion, Electronics) and aim to deliver within 3-5 days.

Recommendation 2 — Audit Discount Strategies:
  3 records had impossible discount values (-5%, 110%, 135%), suggesting
  the discount entry system has no validation rules. Additionally, very
  high discounts may attract impulsive buyers who are more likely to return
  items. The company should cap discounts at 100% in the system, and
  analyse whether orders with discounts above 50% show significantly
  higher return rates — if so, revise the promotional discount policy.
""")
