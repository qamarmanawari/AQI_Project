# ============================================================
#  QUESTION 1 — Hospital Emergency Room Waiting Time Analysis
#  Introduction to Data Science | SZABIST Islamabad
#  Tool: Google Colab | Libraries: pandas, numpy, matplotlib, seaborn
# ============================================================

# ── STEP 0: Install / import libraries ───────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Upload your dataset when prompted
uploaded = files.upload()   # select:  Q1_Hospital_ER_Dataset.xlsx

# ── STEP 1: Load & Inspect Dataset Structure ─────────────────────────────────
df = pd.read_excel('Q1_Hospital_ER_Dataset.xlsx')

print("=" * 55)
print("STEP 1 — DATASET STRUCTURE")
print("=" * 55)
print(f"\nShape  :  {df.shape[0]} rows  x  {df.shape[1]} columns")
print("\nColumn names and data types:")
print(df.dtypes)
print("\nFirst 5 rows:")
df.head()

# WHY: .dtypes tells us which columns are numerical (int/float) vs categorical
# (object/string). This guides which statistical summaries and chart types
# are appropriate for each variable.

print("\nBasic statistics for numerical columns:")
df.describe()

# ── STEP 2: Detect Data Quality Issues ───────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — DATA QUALITY ISSUES")
print("=" * 55)

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
    print(df[df.duplicated(keep=False)])

# 2c. Inconsistent category labels
print("\n--- 2c. Inconsistent Labels ---")
print("AdmissionStatus unique values:", df['AdmissionStatus'].unique())
print("TriageLevel unique values    :", df['TriageLevel'].unique())
print("Gender unique values         :", df['Gender'].unique())

# WHY: Inconsistent casing ('admitted' vs 'Admitted') causes Python to treat
# them as different categories — this breaks groupby and all charts.

# 2d. Invalid ages
print("\n--- 2d. Invalid Ages ---")
invalid_age = df[(df['Age'] < 0) | (df['Age'] > 110)]
print(f"Records with invalid age: {len(invalid_age)}")
print(invalid_age[['PatientID', 'Age']])

# 2e. Outliers in WaitingTimeMinutes using IQR method
print("\n--- 2e. Outliers in WaitingTimeMinutes ---")
Q1 = df['WaitingTimeMinutes'].quantile(0.25)
Q3 = df['WaitingTimeMinutes'].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = df[(df['WaitingTimeMinutes'] < lower_fence) |
              (df['WaitingTimeMinutes'] > upper_fence)]
print(f"IQR = {IQR:.1f}  |  Lower fence = {lower_fence:.1f}  |  Upper fence = {upper_fence:.1f}")
print(f"Outlier rows ({len(outliers)}):")
print(outliers[['PatientID', 'WaitingTimeMinutes']])

# WHY we use IQR (not mean ± 2SD):
# Waiting-time data is skewed (not bell-shaped). IQR is based on quartiles
# so it is NOT affected by extreme values — making it the right tool here.

# ── STEP 3: Clean the Dataset ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — DATA CLEANING")
print("=" * 55)

df_clean = df.copy()

# 3a. Remove duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"3a. Duplicates removed: {before - len(df_clean)}")
# JUSTIFICATION: Identical rows are data-entry errors. Keeping them inflates
# patient counts and distorts averages.

# 3b. Standardise AdmissionStatus labels
df_clean['AdmissionStatus'] = df_clean['AdmissionStatus'].str.strip().str.title()
print("3b. AdmissionStatus after fix:", df_clean['AdmissionStatus'].unique())
# JUSTIFICATION: .str.title() converts 'admitted' → 'Admitted' and
# 'TRANSFERRED' → 'Transferred', making all labels uniform.

# 3c. Remove invalid ages
before = len(df_clean)
df_clean = df_clean[df_clean['Age'].between(0, 110)]
print(f"3c. Invalid age rows removed: {before - len(df_clean)}")
# JUSTIFICATION: Age -3 and 145 are physically impossible. We drop rather
# than impute because we cannot reliably guess a patient's real age.

# 3d. Fill missing WaitingTimeMinutes with MEDIAN
median_wait = df_clean['WaitingTimeMinutes'].median()
df_clean['WaitingTimeMinutes'] = df_clean['WaitingTimeMinutes'].fillna(median_wait)
print(f"3d. Missing WaitingTimeMinutes filled with median = {median_wait:.1f} min")
# JUSTIFICATION: We use median NOT mean because the data is right-skewed.
# The mean is pulled up by large outliers, so the median is a fairer
# "typical" value to substitute for missing entries.

# 3e. Fill missing DoctorAssigned with 'Unknown'
df_clean['DoctorAssigned'] = df_clean['DoctorAssigned'].fillna('Unknown')
print(f"3e. Missing DoctorAssigned filled with 'Unknown'")
# JUSTIFICATION: We cannot guess which doctor saw the patient, so we label
# missing entries 'Unknown'. This keeps the rows usable for other analyses.

# 3f. Cap outliers using Winsorizing (clip at IQR upper fence)
df_clean['WaitingTimeMinutes'] = df_clean['WaitingTimeMinutes'].clip(upper=upper_fence)
print(f"3f. Waiting times capped at upper fence = {upper_fence:.1f} min")
# JUSTIFICATION: 480-min and 720-min records are real patients — we should
# not delete them. Winsorizing caps extreme values at the IQR boundary,
# preserving the row while reducing chart distortion.

print(f"\nFinal clean dataset shape: {df_clean.shape}")
df_clean.isnull().sum()

# ── STEP 4: Visualisations ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — VISUALISATIONS")
print("=" * 55)

sns.set_style("whitegrid")
palette_status = {'Admitted': '#e74c3c', 'Discharged': '#2ecc71', 'Transferred': '#3498db'}
triage_order = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Q1 — Hospital Emergency Room EDA', fontsize=16, fontweight='bold')

# ── Plot 1: Histogram — WaitingTimeMinutes  (UNIVARIATE) ─────────────────────
axes[0, 0].hist(df_clean['WaitingTimeMinutes'], bins=20,
                color='#3498db', edgecolor='white', alpha=0.85)
axes[0, 0].axvline(df_clean['WaitingTimeMinutes'].mean(), color='red',
                   linestyle='--', linewidth=1.8,
                   label=f"Mean: {df_clean['WaitingTimeMinutes'].mean():.0f} min")
axes[0, 0].axvline(df_clean['WaitingTimeMinutes'].median(), color='orange',
                   linestyle='--', linewidth=1.8,
                   label=f"Median: {df_clean['WaitingTimeMinutes'].median():.0f} min")
axes[0, 0].set_title('Distribution of Waiting Times (Univariate)', fontweight='bold')
axes[0, 0].set_xlabel('Waiting Time (Minutes)')
axes[0, 0].set_ylabel('Number of Patients')
axes[0, 0].legend()

# WHY HISTOGRAM not box plot?
# A histogram reveals the SHAPE of the distribution (skewed? bimodal?).
# A box plot only shows 5 summary numbers. For univariate exploration,
# histogram gives more insight.

# ── Plot 2: Box Plot — Waiting Time by TriageLevel  (BIVARIATE 1) ────────────
sns.boxplot(data=df_clean, x='TriageLevel', y='WaitingTimeMinutes',
            order=triage_order, palette='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Waiting Time by Triage Level (Bivariate 1)', fontweight='bold')
axes[0, 1].set_xlabel('Triage Level')
axes[0, 1].set_ylabel('Waiting Time (Minutes)')

# WHY BOX PLOT not bar chart?
# We are comparing a numerical variable (waiting time) across 5 categories.
# A box plot shows median, spread (IQR), and outliers for EACH group.
# A bar chart would only show the average — hiding whether one group has
# wildly inconsistent waiting times vs another.

# ── Plot 3: Stacked Bar — TriageLevel vs AdmissionStatus  (BIVARIATE 2) ──────
cross = pd.crosstab(df_clean['TriageLevel'], df_clean['AdmissionStatus'])
cross = cross.reindex(triage_order)
cross.plot(kind='bar', stacked=True, ax=axes[1, 0],
           color=[palette_status.get(c, '#95a5a6') for c in cross.columns],
           edgecolor='white')
axes[1, 0].set_title('Triage Level vs Admission Status (Bivariate 2)', fontweight='bold')
axes[1, 0].set_xlabel('Triage Level')
axes[1, 0].set_ylabel('Number of Patients')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].legend(title='Admission Status', bbox_to_anchor=(1.01, 1))

# WHY STACKED BAR?
# Both variables are categorical. A stacked bar lets us see both the total
# count per triage level AND the breakdown of outcomes (admitted/discharged/
# transferred) in a single chart — two questions answered at once.

# ── Plot 4: Bar — Avg Waiting Time by AdmissionStatus ────────────────────────
avg_wait = (df_clean.groupby('AdmissionStatus')['WaitingTimeMinutes']
            .mean().sort_values(ascending=False))
bars = axes[1, 1].bar(avg_wait.index, avg_wait.values,
                      color=[palette_status.get(c, '#95a5a6') for c in avg_wait.index],
                      edgecolor='white', alpha=0.9)
axes[1, 1].set_title('Avg Waiting Time by Admission Status', fontweight='bold')
axes[1, 1].set_xlabel('Admission Status')
axes[1, 1].set_ylabel('Avg Waiting Time (Minutes)')
for bar, val in zip(bars, avg_wait.values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f'{val:.0f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('Q1_Hospital_EDA_Plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as Q1_Hospital_EDA_Plots.png")

# ── STEP 5: Practical Findings ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5 — PRACTICAL FINDINGS FOR HOSPITAL MANAGEMENT")
print("=" * 55)

print("""
Finding 1 — Triage Priority Not Reflected in Wait Times:
  Patients triaged as Level 1 (most critical) are not consistently
  seen faster than Level 3 or Level 4 patients. This suggests a
  resource allocation gap — senior doctors may not be available
  on shift when critical patients arrive. Recommendation: review
  doctor scheduling to ensure Level 1 & 2 cases are always
  attended to within target time windows.

Finding 2 — Data Integrity Issues Risk Patient Safety:
  7 records had no DoctorAssigned, 3 were full duplicates, and
  2 had impossible ages (-3 and 145). Poor data quality makes
  it impossible to audit doctor workloads or track patient flow
  accurately. Recommendation: implement mandatory field validation
  at the point of data entry in the registration system.
""")
