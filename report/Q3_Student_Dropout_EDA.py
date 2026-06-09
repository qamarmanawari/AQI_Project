# ============================================================
#  QUESTION 3 — University Student Dropout Risk Study
#  Introduction to Data Science | SZABIST Islamabad
#  Tool: Google Colab | Libraries: pandas, numpy, matplotlib, seaborn
# ============================================================

# ── STEP 0: Import libraries & upload file ───────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

uploaded = files.upload()   # select:  Q3_Student_Dropout_Dataset.xlsx

# ── STEP 1: Dataset Structure — Numerical, Categorical & ID Variables ─────────
df = pd.read_excel('Q3_Student_Dropout_Dataset.xlsx')

print("=" * 60)
print("STEP 1 — DATASET STRUCTURE & VARIABLE IDENTIFICATION")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows  x  {df.shape[1]} columns")
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("""
Variable Classification:
  ID Variable          : StudentID
  Numerical (cont.)    : AttendancePercent, LMSLoginsPerWeek,
                         AssignmentAverage, FamilyIncome
  Categorical (nominal): Department, ScholarshipStatus
  Categorical (ordinal): DropoutRisk  (Low < Medium < High)
""")

# WHY classify variables?
# ID variables should NEVER be used in analysis (each value is unique — no
# pattern exists). Numerical variables → statistics + scatter/box plots.
# Ordinal categorical → bar charts + cross-tabs with meaningful order.

print("\nDescriptive statistics for numerical columns:")
print(df.describe())

# ── STEP 2: Detect Data Quality Issues with Evidence ─────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — DATA QUALITY ISSUES WITH EVIDENCE")
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
    print(df[df.duplicated(keep=False)][['StudentID', 'Department', 'DropoutRisk']])

# 2c. Invalid AttendancePercent (> 100 is impossible)
print("\n--- 2c. Invalid AttendancePercent (> 100%) ---")
invalid_att = df[df['AttendancePercent'] > 100]
print(f"Records with attendance > 100%: {len(invalid_att)}")
print(invalid_att[['StudentID', 'AttendancePercent']])

# WHY is attendance > 100 an issue?
# Attendance is a percentage — it physically cannot exceed 100%.
# Values like 135.5% are data entry errors (possibly raw counts entered
# instead of percentages). They must be fixed before analysis.

# 2d. Inconsistent labels — ScholarshipStatus and DropoutRisk
print("\n--- 2d. Inconsistent Category Labels ---")
print("ScholarshipStatus unique:", df['ScholarshipStatus'].unique())
print("DropoutRisk unique      :", df['DropoutRisk'].unique())

# 'yes' and 'Yes' are the same — Python treats them as 2 different groups.
# 'high', 'High', 'HIGH' same problem in DropoutRisk.

# 2e. Outliers in FamilyIncome using IQR
print("\n--- 2e. Outliers in FamilyIncome ---")
Q1_inc = df['FamilyIncome'].quantile(0.25)
Q3_inc = df['FamilyIncome'].quantile(0.75)
IQR_inc = Q3_inc - Q1_inc
upper_inc = Q3_inc + 1.5 * IQR_inc
outliers_inc = df[df['FamilyIncome'] > upper_inc]
print(f"IQR = {IQR_inc:.0f}  |  Upper fence = {upper_inc:.0f}")
print(f"Outlier records ({len(outliers_inc)}):")
print(outliers_inc[['StudentID', 'FamilyIncome']])

# WHY IQR for FamilyIncome outlier detection?
# Income data is almost always right-skewed — a few very wealthy families
# pull the mean far above the typical value. IQR ignores the extremes and
# focuses on the middle 50% of data, making it the right tool here.

# Summary evidence table
print("\n--- SUMMARY OF ALL DATA QUALITY ISSUES ---")
quality_df = pd.DataFrame({
    'Issue': [
        'Missing AttendancePercent',
        'Missing LMSLoginsPerWeek',
        'Duplicate Rows',
        'AttendancePercent > 100',
        'Inconsistent ScholarshipStatus labels',
        'Inconsistent DropoutRisk labels',
        'FamilyIncome extreme outliers'
    ],
    'Count': [
        df['AttendancePercent'].isnull().sum(),
        df['LMSLoginsPerWeek'].isnull().sum(),
        df.duplicated().sum(),
        len(invalid_att),
        2,   # 'yes' vs 'Yes'
        2,   # 'high'/'LOW' vs title case
        len(outliers_inc)
    ],
    'Impact': [
        'Distribution plot incomplete',
        'Scatter plot distorted',
        'Inflated student counts',
        'Skews attendance distribution upward',
        'Cross-tab splits same group in two',
        'Cross-tab splits same group in two',
        'Distorts income-based analysis'
    ]
})
print(quality_df.to_string(index=False))

# ── STEP 3: Clean the Dataset ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — DATA CLEANING WITH JUSTIFICATIONS")
print("=" * 60)

df_clean = df.copy()

# 3a. Remove duplicate rows
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"3a. Removed {before - len(df_clean)} duplicate rows.")
# JUSTIFICATION: Each StudentID should appear once. Duplicate records inflate
# enrollment counts and distort every grouped analysis.

# 3b. Fix inconsistent ScholarshipStatus labels
df_clean['ScholarshipStatus'] = df_clean['ScholarshipStatus'].str.strip().str.title()
print("3b. ScholarshipStatus after fix:", df_clean['ScholarshipStatus'].unique())
# JUSTIFICATION: 'yes' and 'Yes' are the same scholarship status. .str.title()
# forces all values to consistent Title Case so groupby works correctly.

# 3c. Fix inconsistent DropoutRisk labels
df_clean['DropoutRisk'] = df_clean['DropoutRisk'].str.strip().str.title()
print("3c. DropoutRisk after fix:", df_clean['DropoutRisk'].unique())
# JUSTIFICATION: 'high', 'HIGH', 'High' must all become 'High' — otherwise
# the ordered categories (Low, Medium, High) get split into 5+ groups.

# 3d. Cap invalid AttendancePercent to 100
invalid_before = (df_clean['AttendancePercent'] > 100).sum()
df_clean['AttendancePercent'] = df_clean['AttendancePercent'].clip(upper=100)
print(f"3d. Capped {invalid_before} attendance values exceeding 100% to 100.")
# JUSTIFICATION: Attendance cannot logically exceed 100%. We cap at 100
# rather than deleting the row because all other data for these students
# (grades, income, dropout risk) is still valid and informative.

# 3e. Fill missing AttendancePercent with median
median_att = df_clean['AttendancePercent'].median()
df_clean['AttendancePercent'] = df_clean['AttendancePercent'].fillna(median_att)
print(f"3e. Filled missing AttendancePercent with median = {median_att:.1f}%")
# JUSTIFICATION: Attendance is slightly skewed so median is more
# representative than mean. One missing value — imputation is safe.

# 3f. Fill missing LMSLoginsPerWeek with median
median_lms = df_clean['LMSLoginsPerWeek'].median()
df_clean['LMSLoginsPerWeek'] = df_clean['LMSLoginsPerWeek'].fillna(median_lms)
print(f"3f. Filled missing LMSLoginsPerWeek with median = {median_lms:.1f}")
# JUSTIFICATION: LMS logins are right-skewed. Median avoids being pulled by
# students with unusually high login counts. Only 3 rows affected.

# 3g. Winsorise FamilyIncome outliers
df_clean['FamilyIncome'] = df_clean['FamilyIncome'].clip(upper=upper_inc)
print(f"3g. FamilyIncome outliers capped at {upper_inc:.0f}.")
# JUSTIFICATION: Incomes of 4,300,000 are real but extreme. Winsorizing keeps
# the student in the dataset while limiting distortion in income-based charts.

print(f"\nFinal clean dataset shape: {df_clean.shape}")
print("Remaining missing values:")
print(df_clean.isnull().sum())

# ── STEP 4: Visualisations ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — VISUALISATIONS")
print("=" * 60)

sns.set_style("whitegrid")
risk_order   = ['Low', 'Medium', 'High']
risk_palette = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Q3 — University Student Dropout Risk EDA',
             fontsize=16, fontweight='bold')

# ── Plot 1: Distribution — AttendancePercent  (required) ─────────────────────
axes[0, 0].hist(df_clean['AttendancePercent'], bins=18,
                color='#3498db', edgecolor='white', alpha=0.85)
axes[0, 0].axvline(df_clean['AttendancePercent'].mean(), color='red',
                   linestyle='--', linewidth=1.8,
                   label=f"Mean : {df_clean['AttendancePercent'].mean():.1f}%")
axes[0, 0].axvline(df_clean['AttendancePercent'].median(), color='orange',
                   linestyle='--', linewidth=1.8,
                   label=f"Median: {df_clean['AttendancePercent'].median():.1f}%")
axes[0, 0].set_title('Distribution of Attendance Percent', fontweight='bold')
axes[0, 0].set_xlabel('Attendance (%)')
axes[0, 0].set_ylabel('Number of Students')
axes[0, 0].legend()

# WHY HISTOGRAM (distribution plot)?
# The assignment asks for a "distribution plot" — a histogram is the standard
# way to visualise how a continuous variable is spread. It shows whether
# attendance is clustered (most students at ~75%) or spread out.
# Adding mean and median lines shows whether the data is skewed.

# ── Plot 2: Scatter — LMSLoginsPerWeek vs AssignmentAverage  (required) ───────
scatter_colors = [risk_palette.get(r, '#95a5a6') for r in df_clean['DropoutRisk']]
axes[0, 1].scatter(df_clean['LMSLoginsPerWeek'], df_clean['AssignmentAverage'],
                   c=scatter_colors, alpha=0.75, edgecolors='white', s=60)
# Add trend line
m, b = np.polyfit(df_clean['LMSLoginsPerWeek'], df_clean['AssignmentAverage'], 1)
x_line = np.linspace(df_clean['LMSLoginsPerWeek'].min(),
                     df_clean['LMSLoginsPerWeek'].max(), 100)
axes[0, 1].plot(x_line, m * x_line + b, color='black',
                linewidth=1.5, linestyle='--', label=f'Trend (slope={m:.2f})')
# Legend for risk colours
from matplotlib.patches import Patch
legend_patches = [Patch(color=risk_palette[r], label=r) for r in risk_order]
axes[0, 1].legend(handles=legend_patches + [axes[0, 1].lines[0]], title='Dropout Risk')
axes[0, 1].set_title('LMS Logins vs Assignment Average\n(coloured by Dropout Risk)',
                     fontweight='bold')
axes[0, 1].set_xlabel('LMS Logins Per Week')
axes[0, 1].set_ylabel('Assignment Average (%)')

# WHY SCATTER PLOT?
# A scatter plot is the correct chart to show the RELATIONSHIP between two
# continuous numerical variables. Each dot = one student. If dots trend upward
# left-to-right, more logins → better grades. The trend line (polyfit) makes
# the direction of relationship clear even if the dots are spread out.
# Colouring by DropoutRisk adds a third dimension of insight.

# ── Plot 3: Grouped Box Plot — AssignmentAverage by DropoutRisk  (required) ──
sns.boxplot(data=df_clean, x='DropoutRisk', y='AssignmentAverage',
            order=risk_order, palette=risk_palette, ax=axes[1, 0])
axes[1, 0].set_title('Assignment Average by Dropout Risk (Grouped Box Plot)',
                     fontweight='bold')
axes[1, 0].set_xlabel('Dropout Risk Level')
axes[1, 0].set_ylabel('Assignment Average (%)')

# WHY GROUPED BOX PLOT not a bar chart?
# We are comparing a continuous variable (AssignmentAverage) across 3 ordered
# groups (Low/Medium/High risk). A grouped box plot shows:
#   - The median grade for each risk group
#   - The spread (IQR) — are High-risk students consistently low or just
#     variable?
#   - Any outliers within each group
# A bar chart would only show averages — hiding the variance within groups.

# ── Plot 4: Cross-tabulation — ScholarshipStatus vs DropoutRisk  (required) ──
cross = pd.crosstab(df_clean['ScholarshipStatus'], df_clean['DropoutRisk'])
cross = cross[risk_order]   # enforce Low → Medium → High order
cross.plot(kind='bar', ax=axes[1, 1],
           color=[risk_palette[r] for r in risk_order],
           edgecolor='white', alpha=0.9)
axes[1, 1].set_title('Scholarship Status vs Dropout Risk (Cross-Tabulation)',
                     fontweight='bold')
axes[1, 1].set_xlabel('Scholarship Status')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].tick_params(axis='x', rotation=20)
axes[1, 1].legend(title='Dropout Risk', bbox_to_anchor=(1.01, 1))

# WHY GROUPED BAR CHART for cross-tabulation?
# Both variables are categorical. A grouped bar chart is a visual
# cross-tabulation — it shows how dropout risk is distributed within each
# scholarship group. If 'No Scholarship' has a much larger 'High' bar, that
# is direct evidence that financial support reduces dropout risk.

plt.tight_layout()
plt.savefig('Q3_Student_Dropout_EDA_Plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as Q3_Student_Dropout_EDA_Plots.png")

# ── Print the actual cross-tabulation table ───────────────────────────────────
print("\nCross-Tabulation Table — ScholarshipStatus vs DropoutRisk:")
print(cross)
print("\nRow Percentages (% within each Scholarship group):")
print(cross.div(cross.sum(axis=1), axis=0).mul(100).round(1))

# ── STEP 5: Plain-Language Findings for Intervention Plan ────────────────────
print("\n" + "=" * 60)
print("STEP 5 — FINDINGS FOR DROPOUT INTERVENTION PLAN")
print("=" * 60)
print("""
Finding 1 — Low Engagement on LMS Is a Strong Dropout Signal:
  Students with fewer LMS logins per week tend to score lower on
  assignments AND appear more frequently in the High dropout-risk
  group (visible in the scatter plot — red dots cluster at low login
  counts). This suggests that disengagement from the learning platform
  is an early warning sign. The university should set up an automated
  alert: any student logging in fewer than 5 times per week for two
  consecutive weeks should be flagged and contacted by their academic
  advisor for a welfare check.

Finding 2 — Students Without Scholarships Face Significantly Higher Risk:
  The cross-tabulation shows that students with 'No' scholarship status
  make up the largest proportion of High dropout-risk cases. Financial
  stress is a known driver of dropout. The university should expand its
  partial scholarship and emergency financial aid programs, specifically
  targeting students already showing academic warning signs (low
  attendance + low LMS engagement), before they reach the point of
  dropping out.
""")
