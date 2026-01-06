#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import math
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# # Data Loading

# In[2]:


# Load loan-level MBS data
df = pd.read_csv(r"C:\Users\sbalodi\Desktop\final_cph_data.csv")
df1 = pd.DataFrame(df)
df1.head()


# # Exploratory Data Analysis and Preprocessing

# In[3]:


df1.shape


# In[4]:


single_loan_df = df1[df1['Loan_ID'] == 'F14Q10000011']
single_loan_df


# In[5]:


df1.describe()


# In[6]:


df1.isna().sum()


# # Imputation of missing values of ELTV using LOCF
# The most appropriate and robust method for imputing missing values in the time-varying Estimated Loan-to-Value (ELTV) column is Last Observation Carried Forward (LOCF), applied within each loan's time series. This assumes that the property's equity position does not change significantly between reporting periods, which is a safer assumption than using a portfolio-wide average (like the median).

# In[7]:


# 1. Ensure the data is sorted by loan ID and date (CRITICAL for LOCF)
df2 = df1.sort_values(by=['Loan_ID', 'Monthly Reporting Period'])

# 2. Group the data by Loan ID and fill missing values forward (ffill) within that group
df2['Estimated Loan-to-Value (ELTV)'] = df2.groupby('Loan_ID')['Estimated Loan-to-Value (ELTV)'].fillna(method='ffill')

# 3. Handle any initial NaNs (where no previous observation existed for that loan)
# These are loans with missing ELTV right from Month 1. We fill these with the median CLTV/LTV.
# We will use the portfolio-wide median LTV/CLTV as a reasonable starting anchor.
median_cltv = df2['Original Combined Loan-to-Value (CLTV)'].median()
df2['Estimated Loan-to-Value (ELTV)'].fillna(median_cltv, inplace=True)

print("ELTV missing values successfully imputed using LOCF and anchored by median CLTV.")


# In[8]:


df2.isna().sum()


# # Convert date columns to datetime objects

# In[9]:


df2.info()


# In[10]:


date_cols = ['Monthly Reporting Period', 'First Payment Date', 'Maturity Date']
for col in date_cols:
    df2[col] = pd.to_datetime(
        df2[col], 
        # Remove the 'format' argument and use infer_datetime_format=True
        infer_datetime_format=True, 
        errors='coerce'
    )

df2.info()


# # Encoding Categorical Variables

# Occupancy Status

# In[11]:


print(df2['Occupancy Status'].value_counts()) 
# do One Hot Encoding - logic: cardinal categories
# P-pending. I-investment. S-second home


# In[12]:


# 1. Create ALL dummy variables (including P, I, and S)
df2_temp = pd.get_dummies(
    df2, 
    columns=['Occupancy Status'], 
    prefix='Occ'
)

# 2. Identify the column for the safest category ('P')
# Assuming the resulting column is named 'Occ_P'
BASELINE_COLUMN = 'Occ_P'

# 3. Drop the baseline column and assign the result back to df3
if BASELINE_COLUMN in df2_temp.columns:
    df2 = df2_temp.drop(BASELINE_COLUMN, axis=1)
    
else:
    print(f"Error: Could not find the expected baseline column '{BASELINE_COLUMN}'. Check original categories.")

df2['Occ_I'] = df2['Occ_I'].astype(int)
df2['Occ_S'] = df2['Occ_S'].astype(int)

print("Dummy variables successfully converted to 1s and 0s.")
print(df2[['Occ_I', 'Occ_S']].head())


# In[13]:


df2.info()


# # Numerical Variables

# In[14]:


continuous_cols = [
    'Credit Score',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Original UPB',
    'Current Actual UPB',
    'Current Interest Rate',
    'Estimated Loan-to-Value (ELTV)'
]

# Set up the plotting environment
sns.set_style("whitegrid")
# Create a 4x2 grid for the 7 variables
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))
axes = axes.flatten() 

# Iterate over the columns and plot the distribution (histogram with KDE)
for i, col in enumerate(continuous_cols):
    ax = axes[i]
    # Use seaborn to plot the distribution with a kernel density estimate
    sns.histplot(df2[col].dropna(), kde=True, ax=ax, bins=50)
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
# Hide the unused subplot
fig.delaxes(axes[7])

plt.suptitle('Distribution of Continuous Covariates for CPH Model', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('continuous_covariate_distributions.png')
print("continuous_covariate_distributions.png")


# In[15]:


df3 = df2.copy()


# In[16]:


# Apply log1p to handle skewness in balance columns
df3['Original_UPB_Log'] = np.log1p(df3['Original UPB'])
df3['Current_Actual_UPB_Log'] = np.log1p(df3['Current Actual UPB'])


# # Scaling numerical variables

# In[17]:


# Define the final list of columns to be scaled (including log-transformed UPB)
cols_to_scale = [
    'Credit Score',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Current Interest Rate',
    'Estimated Loan-to-Value (ELTV)',
    'Original_UPB_Log', # Use the log-transformed column
    'Current_Actual_UPB_Log' # Use the log-transformed column
]

scaler = StandardScaler()

# Fit and transform the data
df3[cols_to_scale] = scaler.fit_transform(df3[cols_to_scale])

print("All continuous covariates are now ready for the CPH model.")


# In[18]:


df3.head()


# Current Loan Deliquency Status

# In[19]:


print(df3['Current Loan Delinquency Status'].value_counts()) 


# In[20]:


def map_delinquency_to_ordinal_tv_cov(status):
    """
    Converts delinquency status to an ordinal numeric value for use as a CPH predictor.
    The value increases with risk severity.
    """
    status_str = str(status).strip()
    
    # 1. Handle Termination Codes (Severe Distress)
    if status_str in ['RA', 'RF', 'RE', '999']: # REO Acquisition, Foreclosure, Unknown
        return 3 # Treat as maximum distress level for TVC
    
    # 2. Handle Numeric DPD Codes
    try:
        days_late_code = int(status_str)
        
        if days_late_code >= 3:
            return 3 # Cap at 90+ DPD to represent the failure state (3 = 90 DPD, 4 = 120 DPD, etc.)
        else:
            return days_late_code # Returns 0, 1, or 2
            
    except ValueError:
        # Catch other non-numeric non-delinquent codes (e.g., Repayment Plans, etc.)
        return 0 

# Apply the mapping to the column
df4 = df3.copy()
df4['Delinq_Ordinal_TVC'] = df3['Current Loan Delinquency Status'].apply(map_delinquency_to_ordinal_tv_cov)

print("Created 'Delinq_Ordinal_TVC' column for use as a time-varying covariate.")


# In[21]:


df4.info()


# In[22]:


# Survival Target Aggregation (T and Delta) 

# Group the monthly history to find the final status and duration for each loan.
survival_df = df4.groupby('Loan_ID').agg(
    
    # --- SURVIVAL MARKERS ---
    Duration_T=('Loan Age', 'last'),
    Event_Delta=('Delinq_Ordinal_TVC', lambda x: 1 if x.max() >= 3 else 0),
    
    # --- TIME-FIXED COVARIATES (Use 'first' record, as they shouldn't change) ---
    FICO_Score=('Credit Score', 'first'),
    Original_CLTV=('Original Combined Loan-to-Value (CLTV)', 'first'),
    Original_DTI=('Original Debt-to-Income (DTI) Ratio', 'first'),
    Original_UPB=('Original UPB', 'first'),
    Original_Term=('Original Loan Term', 'first'),
    Occ_I=('Occ_I', 'first'), # Binary dummies are fixed
    Occ_S=('Occ_S', 'first'), # Binary dummies are fixed
    
    # --- TIME-VARYING (TVC) / LAST-OBSERVED COVARIATES (Use 'last') ---
    # Captures the final risk state before termination or study end.
    Delinq_Status_Last=('Delinq_Ordinal_TVC', 'last'),
    ELTV_Last=('Estimated Loan-to-Value (ELTV)', 'last'),
    Current_Rate_Last=('Current Interest Rate', 'last'),
    Current_UPB_Last=('Current Actual UPB', 'last'),
    
).reset_index()


# --- 2. Verification ---
print("Final survival dataset successfully structured.")
print(f"Total Observations: {len(survival_df):,}")
print("--- Sample of Final Survival Data ---")
print(survival_df[[
    'Loan_ID', 'Duration_T', 'Event_Delta', 'FICO_Score', 'ELTV_Last', 'Current_UPB_Last'
]].head())


# In[23]:


survival_df.info()


# # Multicollinearity check

# In[24]:


covariates = ['Current Loan Delinquency Status', 'Loan Age', 'Estimated Loan-to-Value (ELTV)','Current Interest Rate','Credit Score','Original Debt-to-Income (DTI) Ratio','Occupancy Status','Original Combined Loan-to-Value (CLTV)']


# In[25]:


# --- 1. Define ALL Predictor Covariates ---
covariates0 = [
    # Time-Fixed (Origination) Covariates
    'FICO_Score',
    'Original_CLTV',
    'Original_DTI',
    'Original_UPB',
    'Original_Term',
    
    # Binary/Categorical Covariates (Encoded)
    'Occ_I', # Investment Property
    'Occ_S', # Second Home
    
    # Time-Varying (Last Observed) Covariates
    'Delinq_Status_Last',    # Ordinal delinquency status
    'ELTV_Last',             # Estimated LTV
    'Current_Rate_Last',     # Current Interest Rate
    'Current_UPB_Last'       # Current Actual UPB
]

# --- 2. Prepare Data for VIF Check ---
# VIF is sensitive to NaNs, though your data is largely clean after imputation/aggregation.
# VIF requires all input columns to be float or integer (which they are after aggregation).
X0 = survival_df[covariates0].copy().dropna()

# --- 3. Check VIF for each covariate ---
vif_data0 = pd.DataFrame()
vif_data0["Feature"] = X0.columns
vif_data0["VIF"] = [
    variance_inflation_factor(X0.values, i) 
    for i in range(len(X0.columns))
]

print("--- Multicollinearity Check (VIF) on Predictors ---")
print("NOTE: VIF is sensitive to high correlation. Values > 5 or 10 may indicate issues.")
print(vif_data0.sort_values(by="VIF", ascending=False))


# In[26]:


cormat = X0.corr()

# --- 2. Plot the Heatmap (Correction 2: Use sns.heatmap instead of sns.plot) ---
plt.figure(figsize=(12, 10))
sns.heatmap(
    cormat,
    annot=True,              # Display the correlation values on the heatmap
    fmt=".2f",               # Format the numbers to two decimal places
    cmap="coolwarm",         # A good diverging color map for correlation
    linewidths=0.5,          # Lines between cells
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Pairwise Correlation Matrix of CPH Predictors', fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig('predictor_correlation_heatmap.png')
print("predictor_correlation_heatmap.png")


# In[27]:


cormat


# In[28]:


survival_df.head()


# # Sacling Continuous Columns in Surviva DF

# In[29]:


# Create a copy of the final aggregated DataFrame to work with
survival_df1 = survival_df.copy()

# --- 1. Apply Log Transformation to UPB (Mandatory for Skewness) ---
# This step addresses the structural skewness of loan balance.
survival_df1['Current_UPB_Log_ZScore'] = np.log1p(survival_df1['Current_UPB_Last'])

# --- 2. Define the full list of columns to be Z-Scored ---
SCALING_COLS = [
    'FICO_Score', 'Original_DTI', 'Original_CLTV', 'Original_Term',
    'Current_Rate_Last', 'ELTV_Last', 
    'Current_UPB_Log_ZScore'
]

# --- 3. Apply Standard Scaling (Z-Score) ---
scaler = StandardScaler()

# Fit and transform the continuous columns in place
survival_df1[SCALING_COLS] = scaler.fit_transform(survival_df1[SCALING_COLS])

print("All continuous covariates are now successfully scaled and ready for CPH fitting.")


# In[30]:


survival_df1.head()


# # Fitting the CPH Model

# In[38]:


covariates1 = [
    # Scaled Continuous Covariates
    'FICO_Score', 'Original_DTI', 'Original_CLTV', 'Original_Term',
    'Current_Rate_Last', 'ELTV_Last', 'Current_UPB_Log_ZScore', 
    
    # Encoded Categorical Covariates (These are already 0/1 integers)
    'Occ_I', 'Occ_S',
    
    # Ordinal TVC
    'Delinq_Status_Last'
]

# Ensure the DataFrame only contains the required columns + survival markers
MODEL_COLS = covariates1 + ['Duration_T', 'Event_Delta']
survival_df2 = survival_df1[MODEL_COLS].dropna()

# --- 1. Fit the CPH Model ---
cph = CoxPHFitter(penalizer=0.1) 

cph.fit(
    survival_df2, 
    duration_col='Duration_T', 
    event_col='Event_Delta', 
    show_progress=True
)

print("\n--- CPH Model Fit Summary ---")
cph.print_summary(model="Mortgage Default Hazard Model (Final Fit)", decimals=4)


# In[39]:


# --- Define the FINAL, CLEANED Predictor List ---
# Dropping Original_Term and Original_CLTV to resolve singularity.
covariates2 = [
    # Scaled Continuous Covariates
    'FICO_Score',
    'Original_DTI',
    'Current_Rate_Last',
    'ELTV_Last',
    'Current_UPB_Log_ZScore', 
    
    # Encoded Categorical Covariates
    'Occ_I', 
    'Occ_S',
    
    # Ordinal TVC
    'Delinq_Status_Last'
]

# Ensure the DataFrame only contains the required columns + survival markers
MODEL_COLS1 = covariates2 + ['Duration_T', 'Event_Delta']
df_fit1 = survival_df2[MODEL_COLS1].dropna()

# --- Refit the CPH Model ---
print("\n--- Attempting CPH Refit with Reduced Variables ---")

cph = CoxPHFitter(penalizer=0.01) 
cph.fit(df_fit1, duration_col='Duration_T', event_col='Event_Delta', show_progress=True)
cph.print_summary()

print("\nCPH Model Fit Successful (if no error occurred).")
cph.print_summary(model="Mortgage Default Hazard Model (Final Stabilized Fit)", decimals=4)


# Action: Eliminate Perfect Redundancy (Dummy Variables)
# When using one-hot encoding, you must always define a baseline category and exclude its corresponding column. While your initial plan accounted for this (by having the Primary Residence be the baseline), the model's intercept often creates perfect collinearity with the remaining set of dummy variables.
# 
# The fastest way to resolve a singular matrix issue is to remove one of the encoded dummy variables, as their total contribution might perfectly explain the intercept.

# In[41]:


# --- Define the FINAL, CLEANED Predictor List ---
# Dropping one of the two remaining binary dummies (Occ_I) 
# to eliminate perfect linear dependency with the model's intercept.
covariates3 = [
    # Scaled Continuous Covariates
    'FICO_Score',
    'Original_DTI',
    'Current_Rate_Last',
    'ELTV_Last',
    'Current_UPB_Log_ZScore', 
    
    # Encoded Categorical Covariates (Keeping only Occ_S)
    'Occ_S', 
    
    # Ordinal TVC
    'Delinq_Status_Last'
]

# Assuming df_model contains all necessary columns after scaling
MODEL_COLS = covariates3 + ['Duration_T', 'Event_Delta']
df_fit2 = survival_df2[MODEL_COLS].dropna()

# --- Refit the CPH Model ---
print("\n--- Attempting CPH Refit with Structural Redundancy Removed ---")

cph = CoxPHFitter(penalizer=0.1)

cph.fit(
    df_fit2, 
    duration_col='Duration_T', 
    event_col='Event_Delta', 
    show_progress=True
)

print("\n--- CPH Model Fit Summary (Penalized) ---")
cph.print_summary(decimals=4)

print("\nCPH Model Fit Successful (if no error occurred).")
cph.print_summary(model="Mortgage Default Hazard Model (Final Stabilized Fit)", decimals=4)


# In[44]:


# Create a clean survival dataframe without multicollinearity
survival_df_clean = survival_df1.copy()

# Step 1: Drop the original Current_UPB_Last since we have the log-transformed version
# Keep only ONE representation of this variable
survival_df_clean = survival_df_clean.drop(['Current_UPB_Last'], axis=1)

# Step 2: Select only the covariates we want (remove highly correlated variables)
# Remove Original_UPB if it's causing issues (highly correlated with Current UPB)
cph_covariates = [
    'FICO_Score',
    'Original_CLTV', 
    'Original_DTI',
    'Original_Term',
    'Occ_I',
    'Occ_S',
    'Delinq_Status_Last',
    'ELTV_Last',
    'Current_Rate_Last',
    'Current_UPB_Log_ZScore'  # Use ONLY the transformed version
]

# Step 3: Create the final dataframe for CPH
survival_df_final = survival_df_clean[['Loan_ID', 'Duration_T', 'Event_Delta'] + cph_covariates].copy()

# Step 4: Check for any remaining NaN values
print(f"Missing values:\n{survival_df_final.isnull().sum()}")
survival_df_final = survival_df_final.dropna()

print(f"\nFinal shape: {survival_df_final.shape}")
print(f"Columns for CPH: {cph_covariates}")

# Step 5: Fit the CPH model 
from lifelines import CoxPHFitter

cph = CoxPHFitter(penalizer=0.01)

# Drop 'Loan_ID' here so the model only sees numeric features
cph.fit(
    survival_df_final.drop('Loan_ID', axis=1), 
    duration_col='Duration_T',
    event_col='Event_Delta',
    show_progress=True
)

# Step 6: Display results
print("\n" + "="*60)
print("CPH Model Successfully Fitted!")
print("="*60)
cph.print_summary()


# In[45]:


# Assuming df_fit contains the duration/event columns and predictors
# The structural fixes (dropping highly correlated variables) are incorporated implicitly.

# --- Define the FINAL, CLEANED Predictor List ---
# Dropping: Original_CLTV, Original_Term (due to VIF), and Occ_S (due to singularity)
FINAL_MODEL_COVARS_STABLE = [
    # Scaled Continuous Covariates 
    'FICO_Score',
    'Original_DTI',
    'Current_Rate_Last',
    'ELTV_Last',
    'Current_UPB_Log_ZScore', 
    
    # Encoded Categorical Covariates (Keeping ONLY the highest risk: Occ_I)
    'Occ_I', 
    
    # Ordinal TVC
    'Delinq_Status_Last'
]

# Ensure df_fit is built with these final columns + survival markers
MODEL_COLS = FINAL_MODEL_COVARS_STABLE + ['Duration_T', 'Event_Delta']
df_fit = survival_df_final[MODEL_COLS].dropna() # Use your defined final dataframe

# --- Refit the CPH Model with Penalization ---
print("\n--- Attempting CPH Refit with Penalization (Final Structural Fix) ---")

# CRITICAL FIX: Add a small L2 penalty (penalizer) to stabilize the matrix inversion.
# This pushes near-zero coefficients slightly away from zero, resolving singularity.
cph = CoxPHFitter(penalizer=0.001) # Use a very small penalty

cph.fit(
    df_fit, 
    duration_col='Duration_T', 
    event_col='Event_Delta', 
    show_progress=True
)

print("\nCPH Model Fit Successful (if no error occurred).")
cph.print_summary(model="Mortgage Default Hazard Model (Final Stabilized Fit)", decimals=4)


# # Proportional Hazards (PH) Assumption Check (Mandatory)

# In[46]:


print("\n--- Proportional Hazards Assumption Check ---")
cph.check_assumptions(df_fit, p_value_threshold=0.05)
# This check is vital; if p-value is low (<0.01), the assumption is violated.
#cph.check_assumptions(survival_df2, p_value_threshold=0.01, show_plots=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




