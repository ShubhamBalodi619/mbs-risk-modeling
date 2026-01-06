#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# In[75]:


file_path = r"C:\Users\sbalodi\Desktop\regression_data.csv" #file downloaded
df = pd.read_csv(file_path)
df= df.drop(["First Payment Date", "Monthly Reporting Period"], axis=1)
df.head()


# In[76]:


df.describe()


# In[77]:


df.info()


# In[78]:


df.shape


# In[79]:


df[df["Loan Sequence Number"].duplicated()]


# In[80]:


df['Default'].value_counts()


# In[81]:


numeric_cols = df.select_dtypes(include=np.number).columns


# In[82]:


print("Total records:", len(df))
print("Total unique loans:", df['Loan Sequence Number'].nunique())


# In[83]:


# Clean Number of Borrowers
df['Number of Borrowers'] = pd.to_numeric(df['Number of Borrowers'], errors='coerce')
df['Number of Borrowers'] = df['Number of Borrowers'].apply(
    lambda x: 99 if pd.isna(x) else x
)

# Number of Borrowers vs Default
plt.figure(figsize=(6, 4))
sns.countplot(x='Number of Borrowers', hue='Default', data=df[df['Number of Borrowers'] != 99])  # Exclude missing (99)
plt.title('Number of Borrowers vs Default')
plt.show()


# In[84]:


# Clean Property State (ensure categorical, no numeric conversion needed)
# Assuming it's already a string, no cleaning beyond ensuring valid states can be added if needed

# Property State vs Default
plt.figure(figsize=(10, 4))
sns.countplot(x='Property State', hue='Default', data=df)
plt.title('Property State vs Default')
plt.xticks(rotation=90)
plt.show()


# In[85]:


# Compute default rate per state
state_default = df.groupby('Property State')['Default'].mean().sort_values()

plt.figure(figsize=(12,5))
sns.barplot(x=state_default.index, y=state_default.values)
plt.xticks(rotation=90)
plt.ylabel('Default Rate')
plt.title('Default Rate by Property State')
plt.show()


# In[86]:


# Summary table per state
state_summary = df.groupby('Property State')['Default'].agg(
    total_loans='count',
    total_defaults='sum'
).reset_index()

# Compute default rate
state_summary['default_rate'] = state_summary['total_defaults'] / state_summary['total_loans']

# Sort by default rate descending
state_summary = state_summary.sort_values(by='default_rate', ascending=False)

# Display the table
print(state_summary)


# In[87]:


# Clean Occupancy Status
df['Occupancy Status'] = df['Occupancy Status'].astype(str).str.strip()
df['Occupancy Status'] = df['Occupancy Status'].apply(
    lambda x: '9' if x not in ['O', 'S', 'I'] else x
)
# Occupancy Status vs Default
plt.figure(figsize=(6, 4))
sns.countplot(x='Occupancy Status', hue='Default', data=df)
plt.title('Occupancy Status vs Default')
plt.xlabel('Occupancy (O=Owner, S=Second, I=Investment, 9=Unknown)')

plt.gca().yaxis.set_major_locator(plt.MultipleLocator(500000))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[88]:


# Clean Credit Score based on guide notes
df['Credit Score'] = pd.to_numeric(df['Credit Score'], errors='coerce')
df['Credit Score'] = df['Credit Score'].apply(
    lambda x: 300 if x < 300 else 850 if x > 850 else 9999 if pd.isna(x) else x
)

# Credit Score vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Credit Score', data=df)
plt.title('Credit Score vs Default')
plt.ylim(200, 900)
plt.show()


# In[89]:


# Clean Original Loan Term (no capping needed, just ensure numeric)
df['Original Loan Term'] = pd.to_numeric(df['Original Loan Term'], errors='coerce')

# Original Loan Term vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Original Loan Term', data=df)
plt.title('Original Loan Term (months) vs Default')
plt.ylim(0, 480)  # Reasonable upper limit (e.g., 40 years in months)
#grid yaxis to each 5 years (60 months)
plt.yticks(range(0, 481, 60), [str(i) for i in range(0, 481, 60)])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[90]:


# Clean Loan Age
df['Loan Age'] = pd.to_numeric(df['Loan Age'], errors='coerce')
df['Loan Age'] = df['Loan Age'].apply(
    lambda x: 999 if pd.isna(x) else x
)

# Loan Age vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Loan Age', data=df[df['Loan Age'] != 999])  # Exclude missing (999)
plt.title('Loan Age (months) vs Default')
plt.ylim(0, 120)  #  upper limit (e.g., 10 years in months)

plt.yticks(range(0, 128, 6))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[91]:


# Clean Remaining Months to Legal Maturity
df['Remaining Months to Legal Maturity'] = pd.to_numeric(df['Remaining Months to Legal Maturity'], errors='coerce')
df['Remaining Months to Legal Maturity'] = df['Remaining Months to Legal Maturity'].apply(
    lambda x: 999 if pd.isna(x) else x
)

# Remaining Months to Legal Maturity vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Remaining Months to Legal Maturity', data=df[df['Remaining Months to Legal Maturity'] != 999])  # Exclude missing (999)
plt.title('Remaining Months to Legal Maturity vs Default')
plt.ylim(0, 480)  # Reasonable upper limit (e.g., 40 years in months)

plt.yticks(range(0, 481, 60), [str(i) for i in range(0, 481, 60)])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[92]:


# Clean Current Interest Rate
df['Current Interest Rate'] = pd.to_numeric(df['Current Interest Rate'], errors='coerce')
df['Current Interest Rate'] = df['Current Interest Rate'].apply(
    lambda x: 99.999 if pd.isna(x) else x
)

# Current Interest Rate vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Current Interest Rate', data=df[df['Current Interest Rate'] != 99.999])  # Exclude missing (99.999)
plt.title('Current Interest Rate vs Default')
plt.ylim(0, 15)  # Reasonable upper limit (e.g., 15%)
plt.show()


# In[93]:


# Clean Original Interest Rate (no capping needed, just ensure numeric)
df['Original Interest Rate'] = pd.to_numeric(df['Original Interest Rate'], errors='coerce')

# Original Interest Rate vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Original Interest Rate', data=df)
plt.title('Original Interest Rate vs Default')
plt.ylim(0, 15)  # Reasonable upper limit for interest rates (e.g., 15%)
plt.show()


# In[94]:


# Clean Original Loan-to-Value (LTV)
df['Original Loan-to-Value (LTV)'] = pd.to_numeric(df['Original Loan-to-Value (LTV)'], errors='coerce')
df['Original Loan-to-Value (LTV)'] = df['Original Loan-to-Value (LTV)'].apply(
    lambda x: 999 if pd.isna(x) else x
)

ltv_values= df[df['Original Loan-to-Value (LTV)'] != 999]['Original Loan-to-Value (LTV)']
ltv_values.describe()


# In[95]:


# count ltv values greater than 100
print("LTV values greater than 100:", (ltv_values > 100).sum())


# In[96]:


# Original LTV vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Original Loan-to-Value (LTV)', data= df[df['Original Loan-to-Value (LTV)'] != 999])  # Exclude missing
plt.ylim(0, 1000)  # Reasonable upper limit for LTV
plt.title('Original LTV vs Default')
plt.show()


# In[97]:


# Clean Original Combined Loan-to-Value (CLTV)
df['Original Combined Loan-to-Value (CLTV)'] = pd.to_numeric(df['Original Combined Loan-to-Value (CLTV)'], errors='coerce')
df['Original Combined Loan-to-Value (CLTV)'] = df['Original Combined Loan-to-Value (CLTV)'].apply(
    lambda x: 999 if pd.isna(x) else x
)

# Original CLTV vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Original Combined Loan-to-Value (CLTV)', data= df[df['Original Combined Loan-to-Value (CLTV)'] != 999])  # Exclude missing (999)
plt.title('Original CLTV vs Default')
plt.ylim(0, 200)  # Reasonable upper limit for CLTV
plt.show()


# In[98]:


# Clean Original Debt-to-Income (DTI) Ratio
df['Original Debt-to-Income (DTI) Ratio'] = pd.to_numeric(df['Original Debt-to-Income (DTI) Ratio'], errors='coerce')
df['Original Debt-to-Income (DTI) Ratio'] = df['Original Debt-to-Income (DTI) Ratio'].apply(
    lambda x: 999 if pd.isna(x) else x
)

# Original DTI Ratio vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Original Debt-to-Income (DTI) Ratio', data= df)
plt.title('Original DTI Ratio vs Default')
plt.ylim(0, 100)
plt.show()


# In[99]:


# Clean Current Actual UPB
df['Current Actual UPB'] = pd.to_numeric(df['Current Actual UPB'], errors='coerce')
df['Current Actual UPB'] = df['Current Actual UPB'].apply(
    lambda x: 0 if x == 0 or pd.isna(x) else x
)

# Current Actual UPB vs Default
plt.figure(figsize=(6, 4))
sns.boxplot(x='Default', y='Current Actual UPB', data= df[df['Current Actual UPB'] != 0])  # Exclude zero balance (000000)
plt.title('Current Actual UPB vs Default')
plt.ylim(0, 2500000)  # Reasonable upper limit in $1,000s

plt.yticks(range(0, 2500001, 100000), [str(i) for i in range(0, 2500001, 100000)])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# In[100]:


# Checking the correaltions
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[101]:


df_clean = df.dropna()


# In[102]:


df_clean['Default'].value_counts()


# In[103]:


target = 'Default'
features = [col for col in df.columns if col not in [target, 'Loan Sequence Number']]
# Define features and target after imputation
X = df_clean.drop(columns=['Default', 'Current Loan Delinquency Status','Property State','Occupancy Status','Loan Sequence Number','Original Loan-to-Value (LTV)','Original UPB','Original Loan Term','Original Interest Rate','Original Combined Loan-to-Value (CLTV)','Loan Age']).copy()
y = df_clean[target].copy()

# Scale numeric features with .loc to avoid SettingWithCopyWarning
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])  # Use .loc for assignment


# In[104]:


X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


# In[105]:


# Resample training set (all defaults + sampled non-defaults)

df_train = pd.concat([X_train_full, y_train_full], axis=1)
df_default = df_train[df_train[target] == 1]
df_nondefault = df_train[df_train[target] == 0]


# In[106]:


# 1:6 ratio
n_nondefault = len(df_default) * 6
replace_nondefault = n_nondefault > len(df_nondefault)


# In[107]:


df_nondefault_sample = df_nondefault.sample(n=n_nondefault, replace=replace_nondefault, random_state=42)
df_resampled = pd.concat([df_default, df_nondefault_sample])

X_train_resampled = df_resampled.drop(columns=[target])
y_train_resampled = df_resampled[target]


# In[108]:


model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_resampled, y_train_resampled)


# In[109]:


X_calib, X_eval, y_calib, y_eval = train_test_split(
    X_test_full, y_test_full, test_size=0.5, stratify=y_test_full, random_state=42)


# In[110]:


calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

y_prob_calibrated = calibrated_model.predict_proba(X_eval)[:, 1]


# In[111]:


plt.hist(y_prob_calibrated, bins=50)
plt.title("Predicted Default Probabilities")
plt.show()


# In[112]:


threshold = 0.03 #low threshold 
y_pred_tuned = (y_prob_calibrated >= threshold).astype(int)

roc_auc = roc_auc_score(y_eval, y_prob_calibrated)
precision = precision_score(y_eval, y_pred_tuned)
recall = recall_score(y_eval, y_pred_tuned)
conf_matrix = confusion_matrix(y_eval, y_pred_tuned)

print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# In[113]:


# ROC Curve
fpr, tpr, _ = roc_curve(y_eval, y_prob_calibrated)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision–Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_eval, y_prob_calibrated)
pr_auc = auc(recalls, precisions)

plt.figure(figsize=(6,5))
plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.show()

print("\nClassification Report:")
print(classification_report(y_eval, y_pred_tuned))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




