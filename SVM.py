import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import time
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Start timer
start_time = time.time()

# Load dataset (assuming correct path)
file_path = "/mnt/data/DE1_0_2008_to_2010_Outpatient_Claims_Sample_2.csv"
data_1 = pd.read_csv(file_path)

# Step 1: Pad diagnosis and procedure codes
for i in range(10):
    col = f"ICD9_DGNS_CD_{i+1}"
    data_1[col] = data_1[col].astype(str).str.strip().str.pad(5, fillchar='0')

for i in range(44):
    col = f"HCPCS_CD_{i+1}"
    data_1[col] = data_1[col].astype(str).str.strip().str.pad(5, fillchar='0')

# Step 2: Map diagnosis codes to categories
diag_mapping = {
    '00-13': 'Infection_&_Parasitic',
    '14-23': 'Neoplasm',
    '24-27': 'Endocrine_Nutritional_Immunity',
    '28': 'Blood',
    '29-31': 'Mental_&_Behavioral',
    '32-38': 'Nervous',
    '39-45': 'Circulatory',
    '46-51': 'Respiratory',
    '52-57': 'Digestive',
    '58-62': 'Genitourinary',
    '68-70': 'Skin',
    '71-73': 'Musculoskeletal',
    '74-75': 'Congenital_Anomaly',
    '80-99': 'Injury_&_Poisining'
}

for i in range(10):
    col = f"ICD9_DGNS_CD_{i+1}"
    new_col = f"Diag{i+1}"
    data_1[new_col] = ""
    for codes, category in diag_mapping.items():
        start, end = codes.split('-') if '-' in codes else (codes, codes)
        if start == end:
            mask = data_1[col].str.startswith(start)
        else:
            mask = data_1[col].str[:2].between(start, end, inclusive="both")
        data_1.loc[mask, new_col] = category

# Step 3: Map procedure codes to categories
proc_mapping = {
    '0': 'Anesthesia',
    '1-6': 'Surgery',
    '7': 'Radiology',
    '8': 'Pathology_Procedure',
    '992-994': 'E&M',
    'A0': 'Ambulance',
    'A42-A80': 'Medical_Supplies',
    'A9': 'Investigational',
    'J0-J8': 'Drugs_other_than_oral',
    'J9': 'Chemotherapy'
}

for i in range(44):
    col = f"HCPCS_CD_{i+1}"
    new_col = f"Proc{i+1}"
    data_1[new_col] = ""
    for codes, category in proc_mapping.items():
        if '-' in codes:
            start, end = codes.split('-')
            mask = data_1[col].str[:len(start)].between(start, end, inclusive="both")
        else:
            mask = data_1[col].str.startswith(codes)
        data_1.loc[mask, new_col] = category

# Step 4: Create diagnosis and procedure sets per claim
diag_cols = [f'Diag{i+1}' for i in range(10)]
proc_cols = [f'Proc{i+1}' for i in range(44)]

data_1['Diagnosis_Set'] = data_1[diag_cols].apply(lambda row: set(row.dropna()) - {""}, axis=1)
data_1['Procedure_Set'] = data_1[proc_cols].apply(lambda row: set(row.dropna()) - {""}, axis=1)

# Step 5: One-hot encode diagnosis categories
all_diag_categories = sorted(set().union(*data_1['Diagnosis_Set']))
for category in all_diag_categories:
    data_1[category] = data_1['Diagnosis_Set'].apply(lambda s: 1 if category in s else 0)

# Step 6: One-hot encode procedure categories
proc_categories = list(set(proc_mapping.values()))
for category in proc_categories:
    data_1[category] = data_1['Procedure_Set'].apply(lambda s: 1 if category in s else 0)

# Step 7: Use One-Class SVM to flag unnecessary procedures
unnecessary = pd.DataFrame(0, index=data_1.index, columns=proc_categories)

for category in proc_categories:
    claims_with_proc = data_1[data_1[category] == 1]
    if len(claims_with_proc) < 50:
        continue
    X = claims_with_proc[all_diag_categories].values
    svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    svm.fit(X)
    pred = svm.predict(X)
    outlier_mask = (pred == -1)
    unnecessary.loc[claims_with_proc.index[outlier_mask], category] = 1

# Step 8: Mark overall unnecessary claim
data_1['Unnecessary_Claim'] = unnecessary.max(axis=1)
data_1['Flagged_Procedures'] = unnecessary.apply(lambda row: [cat for cat in proc_categories if row[cat] == 1], axis=1)

# Step 9: Provider-level aggregation
provider_agg = data_1.groupby('PRVDR_NUM').agg(
    Total_Claims=('CLM_ID', 'nunique'),
    Unnecessary_Count=('Unnecessary_Claim', 'sum'),
    All_Allegations=('Flagged_Procedures', lambda x: set(cat for row in x for cat in row))
)
provider_agg['perc_unnecessary_claims'] = provider_agg['Unnecessary_Count'] / provider_agg['Total_Claims']

# Step 10: Filter suspicious providers using IQR
Q1 = provider_agg['perc_unnecessary_claims'].quantile(0.25)
Q3 = provider_agg['perc_unnecessary_claims'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR

outlier_providers = provider_agg[provider_agg['perc_unnecessary_claims'] > upper_bound]
outlier_providers_sorted = outlier_providers.sort_values('perc_unnecessary_claims', ascending=False)

# Step 11: Export both outlier providers and claim-level details
dashboard_ready_providers = outlier_providers_sorted.reset_index()
dashboard_ready_claims = data_1[data_1['Unnecessary_Claim'] == 1]

# Save as CSV and JSON
dashboard_ready_providers.to_csv("/mnt/data/flagged_providers.csv", index=False)
dashboard_ready_providers.to_json("/mnt/data/flagged_providers.json", orient="records", indent=2)
dashboard_ready_claims.to_csv("/mnt/data/flagged_claims.csv", index=False)

# Execution time
end_time = time.time()
elapsed_time = end_time - start_time

dashboard_ready_providers.head(), dashboard_ready_claims.head(), elapsed_time
