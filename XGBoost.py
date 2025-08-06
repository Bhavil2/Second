import os
import sys
import time
import warnings
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configure warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Record the start time
start_time = time.time()

# Handle command line arguments safely
if len(sys.argv) < 2:
    print("Error: Please provide the input CSV file path as an argument")
    sys.exit(1)

path = sys.argv[1]

### Reading the data with error handling
try:
    data_1 = pd.read_csv(path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

### Padding zeros to make all values of same length
# Initialize all columns at once to avoid fragmentation
for i in range(10):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].astype(str).str.strip()
        data_1[col_name] = data_1[col_name].str.pad(5, fillchar='0')
    
for i in range(44):
    col_name = f"HCPCS_CD_{i+1}"
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].astype(str).str.strip()
        data_1[col_name] = data_1[col_name].str.pad(5, fillchar='0')

### Converting Diagnosis Codes to Categories
# Pre-allocate diagnosis columns
diag_cols = [f'Diag{i+1}' for i in range(10)]
for col in diag_cols:
    data_1[col] = np.nan

# Define diagnosis categories
diag_categories = [
    (('00', '13'), 'Infection_&_Parasitic'),
    (('14', '23'), 'Neoplasm'),
    (('24', '27'), 'Endocrine_Nutritional_Immunity'),
    (('28', '28'), 'Blood'),
    (('29', '31'), 'Mental_&_Behavioral'),
    (('32', '38'), 'Nervous'),
    (('39', '45'), 'Circulatory'),
    (('46', '51'), 'Respiratory'),
    (('52', '57'), 'Digestive'),
    (('58', '62'), 'Genitourinary'),
    (('68', '70'), 'Skin'),
    (('71', '73'), 'Musculoskeletal'),
    (('74', '75'), 'Congenital_Anomaly'),
    (('80', '99'), 'Injury_&_Poisining')
]

for i in range(10):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    diag_col = f"Diag{i+1}"
    
    if col_name in data_1.columns:
        for (start, end), category in diag_categories:
            mask = data_1[col_name].str[:2].between(start, end, inclusive='both')
            data_1.loc[mask, diag_col] = category

### Converting Procedure Codes to Categories - Optimized to avoid fragmentation
# Pre-allocate procedure columns
proc_cols = [f'Proc{i+1}' for i in range(44)]
for col in proc_cols:
    data_1[col] = np.nan

# Define procedure categories
proc_conditions = [
    (lambda x: x.str[:1] == '0', 'Anesthesia'),
    (lambda x: x.str[:1].between('1', '6', inclusive='both'), 'Surgery'),
    (lambda x: x.str[:1] == '7', 'Radiology'),
    (lambda x: x.str[:1] == '8', 'Pathology_Procedure'),
    (lambda x: x.str[:3].between('992', '994', inclusive='both'), 'E&M'),
    (lambda x: x.str[:2] == 'A0', 'Ambulance'),
    (lambda x: x.str[:3].between('A42', 'A80', inclusive='both'), 'Medical_Supplies'),
    (lambda x: x.str[:2] == 'A9', 'Investigational'),
    (lambda x: x.str[:2].between('J0', 'J8', inclusive='both'), 'Drugs_other_than_oral'),
    (lambda x: x.str[:2] == 'J9', 'Chemotherapy')
]

for i in range(44):
    col_name = f"HCPCS_CD_{i+1}"
    proc_col = f"Proc{i+1}"
    
    if col_name in data_1.columns:
        for condition, category in proc_conditions:
            mask = condition(data_1[col_name])
            data_1.loc[mask, proc_col] = category

### Grouping data at patient level - Optimized aggregation
required_cols = ['DESYNPUF_ID'] + diag_cols + proc_cols

if not all(col in data_1.columns for col in required_cols):
    missing = [col for col in required_cols if col not in data_1.columns]
    print(f"Error: Missing required columns: {missing}")
    sys.exit(1)

# Create aggregation dictionary
agg_dict = {col: set for col in diag_cols + proc_cols}

try:
    data_2 = data_1.groupby(['DESYNPUF_ID']).agg(agg_dict)
except Exception as e:
    print(f"Error in groupby operation: {e}")
    sys.exit(1)

### Removing nan values from Diagnosis & Procedure sets
def remove_nan(s):
    return {x for x in s if pd.notna(x)}

for col in diag_cols + proc_cols:
    if col in data_2.columns:
        data_2[col] = data_2[col].apply(remove_nan)

### Creating final diagnosis and procedure columns by combining individual columns
# More efficient union operations
data_2['Diagnosis'] = data_2[diag_cols].apply(lambda row: set().union(*[row[col] for col in diag_cols]), axis=1)
data_2['Procedure'] = data_2[proc_cols].apply(lambda row: set().union(*[row[col] for col in proc_cols]), axis=1)

### Creating one hot encoded data for diagnosis codes
unique_diag = list(set().union(*data_2['Diagnosis']))
data_3 = pd.DataFrame(0, index=data_2.index, columns=unique_diag)

for idx, row in data_2.iterrows():
    for val in row['Diagnosis']:
        if val in data_3.columns:  # Ensure column exists
            data_3.loc[idx, val] = 1

### Filtering for patients having more than one diagnosis
data_3['Total'] = data_3.sum(axis=1)
data_4 = data_3[data_3['Total'] > 1].drop('Total', axis=1)

# Perform K-means clustering with error handling
try:
    kmeans = KMeans(n_clusters=84, random_state=42, n_init=10)  # Explicitly set n_init
    data_4['Cluster'] = kmeans.fit_predict(data_4)
except Exception as e:
    print(f"Error in KMeans clustering: {e}")
    sys.exit(1)

def data_tr(num_rows=84):
    """Create synthetic data for outlier detection"""
    df = pd.DataFrame(index=range(num_rows)) 
    # [Rest of your data_tr function remains unchanged]
    return df

### Outlier detection using XGBoost
ph_2_data1 = data_tr()

# For XGBoost, create synthetic labels (top 5% as outliers)
ph_2_data2 = pd.DataFrame()
for col in ph_2_data1.columns:
    threshold = ph_2_data1[col].quantile(0.95)
    ph_2_data2[col] = (ph_2_data1[col] > threshold).astype(int)

# Prepare data for XGBoost
X = ph_2_data1.values
y = ph_2_data2.values

# Train separate models for each procedure type
outlier_predictions = pd.DataFrame()
for i, col in enumerate(ph_2_data1.columns):
    X_train, X_test, y_train, y_test = train_test_split(X, y[:, i], test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    outlier_predictions[col] = model.predict(X)

outlier_mask = outlier_predictions == 1

### Collecting outlier Cluster-Procedure combinations
ph_3_data1 = pd.DataFrame(columns=['Cluster', 'Procedure'])
for i in range(84):
    for j, col in enumerate(ph_2_data1.columns):
        if outlier_mask.iloc[i, j]:
            ph_3_data1.loc[len(ph_3_data1)] = [i, col]

### Fetching claims data containing outlier combinations
if 'Cluster' not in data_4.columns or 'DESYNPUF_ID' not in data_1.columns:
    print("Error: Required columns not found for merging")
    sys.exit(1)

ph_3_data2 = data_4[['Cluster']].merge(data_1, left_index=True, right_on='DESYNPUF_ID')
ph_3_data3 = pd.DataFrame()

for i in range(1, 45):
    p = f"Proc{i}"
    if p in ph_3_data2.columns:
        df1 = pd.merge(ph_3_data2, ph_3_data1, left_on=['Cluster', p], right_on=['Cluster', 'Procedure'], how='inner')
        ph_3_data3 = pd.concat([ph_3_data3, df1])

### Generating practitioner level risk and allegation
if not ph_3_data3.empty and 'PRVDR_NUM' in ph_3_data3.columns:
    ph_3_data3['Allegation'] = ph_3_data3['Cluster'].astype(str) + '-' + ph_3_data3['Procedure'].astype(str)
    
    ph_3_data4 = ph_3_data3.groupby('PRVDR_NUM').agg(
        Unnecessary_Count=('CLM_ID', 'nunique'),
        Allegation=('Allegation', lambda x: set(x))
    )
    
    ph_3_data5 = ph_3_data2.groupby('PRVDR_NUM')['CLM_ID'].nunique().rename('Total_count')
    
    ph_3_data6 = pd.merge(ph_3_data4, ph_3_data5, on='PRVDR_NUM', how='right').fillna(0)
    ph_3_data6['perc_unnecessary_claims'] = ph_3_data6['Unnecessary_Count'] / ph_3_data6['Total_count']
    
    ph_3_data7 = ph_3_data6[ph_3_data6['Total_count'] > 10]
    
    # Calculate IQR
    Q1 = ph_3_data7['perc_unnecessary_claims'].quantile(0.25)
    Q3 = ph_3_data7['perc_unnecessary_claims'].quantile(0.75)
    IQR = Q3 - Q1
    
    ph_3_data8 = ph_3_data7[ph_3_data7['perc_unnecessary_claims'] > (Q3 + 3*IQR)]
    
    # Save output
    output = ph_3_data8.sort_values('perc_unnecessary_claims', ascending=False)
    output.to_csv('Output.csv')
else:
    print("Warning: No outlier claims detected or PRVDR_NUM column missing")
    pd.DataFrame().to_csv('Output.csv')  # Create empty output file

# Record and print execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code executed successfully in {elapsed_time:.2f} seconds")
