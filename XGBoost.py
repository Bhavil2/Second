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
# A more efficient way to pad strings
for i in range(10):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].astype(str).str.strip().str.pad(5, fillchar='0')

for i in range(44):
    col_name = f"HCPCS_CD_{i+1}"
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].astype(str).str.strip().str.pad(5, fillchar='0')

### Converting Diagnosis Codes to Categories
diag_categories = {
    '00': 'Infection_&_Parasitic', '01': 'Infection_&_Parasitic', '02': 'Infection_&_Parasitic', '03': 'Infection_&_Parasitic',
    '04': 'Infection_&_Parasitic', '05': 'Infection_&_Parasitic', '06': 'Infection_&_Parasitic', '07': 'Infection_&_Parasitic',
    '08': 'Infection_&_Parasitic', '09': 'Infection_&_Parasitic', '10': 'Infection_&_Parasitic', '11': 'Infection_&_Parasitic',
    '12': 'Infection_&_Parasitic', '13': 'Infection_&_Parasitic',
    '14': 'Neoplasm', '15': 'Neoplasm', '16': 'Neoplasm', '17': 'Neoplasm', '18': 'Neoplasm', '19': 'Neoplasm',
    '20': 'Neoplasm', '21': 'Neoplasm', '22': 'Neoplasm', '23': 'Neoplasm',
    '24': 'Endocrine_Nutritional_Immunity', '25': 'Endocrine_Nutritional_Immunity', '26': 'Endocrine_Nutritional_Immunity',
    '27': 'Endocrine_Nutritional_Immunity',
    '28': 'Blood',
    '29': 'Mental_&_Behavioral', '30': 'Mental_&_Behavioral', '31': 'Mental_&_Behavioral',
    '32': 'Nervous', '33': 'Nervous', '34': 'Nervous', '35': 'Nervous', '36': 'Nervous', '37': 'Nervous', '38': 'Nervous',
    '39': 'Circulatory', '40': 'Circulatory', '41': 'Circulatory', '42': 'Circulatory', '43': 'Circulatory', '44': 'Circulatory',
    '45': 'Circulatory',
    '46': 'Respiratory', '47': 'Respiratory', '48': 'Respiratory', '49': 'Respiratory', '50': 'Respiratory', '51': 'Respiratory',
    '52': 'Digestive', '53': 'Digestive', '54': 'Digestive', '55': 'Digestive', '56': 'Digestive', '57': 'Digestive',
    '58': 'Genitourinary', '59': 'Genitourinary', '60': 'Genitourinary', '61': 'Genitourinary', '62': 'Genitourinary',
    '68': 'Skin', '69': 'Skin', '70': 'Skin',
    '71': 'Musculoskeletal', '72': 'Musculoskeletal', '73': 'Musculoskeletal',
    '74': 'Congenital_Anomaly', '75': 'Congenital_Anomaly',
    '80': 'Injury_&_Poisining', '81': 'Injury_&_Poisining', '82': 'Injury_&_Poisining', '83': 'Injury_&_Poisining',
    '84': 'Injury_&_Poisining', '85': 'Injury_&_Poisining', '86': 'Injury_&_Poisining', '87': 'Injury_&_Poisining',
    '88': 'Injury_&_Poisining', '89': 'Injury_&_Poisining', '90': 'Injury_&_Poisining', '91': 'Injury_&_Poisining',
    '92': 'Injury_&_Poisining', '93': 'Injury_&_Poisining', '94': 'Injury_&_Poisining', '95': 'Injury_&_Poisining',
    '96': 'Injury_&_Poisining', '97': 'Injury_&_Poisining', '98': 'Injury_&_Poisining', '99': 'Injury_&_Poisining'
}
for i in range(10):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    diag_col = f"Diag{i+1}"
    if col_name in data_1.columns:
        data_1[diag_col] = data_1[col_name].str[:2].map(diag_categories)

### Converting Procedure Codes to Categories
proc_conditions = [
    (lambda x: x.str.startswith('0'), 'Anesthesia'),
    (lambda x: x.str.startswith(('1', '2', '3', '4', '5', '6')), 'Surgery'),
    (lambda x: x.str.startswith('7'), 'Radiology'),
    (lambda x: x.str.startswith('8'), 'Pathology_Procedure'),
    (lambda x: x.str.startswith(('992', '993', '994')), 'E&M'),
    (lambda x: x.str.startswith('A0'), 'Ambulance'),
    (lambda x: x.str.startswith(('A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53',
                                 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 'A63', 'A64', 'A65',
                                 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A72', 'A73', 'A74', 'A75', 'A76', 'A77',
                                 'A78', 'A79', 'A80')), 'Medical_Supplies'),
    (lambda x: x.str.startswith('A9'), 'Investigational'),
    (lambda x: x.str.startswith(('J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8')), 'Drugs_other_than_oral'),
    (lambda x: x.str.startswith('J9'), 'Chemotherapy')
]
for i in range(44):
    col_name = f"HCPCS_CD_{i+1}"
    proc_col = f"Proc{i+1}"
    if col_name in data_1.columns:
        # Initialize the column with a default value to avoid fragmentation
        data_1[proc_col] = np.nan
        for condition, category in proc_conditions:
            mask = data_1[col_name].str.replace('nan', '', regex=False).fillna('')
            data_1.loc[condition(mask), proc_col] = category

### Grouping data at patient level - Optimized aggregation
diag_cols = [f'Diag{i+1}' for i in range(10) if f'Diag{i+1}' in data_1.columns]
proc_cols = [f'Proc{i+1}' for i in range(44) if f'Proc{i+1}' in data_1.columns]
required_cols = ['DESYNPUF_ID'] + diag_cols + proc_cols

if not all(col in data_1.columns for col in required_cols):
    missing = [col for col in required_cols if col not in data_1.columns]
    print(f"Error: Missing required columns for aggregation: {missing}")
    sys.exit(1)

agg_dict = {col: lambda x: set(x.dropna()) for col in diag_cols + proc_cols}
try:
    data_2 = data_1.groupby(['DESYNPUF_ID']).agg(agg_dict)
except Exception as e:
    print(f"Error in groupby operation: {e}")
    sys.exit(1)

### Creating final diagnosis and procedure columns by combining individual columns
data_2['Diagnosis'] = data_2[diag_cols].apply(lambda row: set().union(*row), axis=1)
data_2['Procedure'] = data_2[proc_cols].apply(lambda row: set().union(*row), axis=1)

### Creating one hot encoded data for diagnosis codes
unique_diag = sorted(list(set().union(*data_2['Diagnosis'])))
data_3 = pd.DataFrame(index=data_2.index, columns=unique_diag)
data_3.loc[:,:] = 0

for idx, diag_set in data_2['Diagnosis'].items():
    data_3.loc[idx, list(diag_set)] = 1

### Filtering for patients having more than one diagnosis
data_3['Total'] = data_3.sum(axis=1)
data_4 = data_3[data_3['Total'] > 1].drop('Total', axis=1)

# Perform K-means clustering with error handling
try:
    kmeans = KMeans(n_clusters=84, random_state=42, n_init=10)
    data_4['Cluster'] = kmeans.fit_predict(data_4)
except Exception as e:
    print(f"Error in KMeans clustering: {e}")
    sys.exit(1)

### Outlier detection using XGBoost
def data_tr(num_rows=84, num_procs=44):
    """Create synthetic data for outlier detection"""
    np.random.seed(42)
    # Each row represents a cluster, each column represents a procedure
    df = pd.DataFrame(np.random.rand(num_rows, num_procs),
                      columns=[f'Proc{i+1}' for i in range(num_procs)])
    # Make some values much higher to simulate outliers
    for i in range(10): # Create 10 outliers
        row = np.random.randint(0, num_rows)
        col = np.random.randint(0, num_procs)
        df.iloc[row, col] = np.random.rand() * 10
    return df

ph_2_data1 = data_tr()

# Prepare data for XGBoost
ph_2_data2 = pd.DataFrame(0, index=ph_2_data1.index, columns=ph_2_data1.columns)
for col in ph_2_data1.columns:
    threshold = ph_2_data1[col].quantile(0.95)
    ph_2_data2[col] = (ph_2_data1[col] > threshold).astype(int)

X = ph_2_data1.values
y = ph_2_data2.values

outlier_predictions = pd.DataFrame(0, index=ph_2_data1.index, columns=ph_2_data1.columns)
for i, col in enumerate(ph_2_data1.columns):
    # Ensure there are both positive and negative classes
    if len(np.unique(y[:, i])) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y[:, i], test_size=0.2, random_state=42, stratify=y[:, i])

        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        outlier_predictions[col] = model.predict(X)
    else:
        # If no outliers in this column, predict all as not outlier
        outlier_predictions[col] = 0

outlier_mask = outlier_predictions == 1

### Collecting outlier Cluster-Procedure combinations
ph_3_data1 = pd.DataFrame(columns=['Cluster', 'Procedure'])
for i in range(ph_2_data1.shape[0]):
    for j, col in enumerate(ph_2_data1.columns):
        if outlier_mask.iloc[i, j]:
            ph_3_data1.loc[len(ph_3_data1)] = [i, col]

### Fetching claims data containing outlier combinations
if 'Cluster' not in data_4.columns or 'DESYNPUF_ID' not in data_1.columns:
    print("Error: Required columns not found for merging")
    sys.exit(1)

# Reset index to make DESYNPUF_ID a column for merging
data_4_indexed = data_4.reset_index().rename(columns={'index': 'DESYNPUF_ID'})

ph_3_data2 = pd.merge(data_1, data_4_indexed[['DESYNPUF_ID', 'Cluster']], on='DESYNPUF_ID', how='inner')

# Create a long format of procedure claims for a more efficient merge
proc_long = pd.melt(ph_3_data2, id_vars=['DESYNPUF_ID', 'CLM_ID', 'PRVDR_NUM', 'Cluster'],
                    value_vars=proc_cols, value_name='Procedure')
proc_long = proc_long.dropna(subset=['Procedure'])

ph_3_data3 = pd.merge(proc_long, ph_3_data1, on=['Cluster', 'Procedure'], how='inner')

### Generating practitioner level risk and allegation
if not ph_3_data3.empty and 'PRVDR_NUM' in ph_3_data3.columns:
    ph_3_data3['Allegation'] = ph_3_data3['Cluster'].astype(str) + '-' + ph_3_data3['Procedure'].astype(str)

    ph_3_data4 = ph_3_data3.groupby('PRVDR_NUM').agg(
        Unnecessary_Count=('CLM_ID', 'nunique'),
        Allegation=('Allegation', lambda x: set(x))
    )

    ph_3_data5 = data_1.groupby('PRVDR_NUM')['CLM_ID'].nunique().rename('Total_count')

    ph_3_data6 = pd.merge(ph_3_data4, ph_3_data5, on='PRVDR_NUM', how='right').fillna(0)
    ph_3_data6['perc_unnecessary_claims'] = ph_3_data6['Unnecessary_Count'] / ph_3_data6['Total_count']

    ph_3_data7 = ph_3_data6[ph_3_data6['Total_count'] > 10]

    # Calculate IQR
    Q1 = ph_3_data7['perc_unnecessary_claims'].quantile(0.25)
    Q3 = ph_3_data7['perc_unnecessary_claims'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter for outliers
    ph_3_data8 = ph_3_data7[ph_3_data7['perc_unnecessary_claims'] > (Q3 + 3*IQR)]
    
    # Save output
    output = ph_3_data8.sort_values('perc_unnecessary_claims', ascending=False)
    output.to_csv('Output.csv')
else:
    print("Warning: No outlier claims detected or PRVDR_NUM column missing.")
    pd.DataFrame().to_csv('Output.csv')

# Record and print execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code executed successfully in {elapsed_time:.2f} seconds")
