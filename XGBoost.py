import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration & Setup
# =============================================================================
# Configure warnings to keep the output clean
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Record the start time to measure execution duration
start_time = time.time()

# =============================================================================
# Data Loading & Initial Validation
# =============================================================================
print("--- Starting Data Loading and Validation ---")

# Handle command line arguments safely
if len(sys.argv) < 2:
    print("Error: Please provide the input CSV file path as an argument.")
    print("Usage: python your_script_name.py <path_to_csv>")
    sys.exit(1)

path = sys.argv[1]

# Check if the file exists before attempting to read
if not os.path.exists(path):
    print(f"Error: The file '{path}' was not found.")
    sys.exit(1)

# Define dtypes for code columns to prevent mixed-type issues
dtype_dict = {f"ICD9_DGNS_CD_{i+1}": str for i in range(25)} # Increased range for safety
dtype_dict.update({f"HCPCS_CD_{i+1}": str for i in range(45)})

try:
    print(f"Reading data from: {path}")
    # Use `low_memory=False` to handle large files with mixed dtypes better
    data_1 = pd.read_csv(path, low_memory=False, dtype=dtype_dict)
    print("CSV file read successfully.")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# =============================================================================
# Data Preprocessing & Cleaning
# =============================================================================
print("\n--- Starting Data Preprocessing ---")

# --- Padding zeros to make all code values a uniform length ---
# This is crucial for accurate mapping and categorization.
# A more robust way to handle mixed dtypes and NaNs.
diag_code_cols = [f"ICD9_DGNS_CD_{i+1}" for i in range(25)]
hcpcs_code_cols = [f"HCPCS_CD_{i+1}" for i in range(45)]

for col_name in diag_code_cols:
    if col_name in data_1.columns:
        # Use fillna('') to replace NaN with an empty string before string operations.
        data_1[col_name] = data_1[col_name].fillna('').astype(str).str.strip().str.pad(5, side='left', fillchar='0')

for col_name in hcpcs_code_cols:
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].fillna('').astype(str).str.strip().str.pad(5, side='left', fillchar='0')

print("Padded diagnosis and procedure codes.")

# --- Converting Diagnosis Codes to High-Level Categories ---
diag_categories = {
    'Infection_&_Parasitic': range(0, 140), 'Neoplasm': range(140, 240),
    'Endocrine_Nutritional_Immunity': range(240, 280), 'Blood': range(280, 290),
    'Mental_&_Behavioral': range(290, 320), 'Nervous': range(320, 390),
    'Circulatory': range(390, 460), 'Respiratory': range(460, 520),
    'Digestive': range(520, 580), 'Genitourinary': range(580, 630),
    'Complications_Pregnancy_Childbirth': range(630, 680), 'Skin': range(680, 710),
    'Musculoskeletal': range(710, 740), 'Congenital_Anomaly': range(740, 760),
    'Perinatal_Conditions': range(760, 780), 'Symptoms_&_Ill-defined': range(780, 800),
    'Injury_&_Poisining': range(800, 1000),
    'Supplementary_V_Codes': 'V', 'Supplementary_E_Codes': 'E'
}

def map_diag_code(code):
    """Maps an ICD-9 code to its category."""
    if not code or not isinstance(code, str) or code.strip() in ['00000', '']:
        return None
    if code.startswith('V'):
        return 'Supplementary_V_Codes'
    if code.startswith('E'):
        return 'Supplementary_E_Codes'
    try:
        # Convert the numeric part of the code to an integer for range checking
        code_num = int(code[:3])
        for category, code_range in diag_categories.items():
            if isinstance(code_range, range) and code_num in code_range:
                return category
    except (ValueError, TypeError):
        return None
    return 'Other'

for i in range(25):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    diag_col = f"Diag_Cat_{i+1}"
    if col_name in data_1.columns:
        data_1[diag_col] = data_1[col_name].apply(map_diag_code)

print("Mapped diagnosis codes to categories.")

# --- Converting Procedure Codes to High-Level Categories ---
def map_hcpcs_code(code):
    """Maps an HCPCS code to its category."""
    if not code or not isinstance(code, str) or code.strip() in ['00000', '']:
        return None
    # Check based on starting characters
    if code.startswith('0'): return 'Anesthesia'
    if any(code.startswith(s) for s in ['1', '2', '3', '4', '5', '6']): return 'Surgery'
    if code.startswith('7'): return 'Radiology'
    if code.startswith('8'): return 'Pathology_Procedure'
    if any(code.startswith(s) for s in ['992', '993', '994']): return 'E&M'
    if code.startswith('A0'): return 'Ambulance'
    if 'A42' <= code[:3] <= 'A80': return 'Medical_Supplies'
    if code.startswith('A9'): return 'Investigational'
    if 'J0' <= code[:2] <= 'J8': return 'Drugs_other_than_oral'
    if code.startswith('J9'): return 'Chemotherapy'
    if code.startswith('G'): return 'Temp_Codes_Procedures_Services'
    if code.startswith('Q'): return 'Temp_Codes'
    return 'Other_Services'

for i in range(45):
    col_name = f"HCPCS_CD_{i+1}"
    proc_col = f"Proc_Cat_{i+1}"
    if col_name in data_1.columns:
        data_1[proc_col] = data_1[col_name].apply(map_hcpcs_code)

print("Mapped procedure codes to categories.")

# =============================================================================
# Data Aggregation at Patient Level
# =============================================================================
print("\n--- Aggregating Data at Patient Level ---")

diag_cat_cols = [f'Diag_Cat_{i+1}' for i in range(25) if f'Diag_Cat_{i+1}' in data_1.columns]
proc_cat_cols = [f'Proc_Cat_{i+1}' for i in range(45) if f'Proc_Cat_{i+1}' in data_1.columns]

# Melt the dataframe to have one category per row for each patient
diag_melt = data_1[['DESYNPUF_ID'] + diag_cat_cols].melt(
    id_vars='DESYNPUF_ID', value_name='Diag_Category'
).dropna()
proc_melt = data_1[['DESYNPUF_ID'] + proc_cat_cols].melt(
    id_vars='DESYNPUF_ID', value_name='Proc_Category'
).dropna()

# Group by patient and aggregate categories into a unique set
patient_diag = diag_melt.groupby('DESYNPUF_ID')['Diag_Category'].unique().apply(list).reset_index()
patient_proc = proc_melt.groupby('DESYNPUF_ID')['Proc_Category'].unique().apply(list).reset_index()

# Merge the diagnosis and procedure data
patient_data = pd.merge(patient_diag, patient_proc, on='DESYNPUF_ID', how='outer')
patient_data = patient_data.fillna('').applymap(lambda x: [] if x == '' else x)

print(f"Aggregated data for {patient_data.shape[0]} unique patients.")

# =============================================================================
# Feature Engineering
# =============================================================================
print("\n--- Performing Feature Engineering ---")

# --- One-Hot Encode the list of categories for both diagnosis and procedures ---
mlb_diag = MultiLabelBinarizer()
diag_features = pd.DataFrame(
    mlb_diag.fit_transform(patient_data['Diag_Category']),
    columns=[f"Diag_{c}" for c in mlb_diag.classes_],
    index=patient_data.index
)

mlb_proc = MultiLabelBinarizer()
proc_features = pd.DataFrame(
    mlb_proc.fit_transform(patient_data['Proc_Category']),
    columns=[f"Proc_{c}" for c in mlb_proc.classes_],
    index=patient_data.index
)

# Combine all features into a single dataframe
features = pd.concat([diag_features, proc_features], axis=1)
features['DESYNPUF_ID'] = patient_data['DESYNPUF_ID']

# Scale features for clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.drop('DESYNPUF_ID', axis=1))
print("Created and scaled features for machine learning.")

# =============================================================================
# Patient Clustering (K-Means)
# =============================================================================
print("\n--- Performing Patient Clustering with K-Means ---")

# Determine the optimal number of clusters using the Elbow Method
# inertia = []
# K = range(2, 11)
# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(features_scaled)
#     inertia.append(kmeans.inertia_)

# For speed, we'll skip the elbow plot in this script and choose a fixed k
# plt.figure(figsize=(8, 4))
# plt.plot(K, inertia, 'bo-')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# Let's assume k=5 is a good number of clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster labels to our feature set
features['Cluster'] = clusters
print(f"Assigned patients to {k} clusters.")

# =============================================================================
# Target Variable Creation & Classification (XGBoost)
# =============================================================================
print("\n--- Training Classifier to Predict High-Risk Patients ---")

# --- Create a Target Variable ---
# Let's define a "High-Risk" patient as someone with a Neoplasm diagnosis
# and who has undergone Chemotherapy. This is a proxy for a complex case.
features['High_Risk'] = (features['Diag_Neoplasm'] & features['Proc_Chemotherapy']).astype(int)

# Check if we have enough samples for both classes
target_counts = features['High_Risk'].value_counts()
print("High-Risk Patient Distribution:\n", target_counts)

if len(target_counts) < 2 or target_counts.min() < 10:
    print("\nWarning: Not enough samples for one of the classes. Classification may not be meaningful.")
    print("Skipping classification part.")
else:
    # --- Prepare Data for XGBoost ---
    X = features.drop(['DESYNPUF_ID', 'High_Risk', 'Cluster'], axis=1)
    y = features['High_Risk']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- Train XGBoost Classifier ---
    # Handle class imbalance with scale_pos_weight
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                        eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)

    print("Training XGBoost model...")
    xgb.fit(X_train, y_train)

    # --- Evaluate the Model ---
    print("Evaluating model performance...")
    y_pred = xgb.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# =============================================================================
# Finalization
# =============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n--- Script executed successfully in {elapsed_time:.2f} seconds ---")
