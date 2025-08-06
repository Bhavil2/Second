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

warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Record the start time
start_time = time.time()

# Access command line arguments
arg1 = sys.argv[1]
path = '\DE1_0_2008_to_2010_Outpatient_Claims_Sample_2.csv' + arg1

### Reading the data
data_1 = pd.read_csv(path)

### Padding zeros to make all values of same length
for i in range(10):
    data_1["ICD9_DGNS_CD_" + str(i+1)] = data_1["ICD9_DGNS_CD_" + str(i+1)].str.strip()
    data_1["ICD9_DGNS_CD_" + str(i+1)] = data_1["ICD9_DGNS_CD_" + str(i+1)].str.pad(5, fillchar='0')
    
for i in range(44):
    data_1["HCPCS_CD_" + str(i+1)] = data_1["HCPCS_CD_" + str(i+1)].str.strip()
    data_1["HCPCS_CD_" + str(i+1)] = data_1["HCPCS_CD_" + str(i+1)].str.pad(5, fillchar='0')

### Converting Diagnosis Codes to Categories
for i in range(10):
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('00', '13', inclusive=True),"Diag"+str(i+1)] = 'Infection_&_Parasitic'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('14', '23', inclusive=True),"Diag"+str(i+1)] = 'Neoplasm'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('24', '27', inclusive=True),"Diag"+str(i+1)] = 'Endocrine_Nutritional_Immunity'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2] == '28',"Diag"+str(i+1)] = 'Blood'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('29', '31', inclusive=True),"Diag"+str(i+1)] = 'Mental_&_Behavioral'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('32', '38', inclusive=True),"Diag"+str(i+1)] = 'Nervous'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('39', '45', inclusive=True),"Diag"+str(i+1)] = 'Circulatory'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('46', '51', inclusive=True),"Diag"+str(i+1)] = 'Respiratory'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('52', '57', inclusive=True),"Diag"+str(i+1)] = 'Digestive'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('58', '62', inclusive=True),"Diag"+str(i+1)] = 'Genitourinary'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('68', '70', inclusive=True),"Diag"+str(i+1)] = 'Skin'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('71', '73', inclusive=True),"Diag"+str(i+1)] = 'Musculoskeletal'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('74', '75', inclusive=True),"Diag"+str(i+1)] = 'Congenital_Anomaly'
    data_1.loc[data_1["ICD9_DGNS_CD_" + str(i+1)].str[:2].between('80', '99', inclusive=True),"Diag"+str(i+1)] = 'Injury_&_Poisining'

### Converting Procedure Codes to Categories
for i in range(44):
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:1] == '0',"Proc" + str(i+1)] = 'Anesthesia'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:1].between('1', '6', inclusive=True),"Proc" + str(i+1)] = 'Surgery'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:1] == '7',"Proc" + str(i+1)] = 'Radiology'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:1] == '8',"Proc" + str(i+1)] = 'Pathology_Procedure'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:3].between('992', '994', inclusive=True),"Proc" + str(i+1)] = 'E&M'   
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:2] == 'A0',"Proc" + str(i+1)] = 'Ambulance'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:3].between('A42', 'A80', inclusive=True),"Proc" + str(i+1)] = 'Medical_Supplies' 
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:2] == 'A9',"Proc" + str(i+1)] = 'Investigational'
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:2].between('J0', 'J8', inclusive=True),"Proc" + str(i+1)] = 'Drugs_other_than_oral'  
    data_1.loc[data_1["HCPCS_CD_" + str(i+1)].str[:2] == 'J9',"Proc" + str(i+1)] = 'Chemotherapy'

### Grouping data at patient level
data_2 = data_1.groupby(['DESYNPUF_ID']).agg({'Diag1':set, 'Diag2':set, 'Diag3':set, 'Diag4':set, 'Diag5':set, 
                                              'Diag6':set, 'Diag7':set, 'Diag8':set, 'Diag9':set, 'Diag10':set,
                                             'Proc1':set, 'Proc2':set, 'Proc3':set, 'Proc4':set, 'Proc5':set,
                                             'Proc6':set, 'Proc7':set, 'Proc8':set, 'Proc9':set, 'Proc10':set,
                                             'Proc11':set, 'Proc12':set, 'Proc13':set, 'Proc14':set, 'Proc15':set,
                                             'Proc16':set, 'Proc17':set, 'Proc18':set, 'Proc19':set, 'Proc20':set,
                                             'Proc21':set, 'Proc22':set, 'Proc23':set, 'Proc24':set, 'Proc25':set,
                                             'Proc26':set, 'Proc27':set, 'Proc28':set, 'Proc29':set, 'Proc30':set,
                                             'Proc31':set, 'Proc32':set, 'Proc33':set, 'Proc34':set, 'Proc35':set,
                                             'Proc36':set, 'Proc37':set, 'Proc38':set, 'Proc39':set, 'Proc40':set,
                                             'Proc41':set, 'Proc42':set, 'Proc43':set, 'Proc44':set})

### Removing nan values from Diagnosis & Procedure sets
def remove_nan(s):
    return {x for x in s if pd.notna(x)}

for i in range(10):
    data_2["Diag"+ str(i+1)] = data_2["Diag"+ str(i+1)].apply(remove_nan)

for i in range(44):
    data_2["Proc"+ str(i+1)] = data_2["Proc"+ str(i+1)].apply(remove_nan)

### Creating final diagnosis and procedure columns by combining individual columns
data_2['Diagnosis'] = data_2.apply(lambda row: row['Diag1'].union(row['Diag2']), axis=1)
for i in range(8):
    data_2['Diagnosis'] = data_2.apply(lambda row: row['Diagnosis'].union(row['Diag'+str(i+3)]), axis=1)
    
data_2['Procedure'] = data_2.apply(lambda row: row['Proc1'].union(row['Proc2']), axis=1)    
for i in range(42):
    data_2['Procedure'] = data_2.apply(lambda row: row['Procedure'].union(row['Proc'+str(i+3)]), axis=1)

### Creating one hot encoded data for diagnosis codes
unique_diag = list(set([val for sublist in data_2['Diagnosis'] for val in sublist]))
data_3 = pd.DataFrame(0, index=data_2.index, columns=unique_diag)

for idx, row in data_2.iterrows():
    for val in row['Diagnosis']:
        data_3.loc[idx, val] = 1

### Filtering for patients having more than one diagnosis
data_3['Total'] = data_3.sum(axis=1)
data_4 = data_3[data_3['Total'] > 1]
data_4 = data_4.drop(data_4.columns[-1], axis=1)

# Perform K-means clustering  
kmeans = KMeans(n_clusters=84, random_state=42)  
data_4['Cluster'] = kmeans.fit_predict(data_4)  

def data_tr(num_rows=84):
    df = pd.DataFrame(index=range(num_rows)) 
    df['Medical_Supplies'] = [278,219,0,0,0,220,0,0,0,345,0,125,0,0,0,0,119,0,0,0,0,456,0,156,123,0,387,0,0,93,109,0,0,0,0,576,0,0,0,154,0,690,0,105,126,157,0,0,559,0,208,149,0,0,0,0,93,0,0,467,0,0,221,0,0,0,0,0,0,0,92,0,95,0,0,0,0,0,0,147,138,0,0,0]
    df['Investigational'] = [5,243,329,197,299,134,71,216,0,411,295,313,268,0,0,0,510,0,0,0,0,355,348,389,317,278,0,344,264,308,302,271,0,0,0,250,324,0,0,441,0,262,0,0,526,380,388,0,461,0,280,418,0,313,0,0,336,0,0,0,291,0,372,0,0,259,0,0,0,0,412,0,0,0,256,0,0,0,0,269,0,0,0,267]
    df['Anesthesia'] = [436,178,4,589,589,509,12,780,8,290,1200,270,1700,7,13,300,370,720,200,9,12,300,1600,380,290,750,500,220,1150,370,300,600,7,4,200,220,250,4,13,300,900,240,9,220,320,340,310,7,890,2,290,360,500,720,12,13,240,14,10,2,769,6,320,7,11,678,9,11,400,2,300,389,489,6,890,10,7,5,632,280,10,10,5,220]
    df['Surgery'] = [519,493,0,1089,768,601,0,756,469,1750,1334,542,1862,468,598,511,1373,758,1680,672,991,866,2118,902,627,857,532,1483,1158,872,643,768,650,0,639,544,724,915,936,1632,1097,649,449,0,1274,655,797,610,982,0,704,882,571,729,412,485,1136,756,678,36,1007,0,754,0,584,843,737,481,539,0,1256,434,507,458,968,499,558,0,649,465,857,423,464,1030]
    df['Chemotherapy'] = [7,5,13,0,94,2,7,0,0,0,0,132,0,0,0,0,147,0,0,0,0,113,0,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,0,0,0,0,183,133,132,0,181,0,0,170,0,0,0,0,0,0,0,0,0,0,137,0,0,0,0,0,0,0,115,0,0,0,0,0,0,0,0,0,0,0,0,0]
    df['Pathology_Procedure'] = [611,499,520,2,774,2,249,817,486,1955,1576,544,2342,594,696,517,1402,856,2046,705,1113,870,2771,914,631,881,541,1573,1306,881,646,792,750,327,649,551,733,1202,1091,1699,1283,658,491,395,1281,657,802,838,985,154,708,884,587,735,417,507,1229,811,706,0,1078,500,755,436,593,882,771,494,542,438,1288,436,530,576,1010,544,574,462,681,465,1158,446,493,1127]
    df['Radiology'] = [483,490,1,1019,754,650,0,739,464,1701,1408,537,1538,505,552,503,1349,757,1296,645,792,858,2546,898,627,839,526,1375,1178,847,632,755,567,0,617,539,720,893,938,1589,1054,637,470,0,1258,649,785,622,964,0,680,876,538,724,404,457,1153,676,610,34,999,0,750,0,565,808,663,465,522,0,1209,428,451,485,927,489,518,0,635,461,791,422,453,1032]
    df['Ambulance'] = [246,0,0,0,0,0,0,467,0,0,0,0,0,0,0,0,191,0,0,0,0,0,0,183,456,0,0,0,0,168,156,0,231,0,0,467,0,0,0,200,0,0,0,0,217,153,148,0,195,0,139,225,0,158,0,0,0,0,0,0,0,0,198,0,0,0,0,0,386,0,192,0,0,0,0,0,0,0,0,137,0,0,0,0]
    df['E&M'] = [474,488,0,0,754,616,0,704,464,1548,1263,538,1525,437,547,505,1356,728,1302,649,818,837,1995,893,627,833,0,1335,1062,848,638,750,555,0,612,531,717,793,0,1566,981,637,451,0,1247,651,784,604,969,0,693,875,556,727,401,454,1111,716,632,78,986,0,754,0,581,795,684,469,537,0,0,429,456,413,915,471,515,0,632,461,718,0,463,999]
    df['Drugs_other_than_oral'] = [0,455,0,0,629,0,0,520,361,968,877,515,834,0,0,411,1098,481,724,453,499,725,0,759,556,627,452,849,712,711,581,598,381,0,487,458,619,589,528,1082,696,0,0,379,1073,601,696,0,937,0,612,793,424,608,0,0,794,483,404,0,693,0,692,0,448,577,514,371,456,0,925,399,0,0,620,0,373,0,423,437,446,0,0,697]
    return df

### Outlier detection using XGBoost
ph_2_data1 = data_tr()

# For XGBoost, we need labeled data. Since this is unsupervised, we'll create synthetic labels
# where we consider the top 5% of values in each column as outliers (1) and others as inliers (0)
ph_2_data2 = pd.DataFrame()

for col in ph_2_data1.columns:
    threshold = ph_2_data1[col].quantile(0.95)
    ph_2_data2[col] = (ph_2_data1[col] > threshold).astype(int)

# Prepare data for XGBoost
X = ph_2_data1.values
y = ph_2_data2.values

# Since we have multiple target columns, we'll train a separate model for each
outlier_predictions = pd.DataFrame()

for i, col in enumerate(ph_2_data1.columns):
    # Split data for this procedure type
    X_train, X_test, y_train, y_test = train_test_split(X, y[:, i], test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict on all data
    preds = model.predict(X)
    outlier_predictions[col] = preds

# Convert predictions to boolean (True for outliers)
outlier_mask = outlier_predictions == 1

### Collecting outlier Cluster-Procedure Combination 
ph_3_data1 = pd.DataFrame(columns=['Cluster', 'Procedure'])

for i in range(84):
    for j in range(10):
        if outlier_mask.iloc[i,j] == True:
            ph_3_data1.loc[len(ph_3_data1)] = [i, outlier_mask.columns[j]]

### Fetching claims data containing Outlier cluster-procedure combinations
ph_3_data2 = data_4[['Cluster']].merge(data_1, how='inner', left_index=True, right_on='DESYNPUF_ID')
ph_3_data3 = pd.DataFrame(columns=ph_3_data2.columns.tolist() + ['Procedure'])

for i in range(1,45):
    p = "Proc"+ str(i)
    df1 = pd.merge(ph_3_data2, ph_3_data1, left_on=['Cluster', p], right_on=['Cluster', 'Procedure'], how='inner') 
    ph_3_data3 = pd.concat([ph_3_data3, df1])

### Generating practitioner level risk and allegation 
ph_3_data3['Allegation'] = ph_3_data3['Cluster'].astype(str) + '-' + ph_3_data3['Procedure'].astype(str)  
ph_3_data4 = ph_3_data3.groupby('PRVDR_NUM').agg(Unnecessary_Count=('CLM_ID', 'nunique'), Allegation=('Allegation', lambda x: set(x.tolist())))

ph_3_data5 = ph_3_data2.groupby('PRVDR_NUM').agg(Total_count=('CLM_ID', 'nunique'))

ph_3_data6 = pd.merge(ph_3_data4, ph_3_data5, on='PRVDR_NUM', how='right').fillna(0)
ph_3_data6['perc_unnecessary_claims'] = ph_3_data6['Unnecessary_Count'].div(ph_3_data6['Total_count'])

ph_3_data7 = ph_3_data6[ph_3_data6['Total_count'] > 10]

### Applying IQR to select the target population 
Q1 = ph_3_data7['perc_unnecessary_claims'].quantile(0.25)
Q3 = ph_3_data7['perc_unnecessary_claims'].quantile(0.75)
IQR = Q3 - Q1

ph_3_data8 = ph_3_data7[ph_3_data7['perc_unnecessary_claims'] > (Q3 + 3*IQR)]

### Sort the values and display the output 
output = ph_3_data8.sort_values(by=['perc_unnecessary_claims'], ascending=False)
output.to_csv('C:\\Users\\HOME\\Desktop\\Saransh\\Mtech\\Dissertation\\Code\\Output.csv')

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Code executed successfully using XGBoost") 
print(f"Execution time: {elapsed_time:.2f} seconds")
