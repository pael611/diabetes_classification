#!/usr/bin/env python
# coding: utf-8

# **PROYEK PREDIKSI ANALISIS DIABETES**

# Nama: Rafael Siregar<br>
# Dicoding e-mail: rafael_siregar@students.polmed.ac.id<br>
# username: rafael_siregar 

# ***Import Library***

# In[592]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, average_precision_score, PrecisionRecallDisplay,
    make_scorer,mean_squared_error
)


# ***Load Dataset***

# In[593]:


#Melakuakan Load dataset Diabetes dataset.csv
df_dataset = pd.read_csv('dataset.csv')
#Melakukan pengecekan data
df_dataset.head()


# ***Melakuakn Exploratory Data Analysis - Deskripsi Variabel pada Dataset yang digunakan***

# In[594]:


#Melakukan pengecekan missing value
df_dataset.info()
#Mengecek deskripsi Variabel data
df_dataset.describe()


# ***Menangani Missing Values dari Tiap kolom yang ada***

# In[595]:


#Cek berapa banyak nilai 0 pada tiap kolom yang ada di dataset
df_dataset.isin([0]).sum()
#menghapus nilai 0 pada kolom Glucose, BloodPressure, SkinThickness, Insulin, BMI
df_dataset = df_dataset.loc[(df_dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] != 0).all(axis=1)]
df_dataset = df_dataset.reset_index(drop=True)
df_dataset.describe()


# 

# In[596]:


#Menangani Outliers pada dataset kecuali kolom Outcome menggunakan IQR
#visualisasi boxplot untuk mendeteksi outliers pada tiap kolom
sns.boxplot(x=df_dataset['Pregnancies'])
plt.show()
sns.boxplot(x=df_dataset['Glucose'])
plt.show()
sns.boxplot(x=df_dataset['BloodPressure'])
plt.show()
sns.boxplot(x=df_dataset['SkinThickness'])
plt.show()
sns.boxplot(x=df_dataset['Insulin'])
plt.show()
sns.boxplot(x=df_dataset['BMI'])
plt.show()
sns.boxplot(x=df_dataset['DiabetesPedigreeFunction'])
plt.show()
sns.boxplot(x=df_dataset['Age'])
plt.show()


# In[597]:


#Menerapkan IQR untuk menghilangkan outliers 
numeric_columns = df_dataset.select_dtypes(include=[np.number]).columns
#hitung Q1 dan Q3 dan IQR pada kolom numerik
Q1 = df_dataset[numeric_columns].quantile(0.25)
Q3 = df_dataset[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
#Membuat Filter untuk menghapus baris yang mengandung outliers di kolom numerik
filters_outliers = ~((df_dataset[numeric_columns] < (Q1 - 1.5 * IQR)) | 
                     (df_dataset[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
#Terapkan filter ke dataset asli
df_dataset = df_dataset[filters_outliers]
#cek ukuran dataset setelah menghapus outliers
df_dataset = df_dataset.reset_index(drop=True)
df_dataset.shape
df_dataset.head()


# ***Univariate Analysis***

# In[598]:


#Melakukan Univariate Analysis pada semua kolom
df_dataset.hist(bins=50, figsize=(20, 15))
plt.show()


# ***Exploratory Data Analysis - Multivariate Analysis***

# In[599]:


# MElakuakn pengecekan hubungan antar variabel
sns.pairplot(df_dataset, diag_kind='kde')


# In[600]:


# Menggunakan heatmap untuk melihat korelasi antar variabel
plt.figure(figsize=(12, 8))
correlation_matrix = df_dataset.corr().round(2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# ***Data Preparation***

# In[601]:


# Kita akan melakukan preparasi data untuk model machine learning nantinya dengan tahapan REduksi Dimensi, Pembagian Data,Standarasi data
# Melakukan reduksi dimensi dengan PCA
pca = PCA(n_components=5,random_state=42)
pca.fit(df_dataset[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']])
princ_components = pca.transform(df_dataset[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']])
pca.explained_variance_ratio_.round(3)


# In[602]:


#Mereduksi dimensi menjadi 2 komponen utama
pca = PCA(n_components=2, random_state=42)
pca.fit(df_dataset[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']])
df_dataset[['dimension1', 'dimension2']] = pca.transform(df_dataset[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']])
df_dataset.drop(['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age'], axis=1, inplace=True)
df_dataset.head()


# In[603]:


#split data menjadi data latih dan data uji
x = df_dataset.drop('Outcome', axis=1)
y = df_dataset['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Cek ukuran data latih dan data uji
print(f"Ukuran data latih: {x_train.shape}, Ukuran data uji: {x_test.shape}")


# In[604]:


# Melakukan Standarisasi data
# Scaling semua kolom numerik (setelah PCA)
numeric_cols = x_train.select_dtypes(include=[np.number]).columns
scaler_all = StandardScaler()
scaler_all.fit(x_train[numeric_cols])
x_train[numeric_cols] = scaler_all.transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler_all.transform(x_test[numeric_cols])


# ***Model Development***

# ***MEALAKUKAN IMBALANCING***

# In[605]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)


# In[606]:


#Modelling dengan KNN, Random Forest, dan Boosting
knn_bal = KNeighborsClassifier(n_neighbors=12)
knn_bal.fit(x_train_bal, y_train_bal)

rf_bal = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
rf_bal.fit(x_train_bal, y_train_bal)

boosting_bal = AdaBoostClassifier(learning_rate=0.1, random_state=42, algorithm="SAMME")
boosting_bal.fit(x_train_bal, y_train_bal)
#Meyimpan model ke dalam dictionary
models_bal = {
    'KNN_bal': knn_bal,
    'Random Forest_bal': rf_bal,
    'Boosting_bal': boosting_bal
}


# ***MELAKUKAN EVALUASI MODEL***

# In[607]:


# --- Evaluasi Model Balancing ---
from sklearn.metrics import classification_report, mean_squared_error

for model_name, model in models_bal.items():
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, model.predict(x_test)))


# In[608]:


# --- Threshold Adjustment pada Random Forest Balancing ---
y_proba_rf_bal = rf_bal.predict_proba(x_test)[:, 1]
# Contoh threshold 0.3
y_pred_rf_bal_thresh_0_5 = (y_proba_rf_bal >= 0.5).astype(int)
print("Classification Report for Random Forest_bal (threshold=0.5):")
print(classification_report(y_test, y_pred_rf_bal_thresh_0_5))


# In[609]:


# --- Visualisasi MSE Model Balancing & Threshold Adjustment ---
mse_bal = pd.DataFrame(columns=['train_mse', 'test_mse'],
                       index=['KNN_bal', 'Random Forest_bal', 'Boosting_bal'])
for model_name, model in models_bal.items():
    mse_bal.loc[model_name, 'train_mse'] = mean_squared_error(y_train, model.predict(x_train))
    mse_bal.loc[model_name, 'test_mse'] = mean_squared_error(y_test, model.predict(x_test))

mse_bal.loc['Random Forest (thresh=0.5)', 'test_mse'] = mean_squared_error(y_test, y_pred_rf_bal_thresh_0_5)


# In[610]:


fig, ax = plt.subplots()
mse_bal['test_mse'].sort_values(ascending=False).plot(kind='barh', ax=ax, zorder=3, color='skyblue')
ax.set_title('Test MSE Model Balancing & Threshold Adjustment')
ax.set_xlabel('Test MSE')
ax.grid(zorder=0)
plt.show()


# In[611]:


# --- Prediksi untuk semua Data pada Data Uji ---
prediksi = x_test.copy()
pred_dict_bal = {'y_true': y_test.values}
for name, model in models_bal.items():
    pred_dict_bal['prediksi_' + name] = model.predict(prediksi).round(1)
# Threshold adjustment pada Random Forest balancing (threshold 0.3)
y_proba_rf_bal_10 = rf_bal.predict_proba(prediksi)[:, 1]
pred_dict_bal['prediksi_RF_bal_thresh_0.5'] = (y_proba_rf_bal_10 >= 0.5).astype(int)

hasil_prediksi_10 = pd.DataFrame(pred_dict_bal, index=prediksi.index)
print("Prediksi 10 data pertama pada data uji:")
print(hasil_prediksi_10)

# --- Hitung Rasio Akurasi pada 10 Data Pertama ---
def calculate_accuracy_ratio(predictions, actual):
    correct_predictions = (predictions == actual).sum()
    total_predictions = len(predictions)
    accuracy_ratio = (correct_predictions / total_predictions) * 100
    return accuracy_ratio

for col in hasil_prediksi_10.columns[1:]:
    acc = calculate_accuracy_ratio(hasil_prediksi_10[col], hasil_prediksi_10['y_true'])
    print(f"Akurasi {col}: {acc:.2f}%")

