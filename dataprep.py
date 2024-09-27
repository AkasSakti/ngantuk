import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# 1. Membaca Dataset
# Misalkan file CSV bernama 'customer_data.csv'
df = pd.read_csv("mall_customers_data.csv")

# 2. Menampilkan 5 baris pertama data
print("Data Awal:\n", df.head())

# 3. Memeriksa Nilai yang Hilang (Missing Values)
print("\nMemeriksa Nilai yang Hilang:\n", df.isnull().sum())

# 4. Mengisi Nilai yang Hilang
# Mengisi nilai yang hilang di kolom numerik dengan rata-rata (mean)
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Mengisi nilai yang hilang di kolom kategorikal dengan modus (nilai terbanyak)
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

print("\nData Setelah Mengisi Nilai Hilang:\n", df.head())

# 5. Encoding Data Kategorikal
# Mengubah kolom 'Gender' menjadi numerik menggunakan Label Encoding
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

print("\nData Setelah Encoding Kolom 'Gender':\n", df.head())

# 6. Deteksi dan Penanganan Outlier
# Menggunakan metode IQR untuk mendeteksi outlier pada kolom 'AnnualIncome'
Q1 = df['AnnualIncome'].quantile(0.25)
Q3 = df['AnnualIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menandai outlier pada kolom 'AnnualIncome'
df['Outlier_AnnualIncome'] = np.where((df['AnnualIncome'] < lower_bound) | (df['AnnualIncome'] > upper_bound), True, False)

# Menghapus outlier
df = df[df['Outlier_AnnualIncome'] == False]
df.drop(columns=['Outlier_AnnualIncome'], inplace=True)

print("\nData Setelah Menghapus Outlier di Kolom 'AnnualIncome':\n", df.head())

# 7. Normalisasi Data
# Normalisasi menggunakan Min-Max Scaler pada kolom 'AnnualIncome' dan 'SpendingScore(1-100)'
scaler = MinMaxScaler()
df[['AnnualIncome', 'SpendingScore']] = scaler.fit_transform(df[['AnnualIncome', 'SpendingScore']])

print("\nData Setelah Normalisasi:\n", df.head())

# 8. Standarisasi Data
# Standarisasi menggunakan Standard Scaler pada kolom 'Age'
scaler_standard = StandardScaler()
df['Age'] = scaler_standard.fit_transform(df[['Age']])

print("\nData Setelah Standarisasi Kolom 'Age':\n", df.head())

# 9. Menyimpan Data yang Sudah Diproses ke File Baru
df.to_csv("customer_data_cleaned.csv", index=False)
print("\nData yang Sudah Diproses Disimpan ke 'customer_data_cleaned.csv'")
