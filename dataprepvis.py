import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
df = pd.read_csv('customer_data_cleaned.csv')

# Menampilkan informasi dasar
print(df.head())
print(df.describe())
print(df.info())

# Mengatur gaya visualisasi seaborn
sns.set(style="whitegrid")

# 1. Histogram untuk distribusi 'Age'
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=10, kde=True, color='skyblue') # 'palette' dihapus
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Histogram untuk distribusi 'AnnualIncome'
plt.figure(figsize=(8, 6))
sns.histplot(df['AnnualIncome'], bins=10, kde=True, color='lightgreen') # 'palette' dihapus
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()

# 3. Histogram untuk distribusi 'SpendingScore'
plt.figure(figsize=(8, 6))
sns.histplot(df['SpendingScore'], bins=10, kde=True, color='salmon') # 'palette' dihapus
plt.title('Distribution of Spending Score')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.show()

# 4. Countplot untuk melihat distribusi 'Gender'
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df, palette='pastel')  # 'hue' tidak digunakan, maka 'palette' akan diabaikan
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 5. Scatterplot untuk melihat hubungan antara 'AnnualIncome' dan 'SpendingScore'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Gender', data=df, s=100) # 'palette' akan digunakan
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Gender')
plt.show()

# 6. Boxplot untuk melihat distribusi 'SpendingScore' berdasarkan 'Gender'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='SpendingScore', data=df, palette='pastel') # 'hue' dihapus, karena hanya satu kategori yang digunakan
plt.title('Spending Score Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()

# 7. Pairplot untuk melihat hubungan antar variabel numerik
plt.figure(figsize=(12, 10))
sns.pairplot(df[['Age', 'AnnualIncome', 'SpendingScore']], kind='scatter', diag_kind='kde', markers='+')
plt.suptitle('Pairplot of Age, Annual Income, and Spending Score', y=1.02)
plt.show()
