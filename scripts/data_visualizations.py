import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("data_core.csv")
df = df.drop(columns=["Fertilizer Name", "Soil Type"], axis=1)

crop_counts = df['Crop Type'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=crop_counts.values, y=crop_counts.index)
plt.title("Distribution of Crop Types")
plt.xlabel("Count")
plt.ylabel("Crop Type")
plt.tight_layout()
plt.show()

numerical_features = df.drop(columns=['Crop Type']).select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

melted = pd.melt(df, id_vars='Crop Type', value_vars=['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
plt.figure(figsize=(14, 8))
sns.boxplot(data=melted, x='value', y='Crop Type', hue='variable')
plt.title("Boxplots of Features by Crop Type")
plt.xlabel("Value")
plt.ylabel("Crop Type")
plt.legend(title='Feature')
plt.tight_layout()
plt.show()