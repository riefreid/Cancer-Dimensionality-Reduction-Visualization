# Global imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# TASK 1 | PC1 & PC2 Visualization
# ________________________________
# Loading csv data into a pandas dataframe & saving the column of class labels
df = pd.read_csv('lncRNA_5_Cancers.csv')
cancer_types_column = df.columns[-1]

# Clean the data of all non-numeric columns
numeric_data = df.iloc[:, 1:-1]

# Performing PCA with 2 components
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numeric_data)

# Extract the actual cancer types data from the specified column
cancer_types = df[cancer_types_column]

# Create a scatter plot of the reduced data
colors = {'KIRC': 'red', 'LUAD': 'blue', 'LUSC': 'green', 'PRAD': 'orange', 'THCA': 'purple'}
plt.figure(figsize=(8, 6))
for cancer_type in colors.keys():
    indices = cancer_types == cancer_type
    plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=cancer_type, c=colors[cancer_type], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Figure 1 | PCA Visualization of 5 Cancer Types')
plt.legend()
# plt.show()


# TASK 2 | Violin Plot for PC1 & PC2
# __________________________________
# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=reduced_data, columns=['PC1', 'PC2'])

# Create two violin plots for PC1 and PC2
plt.figure(figsize=(12, 6))  # Adjust the figure size if needed
plt.subplot(1, 2, 1)
sns.violinplot(y='PC1', data=pca_df, color='skyblue')
plt.title('Figure 2a | Violin Plot of PC1')
plt.subplot(1, 2, 2)
sns.violinplot(y='PC2', data=pca_df, color='salmon')
plt.title('Figure 2b | Violin Plot of PC2')
plt.tight_layout()  # Ensures proper spacing between subplots
# plt.show()

# TASK 3 | t-SNE 1 and t-SNE 2 Visualization
# __________________________________________
# Initialize t-SNE with two components
tsne = TSNE(n_components=2, random_state=42)
# Fit t-SNE on the numeric data and get the transformed data
reduced_data = tsne.fit_transform(numeric_data)
tsne_df = pd.DataFrame(data=reduced_data, columns=['t-SNE 1', 't-SNE 2'])
# Create a scatter plot of the reduced data with t-SNE components
plt.figure(figsize=(8, 6))
for cancer_type in colors.keys():
    indices = cancer_types == cancer_type
    plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=cancer_type, c=colors[cancer_type], alpha=0.5)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Figure 3 | t-SNE Visualization of Cancer Data')
plt.legend(title='Cancer Type', loc='best')  # Add a legend with the cancer types
plt.show()


# TASK 4 | Violin Plot for t-SNE 1 and t-SNE 2
# ____________________________________________
# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(data=reduced_data, columns=['t-SNE 1', 't-SNE 2'])

# Create two violin plots for t-SNE 1 and t-SNE 2
plt.figure(figsize=(12, 6))  # Adjust the figure size if needed

plt.subplot(1, 2, 1)
sns.violinplot(x='t-SNE 1', data=tsne_df, color='skyblue')
plt.title('Figure 4a | Violin Plot of t-SNE 1')

plt.subplot(1, 2, 2)
sns.violinplot(x='t-SNE 2', data=tsne_df, color='salmon')
plt.title('Figure 4b | Violin Plot of t-SNE 2')

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()
