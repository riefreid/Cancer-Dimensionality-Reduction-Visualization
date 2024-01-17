# Cancer Dimensionality Reduction Visualization

This repository contains Python code for visualizing cancer data using dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE). The code provides insights into the distribution of cancer types in a reduced feature space.

## Contents

1. **PC1 & PC2 Visualization**
   - Scatter plot visualizing the data in the first two Principal Components.
   - Color-coded points for different cancer types.

2. **Violin Plots for PC1 & PC2**
   - Violin plots illustrating the distribution of data along PC1 and PC2.

3. **t-SNE 1 and t-SNE 2 Visualization**
   - Scatter plot visualizing the data in the first two t-SNE components.
   - Color-coded points for different cancer types.

4. **Violin Plots for t-SNE 1 and t-SNE 2**
   - Violin plots illustrating the distribution of data along t-SNE 1 and t-SNE 2.

## Usage

1. Clone the repository:
   git clone https://github.com/your-username/Cancer-Dimensionality-Reduction-Visualization.git
   
2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the Python script:
   python visualize_cancer_data.py

## Dependencies
- pandas
- seaborn
- matplotlib
- scikit-learn
