# Code Reference Guide
## Practical Python Implementation for Data Mining Project

---

## 1. Complete Data Mining Pipeline

### Full Code Structure

```python
# ============================================================================
# PRACTICAL DATA MINING PROJECT: Kenya Voting Patterns Analysis
# Complete Pipeline Implementation
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================================

def load_and_explore_data(filepath):
    """Load dataset and perform initial exploration"""
    df = pd.read_excel(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    return df

# ============================================================================
# SECTION 2: DATA CLEANING & PREPROCESSING
# ============================================================================

def prepare_data(df):
    """Clean, engineer features, and prepare for clustering"""
    
    df_clean = df.copy()
    
    # Feature engineering
    bins = [18, 25, 35, 45, 55, 65, 75, 91]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-90']
    df_clean['Age_Band'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, right=False)
    
    # Binary encoding for turnout
    df_clean['Turnout_Binary'] = (df_clean['Voting Turnout'] == 'Yes').astype(int)
    
    return df_clean

def encode_and_scale_features(df_clean):
    """Encode categorical variables and scale features"""
    
    # Define features for clustering
    features_to_cluster = ['Age', 'Turnout_Binary', 'Gender', 'Registered Party', 
                          'Presidential Vote', 'MP Vote', 'Region']
    
    df_features = df_clean[features_to_cluster].copy()
    
    # Label encoding for categorical features
    label_encoders = {}
    categorical_features = ['Gender', 'Registered Party', 'Presidential Vote', 
                           'MP Vote', 'Region']
    
    for col in categorical_features:
        le = LabelEncoder()
        df_features[f'{col}_Encoded'] = le.fit_transform(df_features[col])
        label_encoders[col] = le
    
    # Select only encoded features
    clustering_features = ['Age', 'Turnout_Binary', 'Gender_Encoded', 
                          'Registered Party_Encoded', 'Presidential Vote_Encoded', 
                          'MP Vote_Encoded', 'Region_Encoded']
    X = df_features[clustering_features].copy()
    
    # StandardScaler normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=clustering_features)
    
    return X_scaled, label_encoders, clustering_features

# ============================================================================
# SECTION 3: DIMENSIONALITY REDUCTION (PCA)
# ============================================================================

def apply_pca(X_scaled, variance_threshold=0.85):
    """Apply PCA and reduce dimensionality"""
    
    # Fit PCA with all components first to understand variance
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    
    # Apply PCA with optimal components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original features: {X_scaled.shape[1]}")
    print(f"Reduced features: {n_components}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, pca, n_components

# ============================================================================
# SECTION 4: OPTIMAL CLUSTER DETERMINATION (ELBOW METHOD)
# ============================================================================

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Determine optimal number of clusters using Elbow method"""
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        
        inertias.append(kmeans_temp.inertia_)
        score = silhouette_score(X_scaled, kmeans_temp.labels_)
        silhouette_scores.append(score)
        
        print(f"K={k}: Inertia={kmeans_temp.inertia_:.2f}, "
              f"Silhouette={score:.4f}")
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    return optimal_k, inertias, silhouette_scores

# ============================================================================
# SECTION 5: K-MEANS CLUSTERING
# ============================================================================

def train_kmeans(X_scaled, n_clusters=4):
    """Train K-Means model"""
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',      # Deterministic initialization
        max_iter=300,
        random_state=42,
        n_init=10
    )
    
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return kmeans, cluster_labels

# ============================================================================
# SECTION 6: MODEL EVALUATION
# ============================================================================

def evaluate_clustering(X_scaled, cluster_labels):
    """Calculate clustering quality metrics"""
    
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Silhouette analysis per cluster
    silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
    
    for i in range(len(np.unique(cluster_labels))):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        avg_silhouette = cluster_silhouette_vals.mean()
        print(f"Cluster {i}: {len(cluster_silhouette_vals)} samples, "
              f"Avg silhouette: {avg_silhouette:.4f}")
    
    return silhouette_avg, davies_bouldin

# ============================================================================
# SECTION 7: CLUSTER ANALYSIS
# ============================================================================

def analyze_clusters(df_clean, cluster_labels, n_clusters):
    """Characterize each cluster"""
    
    df_clean['Cluster'] = cluster_labels
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        
        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id} (n={len(cluster_data)})")
        print(f"{'='*70}")
        
        # Demographics
        print(f"Average Age: {cluster_data['Age'].mean():.1f} years")
        print(f"Female %: {(cluster_data['Gender']=='Female').mean()*100:.1f}%")
        
        # Voting behavior
        print(f"Turnout: {(cluster_data['Voting Turnout']=='Yes').mean()*100:.1f}%")
        
        # Top parties
        print(f"\nTop Registered Parties:")
        for party, count in cluster_data['Registered Party'].value_counts().head(3).items():
            print(f"  {party}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        # Top regions
        print(f"Top Regions:")
        for region, count in cluster_data['Region'].value_counts().head(3).items():
            print(f"  {region}: {count} ({count/len(cluster_data)*100:.1f}%)")

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    # Load and explore data
    df = load_and_explore_data('kenya_voting_patterns.xlsx')
    
    # Data preparation
    df_clean = prepare_data(df)
    
    # Encoding and scaling
    X_scaled, label_encoders, features = encode_and_scale_features(df_clean)
    
    # PCA dimensionality reduction
    X_pca, pca, n_components = apply_pca(X_scaled)
    
    # Find optimal K
    optimal_k, inertias, silhouette_scores = find_optimal_k(X_scaled)
    
    # Train K-Means
    kmeans, cluster_labels = train_kmeans(X_scaled, n_clusters=4)
    
    # Evaluate
    silhouette_avg, davies_bouldin = evaluate_clustering(X_scaled, cluster_labels)
    
    # Analyze clusters
    analyze_clusters(df_clean, cluster_labels, 4)
    
    print("\n✓ Analysis complete!")

# ============================================================================

if __name__ == "__main__":
    main()
```

---

## 2. Key Function Reference

### StandardScaler Implementation

```python
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
X_scaled = scaler.fit_transform(X)

# Access parameters
print(f"Mean: {scaler.mean_}")
print(f"Std: {scaler.scale_}")

# Transform new data
X_new_scaled = scaler.transform(X_new)

# Inverse transform
X_original = scaler.inverse_transform(X_scaled)
```

### LabelEncoder Implementation

```python
from sklearn.preprocessing import LabelEncoder

# For each categorical column
le = LabelEncoder()
df['Party_Encoded'] = le.fit_transform(df['Registered Party'])

# View mapping
print(dict(zip(le.classes_, le.transform(le.classes_))))
# Output: {'Ford Kenya': 0, 'Jubilee': 1, 'ODM': 2, 'UDA': 3, 'Wiper': 4}

# Transform new data
new_party_encoded = le.transform(['UDA', 'ODM'])

# Inverse transform
original_party = le.inverse_transform([3, 2])
```

### PCA Implementation

```python
from sklearn.decomposition import PCA

# Create PCA with number of components
pca = PCA(n_components=6, random_state=42)

# Fit and transform
X_pca = pca.fit_transform(X_scaled)

# Variance explained
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

# Components (loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Inverse transform (reconstruction)
X_reconstructed = pca.inverse_transform(X_pca)
```

### K-Means Implementation

```python
from sklearn.cluster import KMeans

# Create and fit model
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    random_state=42,
    n_init=10
)

# Fit model
kmeans.fit(X_scaled)

# Get cluster labels
labels = kmeans.labels_

# Predict for new data
new_labels = kmeans.predict(X_new_scaled)

# Get centroids
centroids = kmeans.cluster_centers_

# Inertia (sum of squared distances to nearest centroid)
print(kmeans.inertia_)

# Number of iterations until convergence
print(kmeans.n_iter_)
```

### Silhouette Score Implementation

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall silhouette score
score = silhouette_score(X_scaled, labels)

# Per-sample silhouette values
silhouette_vals = silhouette_samples(X_scaled, labels)

# Per-cluster analysis
for cluster_id in range(n_clusters):
    cluster_silhouette_vals = silhouette_vals[labels == cluster_id]
    avg = cluster_silhouette_vals.mean()
    print(f"Cluster {cluster_id}: {avg:.4f}")
```

---

## 3. Data Preparation Patterns

### Handling Missing Values

```python
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_clean = df.dropna()

# Fill missing values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill categorical with mode
df['Party'].fillna(df['Party'].mode()[0], inplace=True)
```

### Handling Outliers

```python
# Identify outliers using IQR
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

# Remove outliers
df_clean = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
```

### Feature Engineering

```python
# Categorical binning
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 75, 91],
                          labels=['18-24', '25-34', '35-44', '45-54', 
                                  '55-64', '65-74', '75-90'])

# Binary encoding
df['High_Turnout'] = (df['Voting_Turnout'] > df['Voting_Turnout'].median()).astype(int)

# Create interaction features
df['Age_Turnout'] = df['Age'] * df['Turnout_Binary']

# One-hot encoding (alternative to label encoding)
df_encoded = pd.get_dummies(df[['Gender', 'Party']], drop_first=True)
```

---

## 4. Visualization Code Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Elbow plot
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

# Cluster distribution
plt.figure(figsize=(8, 6))
cluster_sizes = np.bincount(labels)
plt.bar(range(len(cluster_sizes)), cluster_sizes)
plt.xlabel('Cluster ID')
plt.ylabel('Number of Samples')
plt.title('Cluster Size Distribution')
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Silhouette plot (reference)
from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(X_scaled, labels)
# [Visualization code for silhouette diagram]
```

---

## 5. Model Persistence (Save/Load)

```python
import joblib

# Save trained model
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')

# Load trained model
kmeans_loaded = joblib.load('kmeans_model.pkl')
scaler_loaded = joblib.load('scaler.pkl')
pca_loaded = joblib.load('pca_model.pkl')

# Use loaded model for predictions
new_clusters = kmeans_loaded.predict(X_new_scaled)
```

---

## 6. Cross-Validation for Clustering Stability

```python
from sklearn.model_selection import KFold

def evaluate_clustering_stability(X, n_clusters=4, n_splits=5):
    """Assess stability of clustering across splits"""
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # Train on fold
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_train)
        
        # Evaluate on fold
        test_labels = kmeans.predict(X_test)
        silhouette_scores.append(silhouette_score(X_test, test_labels))
        davies_bouldin_scores.append(davies_bouldin_score(X_test, test_labels))
    
    print(f"Silhouette: {np.mean(silhouette_scores):.4f} (+/- {np.std(silhouette_scores):.4f})")
    print(f"Davies-Bouldin: {np.mean(davies_bouldin_scores):.4f} (+/- {np.std(davies_bouldin_scores):.4f})")
    
    return silhouette_scores, davies_bouldin_scores
```

---

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Note: GridSearchCV doesn't directly support K-Means (unsupervised)
# Manual grid search instead:

def grid_search_kmeans(X, k_values, n_trials=10):
    """Manual grid search over k values"""
    
    results = {}
    
    for k in k_values:
        silhouette_list = []
        davies_bouldin_list = []
        
        for _ in range(n_trials):
            kmeans = KMeans(n_clusters=k, n_init=10)
            labels = kmeans.fit_predict(X)
            
            silhouette_list.append(silhouette_score(X, labels))
            davies_bouldin_list.append(davies_bouldin_score(X, labels))
        
        results[k] = {
            'silhouette_mean': np.mean(silhouette_list),
            'davies_bouldin_mean': np.mean(davies_bouldin_list),
            'silhouette_std': np.std(silhouette_list),
            'davies_bouldin_std': np.std(davies_bouldin_list)
        }
    
    return results
```

---

## 8. Performance Profiling

```python
import time

# Time individual steps
start = time.time()
X_scaled = scaler.fit_transform(X)
print(f"Scaling time: {time.time() - start:.2f}s")

start = time.time()
X_pca = pca.fit_transform(X_scaled)
print(f"PCA time: {time.time() - start:.2f}s")

start = time.time()
kmeans = KMeans(n_clusters=4, n_init=10).fit(X_scaled)
print(f"K-Means training time: {time.time() - start:.2f}s")

# Memory usage
import sys
print(f"X memory: {sys.getsizeof(X) / 1024 / 1024:.2f} MB")
print(f"X_scaled memory: {sys.getsizeof(X_scaled) / 1024 / 1024:.2f} MB")
```

---

## Quick Reference Card

```python
# Data Loading
df = pd.read_excel('file.xlsx')

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Evaluate
silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)

# Export Results
df['Cluster'] = labels
df.to_csv('results_with_clusters.csv')
```

---

For full working notebook, see: `Data_Mining_Project.ipynb`
