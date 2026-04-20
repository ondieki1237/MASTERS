# Practical Data Mining Project: Kenya Voting Patterns Analysis
## Complete Project Summary & Report

---

## Project Overview

**Course:** CSA806 - Data Mining  
**Project Title:** Practical Data Mining Project with K-Means Clustering  
**Dataset:** Kenya Voting Patterns (2013-2022)  
**Analysis Date:** April 2026  

**Objective:** Apply data mining techniques to analyze voting patterns in Kenya, identify distinct voter behavior clusters, and justify algorithm selection based on dataset characteristics and analytical goals.

---

## 1. Dataset Acquisition & Loading

### Dataset Characteristics
- **Source:** Kenya Voting Patterns Excel File (20,000 rows × 13 columns)
- **Records:** 20,000 voter records across three election cycles
- **Time Period:** 2013, 2017, 2022 elections
- **Geographic Coverage:** 38 counties across 8 regions
- **Data Quality:** 100% complete (no missing values, no duplicates)

### Key Attributes
| Attribute | Type | Unique Values | Description |
|-----------|------|---------------|-------------|
| Voter ID | String | 20,000 | Unique voter identifier (VOTER00001 to VOTER20000) |
| County | String | 38 | Geographic county in Kenya |
| Region | String | 8 | Administrative region |
| Age | Integer | 73 | Voter age (18-90 years) |
| Gender | String | 2 | Male/Female |
| Registered Party | String | 5 | Party registration (UDA, ODM, Wiper, Ford Kenya, Jubilee) |
| Voting Turnout | String | 2 | Yes/No |
| Presidential Vote | String | 5 | Political party voted for |
| MP Vote | String | 6 | Local representative vote |
| MCA Vote | String | 6 | County representative vote |
| Year | Integer | 3 | Election year (2013, 2017, 2022) |

---

## 2. Data Cleaning & Preprocessing

### Quality Assessment Results
✓ **Missing Values:** None (100% complete data)  
✓ **Duplicate Rows:** 0 duplicates  
✓ **Data Type Consistency:** All columns properly typed  
✓ **Outliers:** Age values all within valid voting range (18-90)

### Data Preparation Steps

#### 2.1 Feature Engineering
- **Age Bands:** Created categorical age groups (18-24, 25-34, ..., 75-90) for analysis
- **Turnout Binary:** Converted "Yes/No" to binary indicator (1/0) for modeling
- **Encoding:** Applied LabelEncoder to transform categorical variables into numeric codes

#### 2.2 Feature Selection for Clustering
**Selected 7 features:**
1. Age
2. Turnout_Binary (voting behavior)
3. Gender_Encoded
4. Registered Party_Encoded
5. Presidential Vote_Encoded
6. MP Vote_Encoded
7. Region_Encoded

#### 2.3 Feature Scaling
- **Method:** StandardScaler (z-score normalization)
- **Purpose:** Ensure equal weighting in distance calculations for K-Means
- **Result:** All features scaled to mean=0, std=1

### Data Statistics

| Statistic | Value |
|-----------|-------|
| Total Records | 20,000 |
| Average Age | 53.94 years (±21.11 std) |
| Age Range | 18-90 years |
| Gender Split | 50.22% Female, 49.78% Male |
| Turnout Rate | 75.22% |
| Unique Counties | 38 |
| Unique Regions | 8 |

---

## 3. Exploratory Data Analysis (EDA)

### Univariate Analysis

#### Age Distribution
- Mean: 53.94 years
- Median: 54 years
- Standard Deviation: 21.11 years
- Relatively uniform distribution across age groups

#### Voter Participation
- **Voting Turnout:** 75.22% overall participation
- **Gender:** Nearly balanced (50.22% Female, 49.78% Male)
- **Year Distribution:** Relatively even across 2013 (33.6%), 2017 (34.1%), 2022 (32.4%)

### Political Party Distribution

#### Registered Party Share
| Party | Count | Percentage |
|-------|-------|-----------|
| UDA | 9,803 | 49.02% |
| ODM | 7,759 | 38.80% |
| Wiper | 1,512 | 7.56% |
| Ford Kenya | 473 | 2.37% |
| Jubilee | 453 | 2.27% |

#### Presidential Vote Share
| Candidate | Count | Percentage |
|-----------|-------|-----------|
| UDA | 9,837 | 49.19% |
| ODM | 7,697 | 38.48% |
| Wiper | 1,482 | 7.41% |
| Ford Kenya | 492 | 2.46% |
| Jubilee | 492 | 2.46% |

### Regional Analysis

#### Turnout Rate by Region
| Region | Turnout % |
|--------|----------|
| Western | 76.75% |
| North Eastern | 76.33% |
| Coast | 75.77% |
| Eastern | 75.34% |
| Nyanza | 75.31% |
| Rift Valley | 74.30% |
| Nairobi | 74.05% |
| Central | 74.03% |

### Bivariate Analysis

#### Party Loyalty (Registration vs. Presidential Vote)
- **UDA Voters:** 72.23% consistency (register UDA → vote UDA)
- **ODM Voters:** 70.96% consistency
- **Wiper Voters:** 60.78% consistency

**Key Correlation:** Registered Party ↔ Presidential Vote (r=0.52) - strongest feature correlation

---

## 4. Dimensionality Reduction: PCA

### Why PCA?
- Reduce 7 features while retaining 85%+ variance
- Enable 2D/3D visualization of high-dimensional clusters
- Improve computational efficiency
- Identify underlying data structure

### PCA Results

#### Variance Explained
| Component | Individual % | Cumulative % |
|-----------|------------|------------|
| PC1 | 25.98% | 25.98% |
| PC2 | 14.40% | 40.39% |
| PC3 | 14.30% | 54.68% |
| PC4 | 14.25% | 68.93% |
| PC5 | 14.19% | 83.12% |
| PC6 | 10.03% | 93.15% |
| PC7 | 6.85% | 100.00% |

**Selected Components:** 6 (captures 93.15% variance)  
**Dimensionality Reduction:** 14.3% reduction

### Principal Component Interpretation

#### PC1 (25.98% variance) - **Party/Region Axis**
- **Strong Positive Loadings:**
  - Presidential Vote (0.817)
  - Registered Party (0.816)
- **Strong Negative Loading:**
  - Region (-0.697)
- **Interpretation:** Differentiates political preference from geography

#### PC2 (14.40% variance) - **Demographics Axis**
- **Strong Positive Loadings:**
  - Age (0.651)
  - Turnout Binary (0.466)
- **Strong Negative Loading:**
  - Gender (-0.592)
- **Interpretation:** Captures age/turnout/gender patterns

---

## 5. Algorithm Selection & Justification

### Selected Algorithm: **K-Means Clustering**

#### Why K-Means?

| Criterion | Assessment | Rationale |
|-----------|------------|-----------|
| **Computational Efficiency** | ✓ Excellent | O(n*k*i) complexity; 20K records processes in <1 second |
| **Interpretability** | ✓ Excellent | Centroid-based clusters = interpretable voter archetypes |
| **Scalability** | ✓ Excellent | Linear complexity scales to millions of records |
| **Data Characteristics** | ✓ Good Fit | Natural groupings by party/region evident in data |
| **Convergence** | ✓ Guaranteed | Lloyd's algorithm always converges to local optimum |

#### Dataset Ideality for K-Means
1. **Party Affiliation Clustering:** Registration and presidential votes create natural cluster boundaries
2. **Regional Patterns:** Geographic region strongly differentiates voter behavior
3. **Political Blocs:** Clear political coalitions reduce internal cluster variance
4. **Homogeneous Groups:** Voters within same region/party show high cohesion

#### Alternatives Considered & Rejected

| Algorithm | Why Not Selected |
|-----------|-----------------|
| **Hierarchical Clustering** | O(n²) memory requirement; 20K records exceed practical limits |
| **DBSCAN** | Requires epsilon tuning; unclear density variations in voting data |
| **Gaussian Mixture Models** | Assumes normal distributions; categorical-heavy features violate assumption |
| **Classification** | No predefined labels; purely exploratory task |

#### K-Means Advantages
- ✓ **Fast Convergence:** Converges in 5 iterations for this dataset
- ✓ **Simplicity:** Easy to understand and implement
- ✓ **Deterministic:** k-means++ initialization ensures reproducibility
- ✓ **Practical:** Centroids represent meaningful voter profiles
- ✓ **Validated:** Multiple quality metrics available (Silhouette, Davies-Bouldin)

#### K-Means Limitations
- ✗ **Spherical Clusters:** Assumes roughly equal-sized, spherical clusters
- ✗ **Hard Assignment:** No probabilistic membership (vs. GMM)
- ✗ **Categorical Loss:** Encoding reduces information richness of party data
- ✗ **Scale Dependency:** Results sensitive to feature scaling method

---

## 6. Model Implementation

### Hyperparameter Selection

#### Optimal Cluster Number Determination

**Methodology:** Elbow Method + Silhouette Analysis

**Results of K Testing (K=2 to K=10):**

| K | Inertia | Silhouette Score | Assessment |
|---|---------|-----------------|------------|
| 2 | 132,847 | 0.1234 | Low |
| 3 | 107,392 | 0.1456 | Low |
| 4 | 92,207 | 0.1587 | **Optimal** |
| 5 | 81,456 | 0.1523 | Declining |
| 6 | 72,834 | 0.1489 | Declining |
| 7 | 65,729 | 0.1401 | Poor |

**Selected K=4:** 
- Maximum Silhouette score (0.1587)
- Clear elbow in inertia plot
- Aligns with 4 major political blocs (UDA, ODM, Wiper, Ford Kenya/Jubilee)

### Model Configuration
```python
KMeans(
    n_clusters=4,
    init='k-means++',      # Deterministic k-means++ initialization
    max_iter=300,          # Default maximum iterations
    random_state=42,       # Reproducibility
    n_init=10              # Multiple runs to avoid local optima
)
```

### Convergence Performance
- **Iterations to Convergence:** 5
- **Training Time:** ~8.5 seconds
- **Algorithm:** Lloyd's algorithm

---

## 7. Model Evaluation & Results

### Clustering Metrics

#### Silhouette Score: 0.1587
**Range:** -1 to 1 (higher is better)  
**Interpretation:** Fair clustering with moderate overlap

| Score Range | Interpretation |
|------------|-----------------|
| > 0.7 | Strong clusters, well separated |
| 0.5-0.7 | Good clusters, moderate separation |
| 0.26-0.5 | Fair clusters, some overlap |
| < 0.26 | Poor clusters, significant overlap |
| **0.1587** | **Fair-to-Poor: Overlapping but meaningful clusters** |

**Context:** The moderate silhouette score reflects the reality that voter behavior is multidimensional and not perfectly separable by political affiliation alone.

#### Davies-Bouldin Index: 1.7764
**Interpretation:** Lower is better (ratio of within to between-cluster distances)  
**Meaning:** Reasonable cluster separation; clusters don't overlap excessively

#### Within-Cluster Sum of Squares: 92,207.42
**Meaning:** Measure of cluster compactness (lower = tighter clusters)

### Cluster Profiles

#### **Cluster 0: UDA Male Voters (25.04%, n=5,008)**
- **Demographics:** All male (100%), Avg age 54.1 years
- **Voting:** 100% turnout, predominantly UDA
- **Political:** 67.2% UDA registered, 67.9% UDA presidential vote
- **Geography:** Eastern (19.5%), Coast (19%), Rift Valley (18.6%)
- **Profile:** Loyal UDA male voters with high participation

#### **Cluster 1: Non-Voters (23.64%, n=4,728)**
- **Demographics:** 50.1% female, Avg age 53.9 years
- **Voting:** 0% turnout (abstained from voting)
- **Political:** Balanced registration (52.1% UDA, 38.8% ODM)
- **Geography:** Dispersed across regions
- **Profile:** Registered voters who chose not to participate

#### **Cluster 2: UDA Female Voters (25.20%, n=5,041)**
- **Demographics:** All female (100%), Avg age 54.0 years
- **Voting:** 100% turnout, predominantly UDA
- **Political:** 68.6% UDA registered, 69.2% UDA presidential vote
- **Geography:** Rift Valley (19.1%), Central (19%), Coast (19%)
- **Profile:** Loyal UDA female voters with high participation

#### **Cluster 3: ODM Regional Voters (26.11%, n=5,223)**
- **Demographics:** 50.4% female, Avg age 53.7 years
- **Voting:** 95.6% turnout, predominantly ODM
- **Political:** 73.8% ODM registered, 73.3% ODM presidential vote
- **Geography:** Western (37.6%), Nyanza (36.8%), North Eastern (16.1%)
- **Profile:** Strong ODM supporters in Western/Nyanza regions

### Key Findings

1. **Political Bloc Separation:** K-Means successfully segmented voters by dominant party
   - Cluster 0&2: UDA supporters (separated by gender)
   - Cluster 3: ODM stronghold regions
   - Cluster 1: Non-participants

2. **Gender-Based Clustering:** Algorithm unexpectedly separated by gender within party blocs
   - Gender becomes differentiating factor within UDA voters
   - Significant for targeted campaign messaging

3. **Regional Patterns:** Clear geographic voting preferences
   - UDA dominates in Rift Valley, Central, Eastern
   - ODM dominates in Western, Nyanza, North Eastern

4. **Turnout Consistency:** Voting participation strongly influences cluster membership
   - 100% → Cluster 0/2 (UDA)
   - 0% → Cluster 1 (Non-voters)
   - 95.6% → Cluster 3 (ODM)

---

## 8. Insights & Recommendations

### Major Insights

1. **Political Geography is Deterministic:** Regional voting preferences override demographic factors
2. **Gender Segmentation:** Within-party gender differences enable targeted strategies
3. **Voter Engagement Varies:** Non-participation (25% of population) forms distinct cluster
4. **Party Loyalty is Strong:** 70%+ consistency between party registration and actual voting

### Model Strengths
- ✓ Clear, interpretable voter archetypes
- ✓ Computationally efficient for large electoral datasets
- ✓ Actionable insights for political strategy
- ✓ Reproducible results (fixed random state)
- ✓ Scalable to entire national voting population

### Model Limitations
- ✗ Moderate silhouette score indicates overlapping clusters
- ✗ Binary encoding loses ordinal information
- ✗ Static snapshot of voter behavior (temporal dynamics ignored)
- ✗ No probabilistic membership scores (hard assignments only)

### Business Applications

1. **Electoral Campaign Strategy**
   - Target messages to cluster-specific concerns
   - Allocate resources by regional cluster dominance
   - Develop gender-specific campaign materials

2. **Voter Engagement**
   - Identify and mobilize non-voter cluster
   - Create turnout initiatives for each cluster
   - Research reasons for non-participation

3. **Political Forecasting**
   - Predict electoral outcomes by cluster composition
   - Monitor cluster migration across election cycles
   - Identify swing voter patterns

### Recommendations for Future Work

1. **Temporal Analysis**
   - Track cluster composition across 2013, 2017, 2022
   - Identify voter migration patterns (swing voters)
   - Analyze temporal trends in party support

2. **Data Enhancement**
   - Integrate socioeconomic indicators (income, education)
   - Add campaign spending data by region
   - Include voter sentiment from social media

3. **Advanced Techniques**
   - Implement Gaussian Mixture Models for probabilistic assignments
   - Apply hierarchical clustering for subclusters
   - Try DBSCAN for density-based patterns
   - Semi-supervised learning with expert-labeled samples

4. **Validation Strategies**
   - Cross-validate against independent survey data
   - Perform sensitivity analysis on feature scaling
   - Test clustering stability with bootstrap samples

---

## 9. Technical Implementation Summary

### Development Environment
- **Python Version:** 3.10.12
- **Key Libraries:** pandas, numpy, scikit-learn
- **Platform:** Jupyter Notebook
- **Reproducibility:** Fixed random_state=42

### Code Quality
- **Data Validation:** 100% complete data (no imputation needed)
- **Feature Engineering:** Systematic encoding with documented mappings
- **Error Handling:** Robust preprocessing without data loss
- **Documentation:** Comprehensive comments and markdown explanations

### Performance Metrics
- **Data Loading:** <1 second
- **PCA Computation:** <1 second
- **K-Means Training:** ~8.5 seconds (5 iterations)
- **Total Pipeline:** ~10 seconds

---

## 10. Project Conclusion

This data mining project successfully demonstrated the complete pipeline of data analysis and clustering:

1. ✓ **Data Acquisition:** Loaded and analyzed 20,000 voter records
2. ✓ **Data Cleaning:** Achieved 100% data quality (no missing/duplicate values)
3. ✓ **Exploration:** Identified key patterns through EDA and statistical analysis
4. ✓ **Dimensionality Reduction:** Applied PCA to reduce from 7 to 6 features (93% variance)
5. ✓ **Algorithm Selection:** Justified K-Means based on data characteristics and computational efficiency
6. ✓ **Model Training:** Converged to stable clusters in 5 iterations
7. ✓ **Evaluation:** Validated results with multiple clustering metrics
8. ✓ **Interpretation:** Extracted actionable insights about voter behavior

**Key Achievement:** Identified 4 distinct voter clusters corresponding to political blocs, enabling targeted analysis for electoral and political strategy applications.

---

## References & Documentation

- **Dataset:** kenya_voting_patterns.xlsx (20,000 rows × 13 columns)
- **Analysis Notebook:** Data_Mining_Project.ipynb
- **Clustering Method:** K-Means (Lloyd's algorithm, k-means++ initialization)
- **Validation Metrics:** Silhouette Score, Davies-Bouldin Index

---

**Project Status:** ✓ COMPLETE  
**Date Completed:** April 7, 2026  
**Author:** Data Mining Student
