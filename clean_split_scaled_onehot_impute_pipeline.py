#!/usr/bin/env python
# coding: utf-8


##################### Data Cleaning ########################################
import pandas as pd
import numpy as np
import os

# Import MIMIC and ZDYY data
mimic_d1 = pd.read_csv('MIMIC_D1')
zdyy_d1 = pd.read_csv('ZDYY_D1.csv')

## D1 data addition and deletion: Delete records with missing critical data where sofa=0, gender, age, hospital status, ICU status are missing
dfmimic = mimic_d1[(mimic_d1['sofa'].notna()) & (mimic_d1['sofa'] != 0) & (mimic_d1['status_hosp'].notna()) & (mimic_d1['status_icu'].notna()) & (mimic_d1['age'].notna()) & (mimic_d1['gender'].notna())]
dfzdyy = zdyy_d1[(zdyy_d1['sofa'].notna()) & (zdyy_d1['sofa'] != 0) & (zdyy_d1['status_hosp'].notna()) & (zdyy_d1['status_icu'].notna()) & (zdyy_d1['age'].notna()) & (zdyy_d1['gender'].notna())]
print(dfmimic[['status_icu', 'age']].describe())
print(dfzdyy[['status_icu', 'age']].describe())

# Data type conversion for MIMIC data
print(list(dfmimic.dtypes))
object_columns = dfmimic.select_dtypes(include='object')
print(object_columns)
dfmimic[object_columns.columns] = object_columns.apply(pd.to_numeric, errors='coerce')
print(dfmimic)

## MIMIC data
# Duplicate value identification
duplicates = dfmimic['subject_id'].duplicated()
print("Duplicate markers in subject_id column:")
print(duplicates)

# Remove duplicate values (no duplicates found in this dataset)
dfmimic1 = dfmimic.drop_duplicates(subset='subject_id', keep='first')
print("\nData after removing duplicates:")
print(dfmimic1)

# ZDYY data
# Duplicate value identification
duplicates = dfzdyy['subject_id'].duplicated()
print("Duplicate markers in subject_id column:")
print(duplicates)

# Remove duplicate values (no duplicates found in this dataset)
dfzdyy1 = dfzdyy.drop_duplicates(subset='subject_id', keep='first')
print("\nData after removing duplicates:")
print(dfzdyy1)


########################### Training, Validation, and Test Set Standardization ##################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Internal dataset
X_mimic = pd.read_csv('MIMIC.csv')
y_mimic = X_mimic['SIC_early_happen']
del X_mimic['SIC_early_happen']
del X_mimic['sofa']
del X_mimic['sofa_coag']
del X_mimic['sofa_cns']
del X_mimic['SIC_score']
# External dataset
X_zdyy = pd.read_csv('ZDYY.csv')
y_zdyy = X_zdyy['SIC_D3']
del X_zdyy['SIC_D3']

# Delete missing values in y, and remove corresponding rows in X
missing_indices_mimic = y_mimic[y_mimic.isnull()].index
X_mimic = X_mimic.drop(index=missing_indices_mimic)
y_mimic = y_mimic.drop(index=missing_indices_mimic)

missing_indices_zdyy = y_zdyy[y_zdyy.isnull()].index
X_zdyy = X_zdyy.drop(index=missing_indices_zdyy)
y_zdyy = y_zdyy.drop(index=missing_indices_zdyy)


## Data Preprocessing
# One-hot encoding for binary categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
categorical_features = [
    'status_hosp', 'status_icu', 'gender', 'Vasoactive_drug_use', 
    'mech_vent', 'lung_infection', 'abdominal_infection', 'cns_infection', 
    'urinary_infection', 'soft_tissue_infection', 'hypertension', 'diabetes', 
    'chronic_kidney_disease', 'chronic_pulmonary_disease', 'chronic_liver_disease', 
    'cerebrovascular_disease', 'crrt_status'
] # All binary categorical variables requiring encoding
def onehot(categorical_features, X):
    available_features = [feature for feature in categorical_features if feature in X.columns]
    if available_features:
        column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), available_features),  
            ],
            remainder='passthrough'
        )
        X_transformed = column_transformer.fit_transform(X)
        new_columns = column_transformer.transformers_[0][1].get_feature_names_out(available_features)
        non_categorical_columns = [col for col in X.columns if col not in available_features]
        new_column_names = list(new_columns) + non_categorical_columns
        X_transformed_df = pd.DataFrame(X_transformed, columns=new_column_names)
        
        # Convert one-hot encoded columns to uint8 type
        for col in new_columns:
            X_transformed_df[col] = X_transformed_df[col].astype('uint8')
        
        return X_transformed_df
    else:
        return X
        
X_mimic_oh = onehot(categorical_features, X_mimic)
X_zdyy_oh = onehot(categorical_features, X_zdyy)


print(X_mimic_oh['d2d'])
# Custom placeholder list, can be adjusted based on actual data
special_values = ["", "NULL", "N/A", "unknown", "not available"]  # Example placeholders

# Replace special values with NaN
X_mimic_oh_1 = X_mimic_oh.replace(special_values, np.nan)
X_zdyy_oh_1 = X_zdyy_oh.replace(special_values, np.nan)
print(X_mimic_oh_1.columns)


## Standardize all tables

# Remove id column (needed in previous data processing)
X_mimic_oh_1.columns = X_mimic_oh_1.columns.str.strip()  # Trim leading/trailing spaces in column names
print(X_mimic_oh_1.columns)
X_zdyy_oh_1.columns = X_zdyy_oh_1.columns.str.strip()  # Trim leading/trailing spaces in column names
print(X_zdyy_oh_1.columns)


id = X_mimic_oh_1.loc[:, 'Unnamed: 0':'hadm_id'].columns
X_mimic_oh_2 = X_mimic_oh_1.drop(columns=id, inplace=False)

# Remove columns with missing values exceeding 35% in the internal dataset
mvrfin = X_mimic_oh_2.apply(lambda x: sum(x.isnull())/len(x), axis = 0)
mvrfin_df = pd.DataFrame(mvrfin).reset_index()
mvrfin_df.columns = ['Column', 'Missing Value Ratio']
output = mvrfin_df[mvrfin_df['Missing Value Ratio'] > 0.35]['Column']
X_mimic_1 = X_mimic_oh_2.drop(columns = output)
common_columns = X_mimic_1.columns.intersection(X_zdyy_oh_1.columns)
X_mimic_2 = X_mimic_1[common_columns]
X_zdyy_2 = X_zdyy_oh_1[common_columns]
print("X_mimic_2 columns:")
print(X_mimic_2.columns)
print(X_mimic_2)
print(X_zdyy_2)



#################################### Characteristics of Missing Values ####################################
# Check if data is MCAR (Missing Completely at Random)


import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings

def little_mcar_test(data, alpha=0.05, bootstrap=False, n_simulations=1000):
    # Check input validity
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    if len(data) < 10:
        warnings.warn("Small sample size may lead to unreliable test", UserWarning)
    
    # Handle duplicate column names
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Generate missing pattern hash values
    missing = data.isnull()
    patterns = missing.apply(lambda x: hash(tuple(x)), axis=1)
    pattern_counts = patterns.value_counts().values
    
    # Calculate expected frequencies
    n, p = data.shape
    col_missing_rates = missing.mean()
    expected = n * np.prod(col_missing_rates)
    
    # Automatically select test method
    if bootstrap or np.any(expected < 5):
        # Perform Bootstrap
        return bootstrap_mcar_test(missing, n_simulations, alpha)
    else:
        # Perform Chi-square test
        chi_sq = np.sum((pattern_counts - expected)**2 / expected)
        df = max(1, len(pattern_counts) - 1 - p)  # Prevent negative degrees of freedom
        p_value = 1 - chi2.cdf(chi_sq, df)
        
        return {
            'method': 'Chi-square',
            'statistic': chi_sq,
            'p_value': p_value,
            'is_mcar': p_value > alpha,
            'expected_freq': expected
        }

def bootstrap_mcar_test(missing, n_simulations=1000, alpha=0.05):
    n, p = missing.shape
    observed_stats = []
    
    # Generate Bootstrap samples
    for _ in range(n_simulations):
        sample = missing.sample(n, replace=True)
        patterns = sample.apply(lambda x: hash(tuple(x)), axis=1)
        cnt = patterns.value_counts().values
        stat = np.sum((cnt - np.mean(cnt))**2) / np.mean(cnt)
        observed_stats.append(stat)
    
    # Calculate actual statistic
    actual_patterns = missing.apply(lambda x: hash(tuple(x)), axis=1)
    actual_counts = actual_patterns.value_counts().values
    actual_stat = np.sum((actual_counts - np.mean(actual_counts))**2) / np.mean(actual_counts)
    
    # Calculate p-value
    p_value = np.mean(np.array(observed_stats) >= actual_stat)
    
    return {
        'method': 'Bootstrap',
        'statistic': actual_stat,
        'p_value': p_value,
        'is_mcar': p_value > alpha,
        'simulations': n_simulations
    }

# Usage example
result = little_mcar_test(X_mimic_2, bootstrap=True)

# Safe formatted output
if result['method'] == 'Chi-square':
    print(f"[Chi-square Test] Statistic: {result['statistic']:.2f}, p-value: {result['p_value']:.4f}, MCAR: {result['is_mcar']}")
else:
    print(f"[Bootstrap] p-value: {result['p_value']:.4f}, MCAR: {result['is_mcar']} (simulations: {result['simulations']})")


import pandas as pd
import numpy as np
import statsmodels.api as sm
def create_pvalue_matrix(data):
    # Identify columns with missing values
    missing_cols = data.columns[data.isnull().any()].tolist()
    # Initialize result matrix
    pvalue_matrix = pd.DataFrame(
        index=missing_cols,
        columns=data.columns,
        dtype=float
    )
    
    # Iterate through each column with missing values as dependent variable
    for target_col in missing_cols:
        # Generate missing indicator variable (0=present, 1=missing)
        y = data[target_col].isnull().astype(int)
        # Iterate through all features as independent variables
        for feature_col in data.columns:
            if feature_col == target_col:
                pvalue_matrix.loc[target_col, feature_col] = np.nan
                continue
            # Copy and process independent variable
            X = data[[feature_col]].copy()
            # Fill missing values in current independent variable
            if X[feature_col].dtype.kind in 'biufc':  # Numeric type
                fill_value = X[feature_col].mean()
            else:  # Categorical type
                fill_value = X[feature_col].mode()[0]
            X_filled = X.fillna(fill_value)
            
            # Add constant term
            X_filled = sm.add_constant(X_filled)
            
            try:
                # Perform logistic regression
                model = sm.Logit(y, X_filled, missing='drop')
                result = model.fit(disp=0)
                
                # Extract p-value for feature column
                pval = result.pvalues[feature_col]
                pvalue_matrix.loc[target_col, feature_col] = pval
            except:
                # Handle cases like singular matrix
                pvalue_matrix.loc[target_col, feature_col] = np.nan
    
    return pvalue_matrix

pvalue_matrix = create_pvalue_matrix(X_mimic_2)


# Visualize missing data patterns
matrix_style = pvalue_matrix.style\
        .background_gradient(cmap='gist_heat_r', axis=None)\
        .format("{:.4f}", na_rep="-")\
        .set_caption("Missing Value Mechanism Analysis (p-values)")
matrix_style



#################### Since samples are MAR (Missing at Random) and MNAR (Missing Not at Random), choose Random Forest Imputation method #######################################


# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_mimic_2, y_mimic,test_size=0.3, random_state=31)
X_val, X_test_mimic, y_val, y_test_mimic = train_test_split(X_temp, y_temp, test_size=0.33, random_state=31)
X_test = pd.concat([X_test_mimic, X_zdyy_2], axis=0)
y_test = pd.concat([y_test_mimic, y_zdyy], axis=0)

'''Replace with actual saving: X_train.to_csv(r"X_train.csv")
y_train.to_csv(r"y_train.csv")
X_val.to_csv(r"X_val.csv")
y_val.to_csv(r"y_val.csv")
X_test.to_csv(r"X_test.csv")
y_test.to_csv(r"y_test.csv")
X_test_mimic.to_csv(r"X_test_mimic.csv") 
y_test_mimic.to_csv(r"y_test_mimic.csv")
X_zdyy_2.to_csv(r"X_test_zdyy.csv") 
y_zdyy.to_csv(r"y_test_zdyy.csv")'''


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

class RandomForestImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.columns_order = []
        self.feature_columns_ = None
        self.imputers = {}
        
    def fit(self, X, y=None):
        X = X.copy()
        self.feature_columns_ = X.columns.tolist()
        # Calculate missing value counts for each column and sort by missing amount
        missing = X.isnull().sum()
        self.columns_order = missing[missing > 0].sort_values().index.tolist()
        # Train models for each column with missing values
        for col in self.columns_order:
            # Get current column labels and feature matrix
            y_col = X[col]
            mask = y_col.notna()  # Non-missing value mask
            
            # Skip completely missing columns
            if mask.sum() == 0:
                continue
                
            # Build feature matrix (need to fill other missing values first)
            features = X.drop(columns=[col])
            
            # Initialize simple imputer
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            X_imputed = imputer.fit_transform(features)
            
            # Train random forest model
            model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                        random_state=self.random_state)
            model.fit(X_imputed[mask], y_col[mask])  # Use only non-missing samples
            
            # Store model and imputer
            self.models[col] = (model, imputer)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.feature_columns_ is None:
            raise NotFittedError("Imputer not fitted yet.")
            
        # Verify column name consistency
        if list(X.columns) != self.feature_columns_:
            raise ValueError("Columns of X do not match those in fit.")
        
        # Impute column by column
        for col in self.columns_order:
            if col not in self.models:
                continue
                
            model, imputer = self.models[col]
            y_col = X[col]
            mask = y_col.isna().to_numpy()  # Convert mask to position-based boolean array instead of index-preserving Series, because indexes are shuffled
            
            # Skip columns with no missing values
            if not mask.any():
                continue
                
            # Prepare feature matrix
            features = X.drop(columns=[col])
            X_imputed = imputer.transform(features)
            
            # Predict missing values
            X.loc[mask, col] = model.predict(X_imputed[mask])
            
        return X

# Usage example
pipeline_imputed = Pipeline([
    ('rf_imputer', RandomForestImputer(n_estimators=100))
])


X_train_imputed = pipeline_imputed.fit_transform(X_train)
X_val_imputed = pipeline_imputed.fit_transform(X_val)
X_test_imputed = pipeline_imputed.fit_transform(X_test)
X_test_mimic_imputed = pipeline_imputed.fit_transform(X_test_mimic)
X_zdyy_imputed = pipeline_imputed.fit_transform(X_zdyy_2)



################################### Convert SOFA score columns to integers after imputation ###########################################
X_train_imputed[['sofa_resp', 'sofa_circ', 'sofa_renal']] = X_train_imputed[['sofa_resp',  'sofa_circ', 'sofa_renal']].round().astype(int)
X_val_imputed[['sofa_resp', 'sofa_circ', 'sofa_renal']] = X_val_imputed[['sofa_resp',  'sofa_circ', 'sofa_renal']].round().astype(int)
X_test_imputed[['sofa_resp', 'sofa_circ', 'sofa_renal']] = X_test_imputed[['sofa_resp',  'sofa_circ', 'sofa_renal']].round().astype(int)
X_test_mimic_imputed[['sofa_resp',  'sofa_circ', 'sofa_renal']] = X_test_mimic_imputed[['sofa_resp', 'sofa_circ', 'sofa_renal']].round().astype(int)
X_zdyy_imputed[['sofa_resp',  'sofa_circ', 'sofa_renal']] = X_zdyy_imputed[['sofa_resp', 'sofa_circ', 'sofa_renal']].round().astype(int)

'''X_train_imputed.to_csv(r"X_train_imputed.csv", index=False)
X_val_imputed.to_csv(r"X_val_imputed.csv", index=False)
X_test_imputed.to_csv(r"X_test_imputed.csv", index=False)
X_test_mimic_imputed.to_csv(r"X_test_mimic_imputed.csv", index=False)
X_zdyy_imputed.to_csv(r"X_zdyy_imputed.csv", index=False)'''




#################################### Standardize all data #########################################


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming X_train_imputed, X_val_imputed, X_test_imputed, X_test_mimic_imputed, X_zdyy_imputed are already defined
pipeline_scaler = Pipeline([
    ('scaler', StandardScaler())
])

# Select columns to standardize (you have already identified these columns)
columns_to_scale = X_train_imputed.loc[:, 'age':'respiratory_rate'].columns

# Create copies, preserving other columns
X_train_non_scaled = X_train_imputed.drop(columns=columns_to_scale)
X_val_non_scaled = X_val_imputed.drop(columns=columns_to_scale)
X_test_non_scaled = X_test_imputed.drop(columns=columns_to_scale)
X_test_mimic_non_scaled = X_test_mimic_imputed.drop(columns=columns_to_scale)
X_zdyy_non_scaled = X_zdyy_imputed.drop(columns=columns_to_scale)



# Standardize only selected columns
X_train_scaled = pd.DataFrame(pipeline_scaler.fit_transform(X_train_imputed[columns_to_scale]), columns=columns_to_scale)
X_val_scaled = pd.DataFrame(pipeline_scaler.fit_transform(X_val_imputed[columns_to_scale]), columns=columns_to_scale)
X_test_scaled = pd.DataFrame(pipeline_scaler.fit_transform(X_test_imputed[columns_to_scale]), columns=columns_to_scale)
X_test_mimic_scaled = pd.DataFrame(pipeline_scaler.fit_transform(X_test_mimic_imputed[columns_to_scale]), columns=columns_to_scale)
X_zdyy_scaled = pd.DataFrame(pipeline_scaler.fit_transform(X_zdyy_imputed[columns_to_scale]), columns=columns_to_scale)

X_train_non_scaled = X_train_non_scaled.reset_index(drop=True)
X_val_non_scaled = X_val_non_scaled.reset_index(drop=True)
X_test_non_scaled = X_test_non_scaled.reset_index(drop=True)
X_test_mimic_non_scaled = X_test_mimic_non_scaled.reset_index(drop=True)
X_zdyy_non_scaled = X_zdyy_non_scaled.reset_index(drop=True)

X_train_scaled = X_train_scaled.reset_index(drop=True)
X_val_scaled = X_val_scaled.reset_index(drop=True)
X_test_scaled = X_test_scaled.reset_index(drop=True)
X_test_mimic_scaled = X_test_mimic_scaled.reset_index(drop=True)
X_zdyy_scaled = X_zdyy_scaled.reset_index(drop=True)

# Merge standardized columns with unchanged columns
X_train_final = pd.concat([X_train_non_scaled, X_train_scaled], axis=1)
X_val_final = pd.concat([X_val_non_scaled, X_val_scaled], axis=1)
X_test_final = pd.concat([X_test_non_scaled, X_test_scaled], axis=1)
X_test_mimic_final = pd.concat([X_test_mimic_non_scaled, X_test_mimic_scaled], axis=1)
X_zdyy_final = pd.concat([X_zdyy_non_scaled, X_zdyy_scaled], axis=1)

# Save final results
'''X_train_final.to_csv(r"X_train_scaler.csv", index=False)
X_val_final.to_csv(r"X_val_scaler.csv", index=False)
X_test_final.to_csv(r"X_test_scaler.csv", index=False)
X_test_mimic_final.to_csv(r"X_test_mimic_scaler.csv", index=False)
X_zdyy_final.to_csv(r"X_zdyy_scaler.csv", index=False)'''


