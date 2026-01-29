#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####################数据清洗########################################
import pandas as pd
import numpy as np
import os

#导入MIMIC和ZDYY数据
mimic_d1 = pd.read_csv('MIMIC_D1')
zdyy_d1 = pd.read_csv('ZDYY_D1.csv')

##D1数据增加与删除：sofa=0，性别，年龄，在院状态，在ICU状态重要数据缺失直接删除；
dfmimic = mimic_d1[(mimic_d1['sofa'].notna()) & (mimic_d1['sofa'] != 0) & (mimic_d1['status_hosp'].notna()) & (mimic_d1['status_icu'].notna()) & (mimic_d1['age'].notna()) & (mimic_d1['gender'].notna())]
dfzdyy = zdyy_d1[(zdyy_d1['sofa'].notna()) & (zdyy_d1['sofa'] != 0) & (zdyy_d1['status_hosp'].notna()) & (zdyy_d1['status_icu'].notna()) & (zdyy_d1['age'].notna()) & (zdyy_d1['gender'].notna())]
print(dfmimic[['status_icu', 'age']].describe())
print(dfzdyy[['status_icu', 'age']].describe())

#数据转换 mimic
print(list(dfmimic.dtypes))
object_columns = dfmimic.select_dtypes(include='object')
print(object_columns)
dfmimic[object_columns.columns] = object_columns.apply(pd.to_numeric, errors='coerce')
print(dfmimic)

##mimic
#重复值识别
duplicates = dfmimic['subject_id'].duplicated()
print("subject_id列中的重复值标记:")
print(duplicates)

#重复值去除，该数据表中没有重复值
dfmimic1 = dfmimic.drop_duplicates(subset='subject_id', keep='first')
print("\n去除重复值后的数据:")
print(dfmimic1)

#zdyy
#重复值识别
duplicates = dfzdyy['subject_id'].duplicated()
print("subject_id列中的重复值标记:")
print(duplicates)

#重复值去除，该数据表中没有重复值
dfzdyy1 = dfzdyy.drop_duplicates(subset='subject_id', keep='first')
print("\n去除重复值后的数据:")
print(dfzdyy1)


# In[ ]:


###########################训练集，验证集和测试集规整化##################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#内部数据集
X_mimic = pd.read_csv('MIMIC.csv')
y_mimic = X_mimic['SIC_early_happen']
del X_mimic['SIC_early_happen']
del X_mimic['sofa']
del X_mimic['sofa_coag']
del X_mimic['sofa_cns']
del X_mimic['SIC_score']
#外部数据集
X_zdyy = pd.read_csv('ZDYY.csv')
y_zdyy = X_zdyy['SIC_D3']
del X_zdyy['SIC_D3']

#删除y缺失值，同时删除对应索引的X中的行
missing_indices_mimic = y_mimic[y_mimic.isnull()].index
X_mimic = X_mimic.drop(index=missing_indices_mimic)
y_mimic = y_mimic.drop(index=missing_indices_mimic)

missing_indices_zdyy = y_zdyy[y_zdyy.isnull()].index
X_zdyy = X_zdyy.drop(index=missing_indices_zdyy)
y_zdyy = y_zdyy.drop(index=missing_indices_zdyy)


# In[2]:


##数据预处理
#对二分类变量进行独热编码
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
categorical_features = [
    'status_hosp', 'status_icu', 'gender', 'Vasoactive_drug_use', 
    'mech_vent', 'lung_infection', 'abdominal_infection', 'cns_infection', 
    'urinary_infection', 'soft_tissue_infection', 'hypertension', 'diabetes', 
    'chronic_kidney_disease', 'chronic_pulmonary_disease', 'chronic_liver_disease', 
    'cerebrovascular_disease', 'crrt_status'
] #所有需要编码的二分类变量
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
        
        # 将独热编码列转换为 uint8 类型
        for col in new_columns:
            X_transformed_df[col] = X_transformed_df[col].astype('uint8')
        
        return X_transformed_df
    else:
        return X
        
X_mimic_oh = onehot(categorical_features, X_mimic)
X_zdyy_oh = onehot(categorical_features, X_zdyy)


# In[3]:


print(X_mimic_oh['d2d'])
# 自定义占位符列表，可以根据数据实际情况进行调整
special_values = ["", "NULL", "N/A", "unknown", "not available"]  # 示例占位符

# 将特殊值替换为 NaN
X_mimic_oh_1 = X_mimic_oh.replace(special_values, np.nan)
X_zdyy_oh_1= X_zdyy_oh.replace(special_values, np.nan)
print(X_mimic_oh_1.columns)


# In[4]:


##规整化所有表格

#去除掉id列，但先前数据处理中需要
X_mimic_oh_1.columns = X_mimic_oh_1.columns.str.strip()  # 清除列名中的前后空格
print(X_mimic_oh_1.columns)
X_zdyy_oh_1.columns = X_zdyy_oh_1.columns.str.strip()  # 清除列名中的前后空格
print(X_zdyy_oh_1.columns)


# In[ ]:


id = X_mimic_oh_1.loc[:, 'Unnamed: 0':'hadm_id'].columns
X_mimic_oh_2 = X_mimic_oh_1.drop(columns=id, inplace=False)

#将内部数据集中缺失值超过35%的列去除
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


# In[7]:


####################################特征值缺失性质####################################
#是否为MCAR


# In[8]:


import numpy as np
import pandas as pd
from scipy.stats import chi2
import warnings

def little_mcar_test(data, alpha=0.05, bootstrap=False, n_simulations=1000):
    # 检查输入有效性
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    if len(data) < 10:
        warnings.warn("样本量过小可能导致检验不可靠", UserWarning)
    
    # 列名去重处理
    data = data.loc[:, ~data.columns.duplicated()]
    
    # 生成缺失模式哈希值
    missing = data.isnull()
    patterns = missing.apply(lambda x: hash(tuple(x)), axis=1)
    pattern_counts = patterns.value_counts().values
    
    # 计算期望频数
    n, p = data.shape
    col_missing_rates = missing.mean()
    expected = n * np.prod(col_missing_rates)
    
    # 自动选择检验方法
    if bootstrap or np.any(expected < 5):
        # 执行Bootstrap
        return bootstrap_mcar_test(missing, n_simulations, alpha)
    else:
        # 执行卡方检验
        chi_sq = np.sum((pattern_counts - expected)**2 / expected)
        df = max(1, len(pattern_counts) - 1 - p)  # 防止自由度为负
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
    
    # 生成Bootstrap样本
    for _ in range(n_simulations):
        sample = missing.sample(n, replace=True)
        patterns = sample.apply(lambda x: hash(tuple(x)), axis=1)
        cnt = patterns.value_counts().values
        stat = np.sum((cnt - np.mean(cnt))**2) / np.mean(cnt)
        observed_stats.append(stat)
    
    # 计算实际统计量
    actual_patterns = missing.apply(lambda x: hash(tuple(x)), axis=1)
    actual_counts = actual_patterns.value_counts().values
    actual_stat = np.sum((actual_counts - np.mean(actual_counts))**2) / np.mean(actual_counts)
    
    # 计算p值
    p_value = np.mean(np.array(observed_stats) >= actual_stat)
    
    return {
        'method': 'Bootstrap',
        'statistic': actual_stat,
        'p_value': p_value,
        'is_mcar': p_value > alpha,
        'simulations': n_simulations
    }

# 使用示例
result = little_mcar_test(X_mimic_2, bootstrap=True)

# 安全格式化输出
if result['method'] == 'Chi-square':
    print(f"[卡方检验] 统计量: {result['statistic']:.2f}, p值: {result['p_value']:.4f}, MCAR: {result['is_mcar']}")
else:
    print(f"[Bootstrap] p值: {result['p_value']:.4f}, MCAR: {result['is_mcar']} (模拟次数: {result['simulations']})")


# In[9]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
def create_pvalue_matrix(data):
    # 识别存在缺失值的列
    missing_cols = data.columns[data.isnull().any()].tolist()
    # 初始化结果矩阵
    pvalue_matrix = pd.DataFrame(
        index=missing_cols,
        columns=data.columns,
        dtype=float
    )
    
    # 遍历每个存在缺失的列作为因变量
    for target_col in missing_cols:
        # 生成缺失指示变量 (0=存在, 1=缺失)
        y = data[target_col].isnull().astype(int)
        # 遍历所有特征作为自变量
        for feature_col in data.columns:
            if feature_col == target_col:
                pvalue_matrix.loc[target_col, feature_col] = np.nan
                continue
            # 复制并处理自变量
            X = data[[feature_col]].copy()
            # 填充当前自变量的缺失值
            if X[feature_col].dtype.kind in 'biufc':  # 数值型
                fill_value = X[feature_col].mean()
            else:  # 分类型
                fill_value = X[feature_col].mode()[0]
            X_filled = X.fillna(fill_value)
            
            # 添加常数项
            X_filled = sm.add_constant(X_filled)
            
            try:
                # 执行逻辑回归
                model = sm.Logit(y, X_filled, missing='drop')
                result = model.fit(disp=0)
                
                # 提取特征列的p值
                pval = result.pvalues[feature_col]
                pvalue_matrix.loc[target_col, feature_col] = pval
            except:
                # 处理奇异矩阵等情况
                pvalue_matrix.loc[target_col, feature_col] = np.nan
    
    return pvalue_matrix

pvalue_matrix = create_pvalue_matrix(X_mimic_2)


# In[10]:


# 将数据缺失形式可视化
matrix_style = pvalue_matrix.style\
        .background_gradient(cmap='gist_heat_r', axis=None)\
        .format("{:.4f}", na_rep="-")\
        .set_caption("Missing Value Mechanism Analysis (p-values)")
matrix_style


# In[11]:


################################样本为MAR（缺失值与其他样本相关）和MNAR，所以选择随机森林插补法#######################################


# In[ ]:


#进行训练集，验证集和测试集的拆分
X_train, X_temp, y_train, y_temp = train_test_split(X_mimic_2, y_mimic,test_size=0.3, random_state=31)
X_val, X_test_mimic, y_val, y_test_mimic = train_test_split(X_temp, y_temp, test_size=0.33, random_state=31)
X_test = pd.concat([X_test_mimic, X_zdyy_2], axis=0)
y_test = pd.concat([y_test_mimic, y_zdyy], axis=0)

'''替换为实际：X_train.to_csv(r"X_train.csv")
y_train.to_csv(r"y_train.csv")
X_val.to_csv(r"X_val.csv")
y_val.to_csv(r"y_val.csv")
X_test.to_csv(r"X_test.csv")
y_test.to_csv(r"y_test.csv")
X_test_mimic.to_csv(r"X_test_mimic.csv") 
y_test_mimic.to_csv(r"y_test_mimic.csv")
X_zdyy_2.to_csv(r"X_test_zdyy.csv") 
y_zdyy.to_csv(r"y_test_zdyy.csv")'''


# In[13]:


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
        # 计算各列缺失值数量并按缺失量排序
        missing = X.isnull().sum()
        self.columns_order = missing[missing > 0].sort_values().index.tolist()
        # 为每个含缺失值的列训练模型
        for col in self.columns_order:
            # 获取当前列的标签和特征矩阵
            y_col = X[col]
            mask = y_col.notna()  # 非缺失值掩码
            
            # 跳过全缺失列
            if mask.sum() == 0:
                continue
                
            # 构建特征矩阵（需先填充其他缺失值）
            features = X.drop(columns=[col])
            
            # 初始化简单填充器
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            X_imputed = imputer.fit_transform(features)
            
            # 训练随机森林模型
            model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                        random_state=self.random_state)
            model.fit(X_imputed[mask], y_col[mask])  # 仅使用非缺失样本
            
            # 存储模型和填充器
            self.models[col] = (model, imputer)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.feature_columns_ is None:
            raise NotFittedError("Imputer not fitted yet.")
            
        # 验证列名一致性
        if list(X.columns) != self.feature_columns_:
            raise ValueError("Columns of X do not match those in fit.")
        
        # 逐列插补
        for col in self.columns_order:
            if col not in self.models:
                continue
                
            model, imputer = self.models[col]
            y_col = X[col]
            mask = y_col.isna().to_numpy()  #将mask转换为基于位置的布尔数组，而不是保留索引的Series，因为索引是被打乱后的
            
            # 跳过无缺失列
            if not mask.any():
                continue
                
            # 准备特征矩阵
            features = X.drop(columns=[col])
            X_imputed = imputer.transform(features)
            
            # 预测缺失值
            X.loc[mask, col] = model.predict(X_imputed[mask])
            
        return X

# 使用示例
pipeline_imputed = Pipeline([
    ('rf_imputer', RandomForestImputer(n_estimators=100))
])


# In[14]:


X_train_imputed = pipeline_imputed.fit_transform(X_train)
X_val_imputed = pipeline_imputed.fit_transform(X_val)
X_test_imputed = pipeline_imputed.fit_transform(X_test)
X_test_mimic_imputed = pipeline_imputed.fit_transform(X_test_mimic)
X_zdyy_imputed = pipeline_imputed.fit_transform(X_zdyy_2)


# In[ ]:


###################################将填补后的数据中SOFA评分一栏作为整数###########################################
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


# In[17]:


####################################将所有的数据标准化#########################################


# In[24]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# 假设 X_train_imputed, X_val_imputed, X_test_imputed, X_test_mimic_imputed, X_zdyy_imputed 已经定义
pipeline_scaler = Pipeline([
    ('scaler', StandardScaler())
])

# 选择要标准化的列（你已经确定了这些列）
columns_to_scale = X_train_imputed.loc[:, 'age':'respiratory_rate'].columns

# 创建一个副本，保留其他列
X_train_non_scaled = X_train_imputed.drop(columns=columns_to_scale)
X_val_non_scaled = X_val_imputed.drop(columns=columns_to_scale)
X_test_non_scaled = X_test_imputed.drop(columns=columns_to_scale)
X_test_mimic_non_scaled = X_test_mimic_imputed.drop(columns=columns_to_scale)
X_zdyy_non_scaled = X_zdyy_imputed.drop(columns=columns_to_scale)


# In[ ]:


# 仅对选择的列进行标准化
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

# 合并标准化后的列和未变动的列
X_train_final = pd.concat([X_train_non_scaled, X_train_scaled], axis=1)
X_val_final = pd.concat([X_val_non_scaled, X_val_scaled], axis=1)
X_test_final = pd.concat([X_test_non_scaled, X_test_scaled], axis=1)
X_test_mimic_final = pd.concat([X_test_mimic_non_scaled, X_test_mimic_scaled], axis=1)
X_zdyy_final = pd.concat([X_zdyy_non_scaled, X_zdyy_scaled], axis=1)

# 保存最终结果
'''X_train_final.to_csv(r"X_train_scaler.csv", index=False)
X_val_final.to_csv(r"X_val_scaler.csv", index=False)
X_test_final.to_csv(r"X_test_scaler.csv", index=False)
X_test_mimic_final.to_csv(r"X_test_mimic_scaler.csv", index=False)
X_zdyy_final.to_csv(r"X_zdyy_scaler.csv", index=False)'''

