#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import matplotlib as mpl
mpl.use('agg')  # 必须作为第一个 matplotlib 相关导入
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediction for early-happened SIC",
    layout="centered",
)


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X_train = pd.read_csv("filled/X_train_imputed.csv")
X_valid = pd.read_csv("filled/X_val_imputed.csv")
X_test = pd.read_csv("filled/X_test_imputed.csv")
X_test_mimic = pd.read_csv("filled/X_test_mimic_imputed.csv")
X_zdyy = pd.read_csv("filled/X_zdyy_imputed.csv")

X_train_scaled = pd.read_csv("filled/X_train_scaler.csv")
X_valid_scaled = pd.read_csv("filled/X_val_scaler.csv")
X_test_scaled = pd.read_csv("filled/X_test_scaler.csv")
X_test_mimic_scaled = pd.read_csv("filled/X_test_mimic_scaler.csv")
X_zdyy_scaled = pd.read_csv("filled/X_zdyy_scaler.csv")

y_train = pd.read_csv("filled/y_train.csv")
y_valid = pd.read_csv("filled/y_val.csv")
y_test = pd.read_csv("filled/y_test.csv")
y_test_mimic = pd.read_csv("filled/y_test_mimic.csv")
y_zdyy = pd.read_csv("filled/y_test_zdyy.csv")


# In[3]:


y_train = y_train['SIC_early_happen']
y_valid = y_valid['SIC_early_happen']
y_test = y_test['0']
y_test_mimic = y_test_mimic['SIC_early_happen']
y_zdyy = y_zdyy['SIC_D3']


# In[4]:


import pandas as pd
import re

def clean_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df
X_train = clean_columns(X_train)
X_valid = clean_columns(X_valid)
X_test = clean_columns(X_test)
X_test_mimic = clean_columns(X_test_mimic)
X_zdyy = clean_columns(X_zdyy)

X_train_scaled = clean_columns(X_train_scaled)
X_valid_scaled = clean_columns(X_valid_scaled)
X_test_scaled = clean_columns(X_test_scaled)
X_test_mimic_scaled = clean_columns(X_test_mimic_scaled)
X_zdyy_scaled = clean_columns(X_zdyy_scaled)


# In[5]:


#将计算出的特征值导入
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "filled", "Features_Final.csv")

try:
    importance_df = pd.read_csv(csv_path, encoding='utf-8')
    top_features = importance_df.head(10)['Feature_Names'].values
except FileNotFoundError:
    st.error(f"文件未找到：{csv_path}")
except Exception as e:
    st.error(f"读取文件失败：{str(e)}")

import re
top_features = [re.sub(r'[^\x00-\x7F]+', '', f) for f in top_features]
X_train = X_train[top_features]
X_valid = X_valid[top_features]
X_test_mimic = X_test_mimic[top_features]
X_zdyy = X_zdyy[top_features]

X_train_scaled = X_train_scaled[top_features]
X_valid_scaled = X_valid_scaled[top_features]
X_test_mimic_scaled = X_test_mimic_scaled[top_features]
X_zdyy_scaled = X_zdyy_scaled[top_features]

y_train = y_train.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)
y_test_mimic = y_test_mimic.reset_index(drop=True)
y_zdyy = y_zdyy.reset_index(drop=True)

train_labels = np.unique(y_train)
valid_labels = np.unique(y_valid)


# In[6]:


import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

neg_count = len(y_train[y_train == 0])
pos_count = len(y_train[y_train == 1])
scale_pos_weight = neg_count / pos_count

xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=50,
    learning_rate=0.0585,
    max_depth=3,
    gamma=1.0,
    subsample=0.6,
    scale_pos_weight=scale_pos_weight,
    colsample_bytree=0.6,
    random_state=42
)

# 使用管道集成过采样与校准
from imblearn.pipeline import make_pipeline
pipeline_xgb = make_pipeline(
    SMOTE(random_state=42),
    CalibratedClassifierCV(
        xgb_model, 
        method='sigmoid', 
        cv=StratifiedKFold(n_splits=5)
    )
)
pipeline_xgb.fit(X_train, y_train)
joblib.dump(pipeline_xgb, "full_pipeline_sic.pkl")


# In[7]:


import pandas as pd
import joblib
import shap
from sklearn.base import is_classifier

# 加载预训练管道（包含SMOTE和校准）
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("full_pipeline_sic.pkl")
    return pipeline

pipeline_xgb = load_pipeline()
X_train_resampled, _ = pipeline_xgb.named_steps['smote'].fit_resample(X_train, y_train)


# In[8]:


def extract_model(pipeline):
    calibrated = pipeline.named_steps['calibratedclassifiercv']
    return calibrated.calibrated_classifiers_[0].estimator
    
model = extract_model(pipeline_xgb)


# In[9]:


@st.cache_resource
def init_shap_explainer(_model, X_background=None):
    return shap.TreeExplainer(
        model=_model,
        data=X_background,
        model_output='probability'
    )


# In[10]:


print(top_features)


# In[11]:


with st.form("prediction_form"):
    col_sofa, col_vent, col_lab1, col_lab2 = st.columns([2, 1.5, 3, 3])
    
    with col_sofa:
        st.markdown("SOFA Scores")
        sofa_circ = st.slider(
            "Circulation", 0, 4, 0, 1,
            help="Scoring based on MAP and vasoactive drug dose."
        )
        sofa_renal = st.slider(
            "Renal", 0, 4, 0, 1,
            help="Scoring based on creatinine and urine output."
        )
        sofa_resp = st.slider(
            "Respiratory", 0, 4, 0, 1,
            help="Scoring based on PaO2/FiO2 and ventilation support."
        )

    with col_vent:
        st.markdown("Respiratory Support")
        mech_vent_1 = st.radio(
            "Mechine Ventilation Status on Day 1",
            options=("yes", "no"),
            index=1,
            help="Whether invasive mechanical ventilation was used on the first day of ICU admission.",
            # 网页6单选组件样式参考
            horizontal=True
        )
    mech_vent_encoded = 1 if mech_vent_1 == "yes" else 0
    
    # 中间列 - 实验室指标1
    with col_lab1:
        st.markdown("Coagulation Laboratory Tests")
        platelet_count = st.number_input(
            "Platelet count (×10⁹/L)", 
            min_value=20, max_value=600, 
            value=150, step=10
        )
        inr = st.number_input(
            "INR",
            min_value=0.5, max_value=5.0,
            value=1.2, step=0.1, format="%.1f"
        )
    
    # 右侧列 - 实验室指标2
    with col_lab2:
        st.markdown("blood Gas Analysis")
        lactate = st.number_input(
            "lactate (mmol/L)",
            min_value=0.5, max_value=15.0,
            value=2.0, step=0.5, format="%.1f"
        )
    
    submitted = st.form_submit_button("Analyze")


# In[16]:


explainer = init_shap_explainer(_model=model, X_background=X_train_resampled)

with st.expander("Features Contribution Visualization", expanded=True):
    try:
        input_df = pd.DataFrame([[sofa_circ, mech_vent_encoded, sofa_renal, platelet_count, inr, sofa_resp, lactate]],
                      columns=["sofa_circ", "mech_vent_1", "sofa_renal", "platelet_count", "inr", "sofa_resp", "lactate"])
        
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"The probability of early-SIC happening: {proba:.1%}")

        # 计算 SHAP 值
        shap_explanation = explainer(input_df)
        shap_values = shap_explanation.values[0]

        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]  # 二分类取正类
        else:
            base_value = explainer.expected_value

        # 格式化特征值
        formatted_features = input_df.copy().astype(object)
        for col in formatted_features.columns:
            if formatted_features[col].dtype in [np.float64, np.int64]:
                formatted_features[col] = formatted_features[col].apply(lambda x: f"{x:.1f}")

        plt.figure(figsize=(18, 5), dpi=300, facecolor='white')
        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values,
            feature_names=list(input_df.columns), 
            features=formatted_features.iloc[0].values,
            matplotlib=True,
            show=False,
            contribution_threshold=0.03,
            figsize=(18,5)
        )

        # 增强样式
        plt.title("Clinical Feature Impacts", fontsize=14, pad=20, fontweight='bold')
        plt.xlabel("SHAP Value Contribution", fontsize=12, labelpad=15)
        plt.xticks(fontsize=10, rotation=0)
        plt.gca().xaxis.set_ticks_position('bottom')
        
        # 设置坐标轴边距
        plt.margins(x=0.15)
        
        # 保存高清图像
        plt.savefig("shap_force.png", 
                   dpi=900, 
                   bbox_inches='tight', 
                   pad_inches=0.2,
                   facecolor='white')
        
        # 在 Streamlit 中显示
        st.image("shap_force.png", use_container_width=True)

    except Exception as e:
        st.error(f"可视化错误: {str(e)}")

