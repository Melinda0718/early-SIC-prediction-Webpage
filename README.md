重症患者早期凝血病预警模型
Early Coagulopathy Warning Model for Critically Ill Patients

📌 项目简介 / Project Overview
本项目旨在开发一个用于重症患者早期凝血病预警的机器学习模型，帮助临床医生提前识别患者的凝血功能异常风险。
This project aims to develop a machine learning model for early coagulopathy warning in critically ill patients, assisting clinicians in identifying the risk of coagulation abnormalities in advance.

项目包含数据清洗、特征工程、模型训练与评估全流程，并提供了基于 Streamlit 的交互式演示界面。
The project includes the entire pipeline of data cleaning, feature engineering, model training and evaluation, and provides an interactive demo interface based on Streamlit.

📁 项目结构 / Project Structure
.
├── .devcontainer/          # 开发容器配置
├── Raw Data/              # 原始数据（含示例数据与清洗流程示例）
│   ├── Example/           # 示例数据
│   └── Cleaned/           # 示例清洗后数据
├── filled/                # 用于 Streamlit 网站的数据/文件
├── Cleaned/               # 已清洗的基线数据
├── clean_split_scaled_onehot_impute_pipeline.py   # 数据预处理管道脚本
├── full_pipeline_sic.pkl  # 训练好的完整管道（包含模型）
├── requirements.txt       # Python 依赖包列表
└── streamlit.py           # Streamlit 应用入口文件


🚀 快速开始 / Quick Start
1. 克隆仓库 / Clone Repository
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

2. 安装依赖 / Install Dependencies
pip install -r requirements.txt

4. 运行 Streamlit 应用 / Run Streamlit App
streamlit run streamlit.py
随后在浏览器中打开提示的本地地址（通常为 http://localhost:8501）。

📊 数据说明 / Data Description
Raw Data/：包含原始示例数据及数据清洗流程示例，可用于理解数据处理过程。
Cleaned/：存放已清洗并准备好用于建模的基线数据。
filled/：存放 Streamlit 应用所需的预处理数据或配置文件。

🧠 模型与管道 / Model & Pipeline
clean_split_scaled_onehot_impute_pipeline.py：实现数据预处理全流程，包括清洗、拆分、缩放、独热编码与缺失值处理。

full_pipeline_sic.pkl：保存了训练好的完整机器学习管道（含预处理与模型），可直接加载用于预测。

🌐 在线演示 / Online Demo
本项目支持通过 Streamlit 快速部署为 Web 应用，用户可通过上传数据或输入特征值获取凝血病风险预警结果。
This project supports quick deployment as a web application via Streamlit, allowing users to upload data or input feature values to obtain early coagulopathy risk warnings.

📌 依赖环境 / Requirements
详见 requirements.txt

📄 许可证 / License
本项目仅供学术研究或临床参考使用，具体使用请遵守相关伦理与法律法规。
This project is for academic research or clinical reference only. Please comply with relevant ethical and legal regulations when using it.

🤝 贡献 / Contribution and Request
欢迎提交 Issue 或 Pull Request 来改进模型或应用。联系方式：gyz2002@126.com.
Contributions are welcome via Issues or Pull Requests to improve the model or application. If Issues, contact gyz2002@126.com.
