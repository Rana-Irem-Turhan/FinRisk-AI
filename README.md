# 🚀 FinRisk-AI: Kredi Skoru Sınıflandırma Sistemi

## 1. 🎯 Problem Tanımı (Problem Definition)
The objective of this project is to build a machine learning model to classify customers' credit scores into three categories: **Good, Standard, and Poor**. This automated system aims to reduce manual underwriting time and improve risk assessment accuracy.

## 2. 💼 İş Perspektifinden Amaç (Business-Focused Objective)
> **Problem Statement:** As a Data Scientist at a global finance company, your core task is to develop an intelligent system that automatically segregates potential customers into distinct credit score brackets. The goal is to **significantly reduce the time and human effort** currently spent on manual credit underwriting while **enhancing the precision** of the bank's risk assessment strategy.

## 3. 🗺️ Proje Kapsamı (Project Scope)
### 3.1. 📖 Dokümantasyon
* [Setup & Installation](docs/00_setup.md)
* [Data Overview](docs/01_data_overview.md)
* [Baseline Models](docs/02_baseline.md)
* [Feature Engineering](docs/03_feature_engineering.md)
* [Model Optimization](docs/04_model_optimization.md)
* [API Deployment](docs/05_model_evaluation.md)

### 3.2. 📓 Notebook'lar
* [First Look 😺](notebooks/Analysis/00_Data_Preparation_Training.ipynb)
* [First Bite 😋](notebooks/Modeling/02_baseline_model.ipynb)
* [Check out the Feature](notebooks/Modeling/03_feature_engineering.ipynb)
* [Baseline Models](notebooks/Modeling/03_feature_engineering.ipynb)
* [Feature Engineering](docs/03_feature_engineering.md)
* [Model Optimization](docs/04_model_optimization.md)
* [Evaluation](docs/05_model_evaluation.md)
* 
## 4. Deployment
**🌐 [Try the Model Instantly](https://huggingface.co/spaces/iremrit/FinRisk-AI)**

To run locally:
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `python src/app.py`
3.  Open browser at `http://localhost:7860`

## 5. Key Findings & Results
*   **Baseline Score**: 60% Accuracy (Logistic Regression).
*   **Final Score**: **75% Accuracy** (Stacking Classifier with Random Forest and XGBoost base estimators).
*   **Top Predictors**: Outstanding Debt, Credit Mix, and Interest Rate.
*   **Business Impact**: Potential to reduce default rates by 15% and cut processing time by 90%.

## 6. Repository Structure


```
FinRisk-AI/
│
├── README.md                          # Project Overview
├── requirements.txt                   # Dependencies
├── .gitignore                         
│
├── data/                              # Raw and Processed Data
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv              
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── docs/                             # Detailed Documentation
│   ├── 00_setup.md
│   ├── 01_data_overview.md
│   ├── 02_baseline.md
│   ├── 03_feature_engineering.md
│   ├── 04_model_optimization.md
│   └── 05_evaluation_report.md
│
├── notebooks/                         # Jupyter Notebooks (EDA -> Pipeline)
│   ├── Analysis/
│   │   └── 00_Data_Preparation_Training.ipynb
│   └── Modeling/
│       ├── 01_EDA.ipynb
│       ├── 02_baseline_model.ipynb
│       ├── 03_feature_engineering.ipynb
│       ├── 04_model_optimization.ipynb
│       └── 05_model_evaluation.ipynb
│       
│
├── src/                               # Source Code
│   ├── templates/                     #UI
│   │   └── index.html
│   ├── models/                        # Saved Artifacts
│   │   ├── final_model.pkl
│   │   └── features.json
│   └── tests/
│       ├── app.py                   # App
│       ├── config.py                # Configuration
│       ├── inference.py             # Prediction Logic
│       └── pipeline.py              # Training Pipeline
│
└── OIG2.png
```
<img width="1862" height="853" alt="image" src="https://github.com/user-attachments/assets/0e259956-69d9-4c82-99d3-0ad0fbb619a3" />


## 📌 To-Do: Business & Model Improvements

- [ ] Validate the final model on a separate holdout test set (Nihai modeli ayrı bir test seti üzerinde doğrulama)
- [ ] Set up model monitoring (monthly accuracy, drift in key features) (Model izleme sistemini kurma)
- [ ] Define decision thresholds for each credit score class (Her sınıf için karar eşiklerini belirleme)
- [ ] Add fallback rules for uncertain predictions (e.g., probability < 55%) (Belirsiz tahminler için yedek kurallar ekleme)
- [ ] Build a feedback loop to compare predicted vs actual scores (Tahmin edilen ve gerçekleşen skorları karşılaştırmak için geri bildirim döngüsü oluşturma)
- [ ] Document model limitations and train credit team on edge cases (Model sınırlamalarını belgeleyip kredi ekibini eğitme)
- [ ] **Refresh UI/UX**: Enhance the look and feel of the Gradio/Streamlit app for a professional presentation (Uygulamanın görsel arayüzünü profesyonel bir sunum için iyileştirme)
- [ ] **Write a Medium Article**: Document the project, methodology, and results for broader technical audience (Projenin metodolojisini ve sonuçlarını açıklayan bir Medium yazısı hazırlama)
- [ ] **Cleanup and Refactor Repo**: Ensure consistent naming conventions, detailed docstrings, and a clean code base (Tutarlı isimlendirme ve detaylı dokümantasyon ile repo düzenini sağlama)
- [ ] Add automated unit tests for core inference logic (Çıkarım mantığı için otomatik birim testleri ekleme)

## Contact
*   **Author**: Rana Irem Turhan
*   **GitHub**: github.com/Rana-Irem-Turhan
*   **LinkedIn**: https://www.linkedin.com/in/irem-turhan/
