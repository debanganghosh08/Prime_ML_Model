# ML Model Comparator & Dashboard | Streamlit Project

Welcome to an end-to-end machine learning dashboard app built for model evaluation and deployment — built for a hackathon, designed for production!

![image](https://github.com/user-attachments/assets/b51917d0-0ad8-49b2-bf90-8f74c32300b6)


# 💻 Tech Stack:
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Scipy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

![image](https://github.com/user-attachments/assets/55cf7837-0ec1-477f-8f3c-5cc32606e119)


## 🌟 Project Highlights

> 🔹 A powerful and stylish Streamlit dashboard for comparing multiple classification models.<br>
> 🔹 Real-time model testing with uploaded CSVs.<br>
> 🔹 Fully tuned pipelines, metrics analysis, and interactive visualizations.<br>

---

## 🔍 Objective

The purpose of this project is to:

1. Train and evaluate **multiple classification algorithms**
2. Use **cross-validation** and **hyperparameter tuning** for optimization
3. Compare models based on metrics like:
   - Accuracy
   - AUC Score
   - F1-Score
   - Precision, Recall, Specificity
4. Visualize and interpret results through an interactive **Streamlit dashboard**
5. Enable end-users to upload their own CSV and get predictions from tuned models.

---

## 🔧 Features

- ✅ Logistic Regression
- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier
- ✅ Support Vector Machine
- ✅ XGBoost / LightGBM
- ✅ Hyperparameter Tuning (Grid Search)
- ✅ Feature Importance Charts
- ✅ Dynamic Bar Graphs (Plotly)
- ✅ Glassmorphic Streamlit UI
- ✅ Upload CSV to Test Models Live
- ✅ Auto-Pickle & Save All Models
- ✅ Responsive layout with dark mode and Fira Code font

---

## 📆 Dataset

Dataset is sourced from Kaggle. After selection:
- Null values handled
- Categorical features encoded
- Numeric features scaled
- Train/test split applied with stratification

Example input format is available in `example_input.csv`.

---

## 🏃️ Getting Started

### ♻️ Clone and Install

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### 🚀 Run the Dashboard

```bash
streamlit run app.py
```

> Make sure `models/` folder exists with pickled files

---

## 🔬 Model Workflow

### 🤖 Pipelines
- Used `Pipeline()` from `sklearn` for each model
- Feature encoding, scaling, and classification in one step

### 📊 Evaluation Metrics
| Metric      | Description |
|-------------|-------------|
| Accuracy    | % Correct predictions |
| AUC Score   | Area under ROC curve |
| F1-Score    | Harmonic mean of precision & recall |
| Precision   | True Positives / Predicted Positives |
| Recall      | True Positives / Actual Positives |
| Specificity | True Negatives / Actual Negatives |

### ✨ Hyperparameter Tuning
- GridSearchCV for exhaustive tuning
- Best parameters auto-selected for each model

---

## 🌎 Dashboard Highlights

- Theme toggle: Light ✨ / Dark 🌚
- Plotly-based interactive charts
- Hover effects, rounded corners, and modern Fira Code font
- Upload `.csv` file to test any tuned model
- Performance chart comparison between Accuracy and AUC

---

## 🌐 Deployment

Easily deploy on [Streamlit Cloud](https://share.streamlit.io/):
```bash
https://share.streamlit.io/yourusername/your-repo-name/main/app.py
```

You can also deploy via:
- Hugging Face Spaces
- Render.com
- Local containerized environments (Docker)

---

## 🔧 Usage Guide

1. Run `app.py`
2. Upload a CSV following the `example_input.csv` format
3. Select any model from the sidebar dropdown
4. Visualize predictions, performance, and insights

---

## 📷 Screenshots *(Optional)*

| Dashboard View | Feature Importances |
|----------------|---------------------|
| ![](screenshots/dashboard.png) | ![](screenshots/features.png) |

---

## 📊 Sample Metrics Table

| Model            | Accuracy | AUC Score | F1 Score | Recall | Specificity |
|------------------|----------|-----------|----------|--------|-------------|
| Random Forest    | 0.92     | 0.94      | 0.91     | 0.90   | 0.93        |
| XGBoost          | 0.91     | 0.95      | 0.90     | 0.89   | 0.92        |

---

## 💼 License

MIT License © 2025 Debangan Ghosh

---

## 🙋‍♂️ Thank You!

Star ⭐ the repo if you liked the project. Contributions, feedback and forks are always welcome!

Connect with me on [LinkedIn](https://linkedin.com/) or drop an issue if you want to collaborate!

