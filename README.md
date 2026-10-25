# 🩺 Liver Disease Detection & Analytics Dashboard

An end-to-end data analytics and machine learning project that explores patient blood-test data and demonstrates liver disease classification using Python, Jupyter Notebook, Scikit-learn, and Streamlit.

## 📌 Project Overview

This project analyzes the Indian Liver Patient Dataset (ILPD) to explore patterns in patient blood-test measurements and build a machine learning classification prototype.

An interactive Streamlit dashboard presents dataset statistics, exploratory data analysis, model evaluation, and demonstration predictions.

## 🎯 Objectives

- Clean and preprocess patient data.
- Perform exploratory data analysis (EDA).
- Visualize blood-test distributions and feature relationships.
- Train and evaluate a machine learning classifier.
- Build an interactive web dashboard using Streamlit.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data processing
- **Matplotlib & Seaborn** — Data visualization
- **Plotly** — Interactive charts
- **Scikit-learn** — Machine learning
- **Jupyter Notebook** — Data analysis and experimentation
- **Streamlit** — Interactive web dashboard
- **Git & GitHub** — Version control

## 📊 Dashboard Features

### 1. Overview
- Dataset statistics and patient record counts
- Feature summary
- Dataset preview

### 2. Exploratory Data Analysis
- Blood-test feature distributions
- Histograms and box plots
- Correlation heatmap
- Target-class distribution

### 3. Model Performance
- Classification accuracy
- Precision, recall, and F1-score
- Confusion matrix
- Classification report

### 4. Prediction Demonstration
- Interactive input form
- Machine learning model output
- Educational demonstration of classification

## 📂 Project Structure

```text
LiverDiseaseProject/
├── app/
│   └── app.py
├── data/
│   └── cleaned_liver_patient.csv
├── models/
│   └── liver_model.pkl
├── liver_analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

*The dataset and trained model may be excluded from GitHub. Generate them locally by following the setup instructions.*

## 📥 Dataset

**Indian Liver Patient Dataset (ILPD)**

Source: [Kaggle — Indian Liver Patient Records](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)

The dataset contains patient demographic information and blood-test measurements used for educational classification experiments.

Please review the dataset's license and terms before redistributing it.

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Afrith111/liver-disease-analytics.git
cd liver-disease-analytics
```

### 2. Create and activate a Conda environment

```bash
conda create -n dp python=3.11
conda activate dp
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the dataset from Kaggle and place the CSV file in the `data/` directory. Ensure its filename and column names match the paths and preprocessing used in the notebook.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `liver_analysis.ipynb` and run the cells in order to prepare the data and train the model.

### 6. Launch the dashboard

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📈 Model Evaluation

The project uses a Random Forest classifier as an initial model. Performance should be reported using actual evaluation results, including precision, recall, F1-score, and a confusion matrix.

Model results may vary depending on preprocessing, data splitting, and class distribution.

## 🚀 Future Improvements

- Compare multiple machine learning algorithms.
- Add model explainability and feature-importance charts.
- Improve dashboard filters and visual design.
- Add automated data-quality checks.
- Deploy the Streamlit dashboard online.
- Add reproducible experiments and model validation.

## ⚠️ Disclaimer

This project is intended for educational and data analytics purposes only. It is not clinically validated, must not be used to diagnose liver disease, and must not replace professional medical advice.

## 👨‍💻 Author

**Mohamed Afrith B.**

GitHub: [@Afrith111](https://github.com/Afrith111)
