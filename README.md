# 1. 😴 Sleep Quality Analysis Microservice

This project is a Python-based microservice that performs data analysis, statistical modeling, and diagnostics on sleep and lifestyle data. It offers:

- ⚙️ Command-line tools via `click`
- ⚡ REST API using `FastAPI`
- 🎛️ Streamlit-based user interface
- ✅ Data validation using Pydantic
- 📊 Graphs and regression analysis using `matplotlib`, `seaborn`, `statsmodels`

---

## 1.1. 📁 Dataset

The project uses:
``Sleep_health_and_lifestyle_dataset.csv``  
which contains features such as:

- Demographics (Age, Gender, Occupation)
- Lifestyle metrics (Physical Activity, Heart Rate, BMI)
- Sleep metrics (Duration, Quality, Disorder)
- Stress levels and derived categories

---

## 1.2. 🏗️ Project Structure

├── cli.py # Command-line interface (analyze, plot, regression, diagnostics)
├── main.py # FastAPI backend
├── streamlit_app.py # Streamlit frontend UI
├── models.py # Pydantic model for validation
├── data_loader.py # CSV loading, cleaning, and record validation
├── plots/ # Folder where all plots are saved
├── Sleep_health_and_lifestyle_dataset.csv
├── sleep_quality-analysis_original.py # Original unrefactored version
├── requirements.txt
└── README.md

---

## 1.3. 💡 Features

### 1.3.1. 🔍 Analyze
- Preprocesses and filters the dataset
- Validates data using `Pydantic`
- Displays metadata:
  - column names
  - data types
  - inferred roles (dependent/independent)

### 1.3.2. 📈 Regression
- Simple linear: `Stress_Level ~ Quality_of_Sleep`
- Multiple linear: `Stress_Level ~ Quality_of_Sleep + Age`
- Non-linear (quadratic): adds `Quality_of_Sleep^2`
- ANOVA comparison of model fits

### 1.3.3. 🧪 Diagnostics
Runs standard statistical tests on the linear regression model:

- **White test** – heteroskedasticity
- **Breusch-Pagan test** – heteroskedasticity
- **Durbin-Watson test** – autocorrelation
- **Jarque-Bera test** – normality of residuals

### 1.3.4. 📊 Plotting
- Histograms and boxplots for numerical variables
- Countplots for categorical variables
- All charts are saved in the `plots/` directory
- If a plot already exists, it is **not regenerated**
- You can view all saved plots via Streamlit interface

---

## 1.4. ⚙️ How to Run (Requires Python version below 3.12.0)

### 1.4.1. Install & activate virtual environment

```bash
python -m venv .venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1.4.2. ⚙️ Run FastAPI backend
uvicorn main:app --reload

### 1.4.3. ⚙️ Run Streamlit UI
streamlit run streamlit_app.py
