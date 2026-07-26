# 📞 Telco Customer Churn Predictor & Automated MLOps Engine

An end-to-end, production-grade supervised machine learning solution designed to predict telecom subscriber cancellation risks using historical account metadata. 

This repository contains the full production asset lifecycle—from exploratory visual data analysis (EDA) and robust preprocessing pipelines to hyperparameter tuning, testing suites, REST API microservices, an interactive operations monitoring dashboard, and self-healing automated retraining cron workflows.

## 🚀 Key Architectural Features


*   **Robust ML Pipeline**: Safe feature processing using Scikit-Learn `ColumnTransformer` scaling and categorical one-hot encoding setups to prevent data leakage.
*   **Imbalance Resilient**: Addresses highly skewed target labels using stratified splits (`stratify=y`) and class-balanced algorithm training weights.
*   **Hyperparameter Tuning**: Utilizes `GridSearchCV` optimization routines over K-Fold Cross-Validation splits.
*   **Production API Service**: Features a high-performance REST API powered by FastAPI, backed by strict Pydantic payload models and inference audit trails.
*   **Statistical Data Drift Monitoring**: Automatically computes dataset distribution divergence shifts in production traffic logs using a two-sample **Kolmogorov-Smirnov (KS) Test**.
*   **Self-Healing Retraining Routine**: Integrates a background crontab trigger that pulls fresh data and overwrites production binaries automatically if severe drift is detected.
*   **Docker Containerized**: Cloud-ready image configurations exposing both frontend dashboard view tabs and raw backend API endpoints.

## 📂 Project Structure

telco-churn-predictor/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions automated workflow pipeline
├── data/                       # Local directory for reference datasets
│   └── telco_raw.csv           # Ingested subscriber data source
├── models/                     # Storage for binaries, log trails, and diagnostics
│   ├── cron_execution.log      # Log file tracing auto-retrain crontab instances
│   ├── inference_audit.log     # File logging live FastAPI prediction payloads
│   └── optimized_model.pkl     # Production machine learning pipeline binary
├── notebook/                   # Research workspace folder
│   └── exploratory_analysis.ipynb # Jupyter Notebook containing raw EDA
├── src/                        # Main operational backend repository code split
│   ├── __init__.py
│   ├── api.py                  # High-performance FastAPI REST API endpoints
│   ├── data_preprocessing.py   # Pure transformation logic for cleaning raw streams
│   ├── drift_detector.py       # Engine computing statistical distributions (KS Test)
│   └── retrain.py              # Automated retraining sequence workflow
├── tests/                      # Testing architecture suite
│   ├── __init__.py
│   ├── test_api.py             # Live integration testing hooks (FastAPI TestClient)
│   ├── test_api_mocked.py      # Bypassed memory unit mocking routines (unittest.mock)
│   └── test_preprocessing.py   # Code isolation and assertions checking dataframe sanity
├── .dockerignore               # Cache protection filter keeping image profiles slim
├── app.py                      # Unified Streamlit frontend user application interface
├── Dockerfile                  # Linux execution ecosystem build blueprint configuration
└── requirements.txt            # Explicit dependency pinning manifest file

## 💻 Installation & Local Initialization

1. **Clone the Repository Layout**
   ```bash
   git clone https://github.com
   cd telco-churn-predictor
   ```

2. **Set Up a Virtual Environment & Ingest Data**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   
   mkdir -p data models
   curl -o data/telco_raw.csv https://githubusercontent.com
   ```

3. **Install Core Requirements & Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🧪 Quality Assurance & Testing Suite

Before running the applications, execute the complete test matrix locally to confirm structural alignment across dataframes, endpoint responses, and mock components:

```bash
pytest -v
```

## ⚙️ Running the Applications

### 2. The FastAPI REST API Microservice
Launches a lightning-fast backend worker serving predictions over highly scalable network routes:

```bash
uvicorn src.api:app --reload --port 8000
```

*Access interactive Swagger UI API reference testing documentation at:* `http://localhost:8000/docs`

---

## 🐳 Containerization Deployment (Docker)


*   **Build the Unified Image**:
    ```bash
    docker build -t telco-churn-app:latest .
    ```
*   **Run the Streamlit Frontend Instance**:
    ```bash
    docker run -d -p 8501:8501 --name churn_frontend telco-churn-app:latest
    ```
*   **Run the FastAPI Backend Instance instead (Override)**:
    ```bash
    docker run -d -p 8000:8000 --name churn_backend telco-churn-app:latest uvicorn src.api:app --host 0.0.0.0 --port 8000
    ```

---

## 🔄 Production MLOps Maintenance (Cron Scheduler)

To run the automated, self-healing data drift check and model retraining script hands-free every night at midnight, add the following cron configuration line to your host server's scheduler via `crontab -e`:

```text
0 0 * * * /usr/bin/python3 /absolute/path/to/project/src/retrain.py >> /absolute/path/to/project/models/cron_execution.log 2>&1
```
