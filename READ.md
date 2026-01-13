# 🏦 Credit Risk Scoring System

An end-to-end MLOps solution for assessing loan applicant risk. This project utilizes **XGBoost** for classification, **FastAPI** for real-time inference, and **MLflow** for experiment tracking. It includes a user-friendly HTML/JS frontend for interacting with the model.

---

## 🚀 Key Features

* **Machine Learning Pipeline:** Modular steps for Data Cleaning, Preprocessing and Training.
* **MLOps Integration:** * **MLflow:** Tracks experiments, parameters, and metrics.
    * **DVC (Optional):** Ready for data version control integration.
* **FastAPI Backend:** Robust REST API with **Pydantic** data validation.
* **Strategy Pattern:** Object-oriented design for extensible preprocessing logic.
* **Risk Banding:** Classifies applicants into actionable categories (**P1** to **P4**).

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned files for training (X_train, etc.)
│   └── interim/              # Intermediate data
├── models/                   # Saved artifacts (model.joblib, encoder, etc.)
├── src/
│   └── credit_risk_model/
│       ├── api/
│       │   ├── app.py        # FastAPI Application
│       │   ├── index.html    # Frontend Interface
│       │   ├── script.js     # Frontend Logic
│       │   └── style.css     # Frontend Styling
│       ├── clean_data.py     # Data cleaning script
│       ├── preprocess.py     # Feature engineering & splitting
│       ├── train.py          # Model training (XGBoost)
│       ├── evaluate.py       # Performance evaluation
│       └── run_pipeline.py   # Pipeline orchestrator
├── metrics.json              # Final model metrics
├── params.yaml               # Central configuration file
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation