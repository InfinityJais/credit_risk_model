from abc import ABC, abstractmethod
import pandas as pd
import joblib
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. Define Input Data Structure ---
class LoanApplication(BaseModel):
    """
    Schema for Loan Application Data.
    Ordered to match the frontend HTML form structure.
    """
    # --- Section 1: Personal & Employment ---
    GENDER: str = Field(..., description="Gender (M, F)")
    MARITALSTATUS: str = Field(..., description="Marital Status (Married, Single)")
    EDUCATION: str = Field(..., description="Education Level (SSC, 12TH, GRADUATE, POST-GRADUATE, OTHERS, PROFESSIONAL)")
    NETMONTHLYINCOME: int = Field(..., description="Net Monthly Income of the applicant")
    Time_With_Curr_Empr: int = Field(..., description="Time spent with current employer (in months)")

    # --- Section 2: Loan Inquiries ---
    last_prod_enq2: str = Field(..., description="Product category of the last inquiry (PL, AL, CC, etc.)")
    first_prod_enq2: str = Field(..., description="Product category of the first inquiry (PL, AL, CC, etc.)")
    enq_L3m: int = Field(..., description="Total inquiries in the last 3 months")
    time_since_recent_enq: int = Field(..., description="Time (days) since the last credit inquiry")
    CC_enq_L12m: int = Field(..., description="Credit Card inquiries in last 12 months")
    PL_enq_L12m: int = Field(..., description="Personal Loan inquiries in last 12 months")
    pct_PL_enq_L6m_of_ever: float = Field(..., description="Percent of Personal Loan inquiries in last 6m vs total ever")
    pct_CC_enq_L6m_of_ever: float = Field(..., description="Percent of Credit Card inquiries in last 6m vs total ever")

    # --- Section 3: Account Portfolio ---
    CC_TL: int = Field(..., description="Count of Credit Card accounts")
    Home_TL: int = Field(..., description="Count of Housing Loan accounts")
    PL_TL: int = Field(..., description="Count of Personal Loan accounts")
    Secured_TL: int = Field(..., description="Count of Secured Loan accounts")
    Unsecured_TL: int = Field(..., description="Count of Unsecured Loan accounts")
    Other_TL: int = Field(..., description="Count of other types of loan accounts")
    Age_Oldest_TL: int = Field(..., description="Age of the oldest opened account")
    Age_Newest_TL: int = Field(..., description="Age of the newest opened account")

    # --- Section 4: Trade Line Performance ---
    pct_tl_open_L6M: float = Field(..., description="Percent of trade lines opened in the last 6 months")
    pct_tl_closed_L6M: float = Field(..., description="Percent of trade lines closed in the last 6 months")
    pct_tl_closed_L12M: float = Field(..., description="Percent of trade lines closed in the last 12 months")
    Tot_TL_closed_L12M: int = Field(..., description="Total trade lines closed in the last 12 months")

    # --- Section 5: Repayment & Delinquency ---
    Tot_Missed_Pmnt: int = Field(..., description="Total count of missed payments across all accounts")
    time_since_recent_payment: int = Field(..., description="Time (days) since the last payment was made")
    max_recent_level_of_deliq: int = Field(..., description="Maximum recent delinquency level")
    recent_level_of_deliq: int = Field(..., description="Most recent level of delinquency")
    num_deliq_6_12mts: int = Field(..., description="Number of delinquencies between 6 and 12 months ago")
    num_times_60p_dpd: int = Field(..., description="Number of times payment was 60+ days past due")
    num_std_12mts: int = Field(..., description="Number of standard payments in last 12 months")

    # --- Section 6: Risk Status Counts ---
    num_sub: int = Field(..., description="Number of sub-standard accounts")
    num_sub_6mts: int = Field(..., description="Number of sub-standard accounts in last 6 months")
    num_sub_12mts: int = Field(..., description="Number of sub-standard accounts in last 12 months")
    num_dbt: int = Field(..., description="Number of doubtful accounts")
    num_dbt_12mts: int = Field(..., description="Number of doubtful accounts in last 12 months")
    num_lss: int = Field(..., description="Number of loss accounts")

    # --- Section 7: Asset Flags ---
    CC_Flag: int = Field(..., description="Flag indicating possession of Credit Card (1=Yes, 0=No)")
    PL_Flag: int = Field(..., description="Flag indicating possession of Personal Loan (1=Yes, 0=No)")
    HL_Flag: int = Field(..., description="Flag indicating possession of Home Loan (1=Yes, 0=No)")
    GL_Flag: int = Field(..., description="Flag indicating possession of Gold Loan (1=Yes, 0=No)")


# --- 2. Scoring Strategy ---
class ScoringStrategy(ABC):
    @abstractmethod
    def load_artifacts(self):
        pass

    @abstractmethod
    def predict(self, data: LoanApplication) -> dict:
        pass

class CreditRiskScorer(ScoringStrategy):
    """
    Handles artifact loading, preprocessing, and prediction logic.
    """
    def __init__(self):
        self.model = None
        self.model_columns = None
        self.label_encoder = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads the trained model, column definitions, and label encoder."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
        
        model_path = os.path.join(project_root, "models/model.joblib")
        columns_path = os.path.join(project_root, "models/model_columns.joblib")
        encoder_path = os.path.join(project_root, "models/label_encoder.joblib")

        try:
            print(f"Loading artifacts from: {project_root}/models/ ...")
            self.model = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
            self.label_encoder = joblib.load(encoder_path)
            print("✅ Artifacts loaded successfully.")
        except FileNotFoundError as e:
            print(f"❌ CRITICAL ERROR: Could not find model files at {project_root}/models/")
            print(f"Specific missing file: {e}")
            raise RuntimeError("Service failed to start: Model artifacts are missing. Run 'dvc repro' first.")

    def _preprocess(self, data: LoanApplication) -> pd.DataFrame:
        """
        Converts API input to a DataFrame and replicates the 
        exact feature engineering steps used in training.
        """
        # 1. Convert Pydantic object to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # 2. Ordinal Encoding (EDUCATION)
        education_map = {
            'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
        }
        df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna(1).astype(int)

        # 3. One-Hot Encoding
        categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        df = pd.get_dummies(df, columns=categorical_cols)

        # 4. Alignment: Ensure API columns match Model columns exactly
        if self.model_columns:
            # Add missing columns with 0
            for col in self.model_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training data
            df = df[self.model_columns]

        # 5. Final Data Type Check
        # Ensure everything is float/int for XGBoost
        df = df.astype(float)

        return df

    def predict(self, data: LoanApplication) -> dict:
        """Runs the full prediction pipeline."""
        if self.model is None or self.label_encoder is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")

        # 1. Preprocess
        processed_df = self._preprocess(data)

        # 2. Predict
        prediction_encoded = self.model.predict(processed_df)[0]
        
        # 3. Decode Label
        if hasattr(self.label_encoder, 'inverse_transform'):
            prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        elif isinstance(self.label_encoder, dict):
            inverted_map = {v: k for k, v in self.label_encoder.items()}
            prediction_label = inverted_map.get(prediction_encoded, "Unknown")
        else:
            prediction_label = str(prediction_encoded)

        # 4. Business Logic Mapping
        status_map = {
            "P1": "Must Give",
            "P2": "Okay to Give",
            "P3": "Risky",
            "P4": "Do Not Give"
        }
        
        decision_status = status_map.get(prediction_label, "Unknown Risk Level")

        return {
            "risk_band": str(prediction_label),
            "status": decision_status
        }


# --- 3. Application Context ---
scorer = CreditRiskScorer()
app = FastAPI(title="Credit Risk Scoring API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def home():
    return {"message": "Credit Risk API is Live."}

@app.post("/predict")
def predict_risk(application: LoanApplication):
    return scorer.predict(application)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)