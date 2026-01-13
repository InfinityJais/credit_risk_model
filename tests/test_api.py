import sys
import os
import pytest
from fastapi.testclient import TestClient

# -------------------------------------------------
# Ensure project root is discoverable
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.credit_risk_model.api.app import app


@pytest.fixture(scope="class")
def client():
    """Shared TestClient for all API tests."""
    return TestClient(app)


class TestCreditRiskAPI:
    """
    API test suite for Credit Risk FastAPI service.
    """

    def test_health_check(self, client):
        """Health check endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_predict_valid_payload(self, client):
        """Prediction endpoint should return valid response."""
        payload = {
            # --- Numeric Features ---
            "pct_tl_open_L6M": 0.5,
            "pct_tl_closed_L6M": 0.0,
            "Tot_TL_closed_L12M": 0,
            "pct_tl_closed_L12M": 0.0,
            "Tot_Missed_Pmnt": 1,
            "CC_TL": 0,
            "Home_TL": 0,
            "PL_TL": 1,
            "Secured_TL": 1,
            "Unsecured_TL": 1,
            "Other_TL": 0,
            "Age_Oldest_TL": 45,
            "Age_Newest_TL": 12,
            "time_since_recent_payment": 50,
            "max_recent_level_of_deliq": 20,
            "num_deliq_6_12mts": 0,
            "num_times_60p_dpd": 0,
            "num_std_12mts": 10,
            "num_sub": 0,
            "num_sub_6mts": 0,
            "num_sub_12mts": 0,
            "num_dbt": 0,
            "num_dbt_12mts": 0,
            "num_lss": 0,
            "recent_level_of_deliq": 5,
            "CC_enq_L12m": 0,
            "PL_enq_L12m": 0,
            "time_since_recent_enq": 20,
            "enq_L3m": 0,
            "NETMONTHLYINCOME": 35000,
            "Time_With_Curr_Empr": 48,
            "CC_Flag": 0,
            "PL_Flag": 1,
            "pct_PL_enq_L6m_of_ever": 0.5,
            "pct_CC_enq_L6m_of_ever": 0.0,
            "HL_Flag": 0,
            "GL_Flag": 0,

            # --- Categorical Features ---
            "MARITALSTATUS": "Married",
            "EDUCATION": "GRADUATE",
            "GENDER": "M",
            "last_prod_enq2": "PL",
            "first_prod_enq2": "others"
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        # ---- Response Contract ----
        assert "risk_band" in data
        assert "status" in data
        assert "confidence" in data

        assert data["risk_band"] in {"P1", "P2", "P3", "P4"}

        expected_status = {
            "P1": "Must Give",
            "P2": "Okay to Give",
            "P3": "Risky",
            "P4": "Do Not Give"
        }

        assert data["status"] == expected_status[data["risk_band"]]

        # ---- Confidence Validation ----
        if data["confidence"] is not None:
            assert 0.0 <= data["confidence"] <= 1.0
