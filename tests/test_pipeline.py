import os
import joblib
import pandas as pd
import pytest
import sys
import numpy as np

# Change Class Name: Must start with "Test" for pytest to find it!
class TestPipeline:
    """
    Integration Test Suite to verify the integrity of the Data Pipeline.
    Checks existence of Raw Data -> Cleaning -> Preprocessing -> Model Artifacts.
    """

    # Define paths to critical artifacts as a class attribute
    PATHS = {
        "raw_data1": "data/raw/case_study1.xlsx",
        "raw_data2": "data/raw/case_study2.xlsx",
        "cleaned_data": "data/interim/merged_data.csv",
        "train_data": "data/processed/X_train.csv",
        "test_data": "data/processed/X_test.csv",
        "model": "models/model.joblib",
        "encoder": "models/label_encoder.joblib"
    }

    def run_all(self):
        """
        Executes all tests manually (Script Mode).
        """
        print("🚀 Starting Pipeline Integration Tests...")
        try:
            self.test_raw_data_exists()
            print(" -> Raw data verified.")
            
            self.test_cleaning_pipeline()
            print(" -> Cleaning pipeline output verified.")
            
            self.test_preprocessing_features()
            print(" -> Preprocessing features verified.")
            
            self.test_model_loading_and_prediction()
            print(" -> Model prediction logic verified.")
            
            print("\n✅ All Pipeline tests passed!")
            
        except AssertionError as e:
            print(f"\n❌ Test Failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            sys.exit(1)

    # --- Test Methods (Pytest Compatible) ---

    def test_raw_data_exists(self):
        """Check if raw input files are present."""
        assert os.path.exists(self.PATHS["raw_data1"]), "Case Study 1 (raw_data1) is missing!"
        assert os.path.exists(self.PATHS["raw_data2"]), "Case Study 2 (raw_data2) is missing!"

    def test_cleaning_pipeline(self):
        """
        Check if clean_data.py produced the output file correctly.
        """
        path = self.PATHS["cleaned_data"]
        if not os.path.exists(path):
            pytest.skip("Cleaned data not found. Run 'dvc repro' or 'src/clean_data.py' first.")
        
        df = pd.read_csv(path)
        assert not df.empty, "Cleaned data is empty."
        
        # PROSPECTID was the merge key, so it must exist
        assert "PROSPECTID" in df.columns, "PROSPECTID column missing in cleaned data."

    def test_preprocessing_features(self):
        """
        Check if X_train has exactly the features the model expects.
        Also ensures no raw strings (Objects) are left.
        """
        path = self.PATHS["train_data"]
        if not os.path.exists(path):
            pytest.skip("Processed data not found. Run 'src/preprocess.py' first.")

        df = pd.read_csv(path)
        
        # 1. Check for a few critical columns to ensure schema is correct
        required_cols = ["pct_tl_open_L6M", "NETMONTHLYINCOME", "EDUCATION"]
        for col in required_cols:
            assert col in df.columns, f"Critical feature {col} missing in X_train."
        
        # 2. Check that NO object (string) columns remain (everything must be numeric for ML)
        string_cols = df.select_dtypes(include=['object']).columns
        assert string_cols.empty, f"X_train still contains string columns: {list(string_cols)}"

    def test_model_loading_and_prediction(self):
        """
        Load the trained model and make a dummy prediction to ensure compatibility.
        """
        model_path = self.PATHS["model"]
        test_path = self.PATHS["test_data"]

        if not os.path.exists(model_path):
            pytest.skip("Model not found. Run 'src/train.py' first.")
        
        if not os.path.exists(test_path):
            pytest.skip("Test data not found for prediction check.")

        try:
            model = joblib.load(model_path)
            X_test = pd.read_csv(test_path)
        except Exception as e:
            pytest.fail(f"Failed to load model or test data: {e}")

        # Pick the first row to predict
        sample = X_test.iloc[[0]]
        
        try:
            prediction = model.predict(sample)
        except Exception as e:
            pytest.fail(f"Model prediction failed on test sample: {e}")
        
        # Prediction should be an integer (0, 1, 2, or 3) representing P1-P4
        pred_val = prediction[0]
        assert isinstance(pred_val, (int, float, np.integer, np.floating)), \
            f"Prediction returned unexpected type: {type(pred_val)}"

# --- Execution ---
if __name__ == "__main__":
    # Script Mode: python tests/test_pipeline.py
    tester = TestPipeline()
    tester.run_all()