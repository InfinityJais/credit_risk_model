import os
import joblib
import pandas as pd
import pytest
import numpy as np  # <--- Critical Fix: Import NumPy directly
import sys

class TestModelArtifacts:
    """
    Test Suite to verify that all trained models and data artifacts 
    exist and are functional before the API starts.
    """
    
    # Define critical files as a class attribute
    REQUIRED_ARTIFACTS = [
        "models/model.joblib",          # The trained Classifier
        "models/label_encoder.joblib",  # To convert predictions back to P1, P2...
        "models/model_columns.joblib",  # To ensure API input matches training features
        "data/processed/X_test.csv",    # Needed for the prediction test
        "params.yaml"                   # Config file
    ]

    def run_all(self):
        """
        Helper method to run all tests manually (Script Mode).
        """
        print("🚀 Starting Model Artifact Verification...")
        try:
            # 1. Check Files
            for file in self.REQUIRED_ARTIFACTS:
                self.test_artifact_exists(file)
            print(" -> All critical files exist.")

            # 2. Check Prediction
            self.test_model_can_load_and_predict()
            print(" -> Model loaded and predicted successfully.")

            # 3. Check Encoder
            self.test_label_encoder_works()
            print(" -> Label Encoder functional.")
            
            print("\n✅ All Model Artifact tests passed!")
            
        except AssertionError as e:
            print(f"\n❌ Test Failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            sys.exit(1)

    # --- Test Methods (Compatible with Pytest) ---

    @pytest.mark.parametrize("filepath", REQUIRED_ARTIFACTS)
    def test_artifact_exists(self, filepath):
        """
        1. Check if the file exists on disk.
        If this fails, it means 'dvc repro' or 'train.py' hasn't run.
        """
        assert os.path.exists(filepath), f"Missing critical file: {filepath}"

    def test_model_can_load_and_predict(self):
        """
        2. Try to load the model and make one prediction.
        Ensures the file is not corrupted and matches the data schema.
        """
        # Skip logic for Pytest if files are missing
        if not os.path.exists("models/model.joblib") or not os.path.exists("data/processed/X_test.csv"):
            pytest.skip("Model or Test Data missing, skipping prediction test.")

        # Load artifacts
        try:
            model = joblib.load("models/model.joblib")
            X_test = pd.read_csv("data/processed/X_test.csv")
        except Exception as e:
            pytest.fail(f"Failed to load artifacts: {e}")

        # Sanity Check
        assert not X_test.empty, "Test dataset is empty!"

        # Select a single row for prediction
        sample_input = X_test.iloc[[0]]

        # Attempt prediction
        try:
            prediction = model.predict(sample_input)
        except Exception as e:
            pytest.fail(f"Model prediction failed: {str(e)}")

        # Check output type (Fixed using numpy directly)
        pred_value = prediction[0]
        assert isinstance(pred_value, (int, float, np.integer, np.floating)), \
            f"Prediction returned unexpected type: {type(pred_value)}"

    def test_label_encoder_works(self):
        """
        3. Check if the label encoder can invert the prediction (e.g., 0 -> P1).
        """
        path = "models/label_encoder.joblib"
        if not os.path.exists(path):
            pytest.skip("Label Encoder missing.")

        try:
            le = joblib.load(path)
            # Test decoding a dummy class (0)
            decoded_label = le.inverse_transform([0])[0]
            
            # We expect a string like 'P1', 'P2'
            assert isinstance(decoded_label, str), "Label encoder returned non-string label"
            
        except Exception as e:
            pytest.fail(f"Label Encoder failed to decode: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Script Mode: python tests/test_model_file.py
    tester = TestModelArtifacts()
    tester.run_all()