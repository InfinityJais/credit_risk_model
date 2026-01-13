from abc import ABC, abstractmethod
import pandas as pd
import os
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class PreprocessingStrategy(ABC):
    """
    Abstract Base Class defining the contract for preprocessing.
    """
    def __init__(self, params):
        self.params = params
        self.random_state = params['base']['random_state']
        self.test_size = params['preprocess']['test_size']
        self.encoder_path = params['train'].get('label_encoder_path', 'models/label_encoder.joblib')

    @abstractmethod
    def preprocess(self, df: pd.DataFrame):
        """
        Takes raw dataframe, performs transformations, and returns splits.
        """
        pass


class CreditRiskPreprocessor(PreprocessingStrategy):
    """
    Implements specific logic for Credit Risk:
    - Feature Selection
    - Ordinal Encoding (Education)
    - One-Hot Encoding
    - Label Encoding (Target)
    """
    def preprocess(self, df: pd.DataFrame):
        print("Starting Preprocessing...")

        # Feature Selection (Hardcoded based on EDA & VIF)
        numeric_features = [
            'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
            'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL',
            'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq',
            'num_deliq_6_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts',
            'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq',
            'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
            'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever',
            'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag'
        ]
        
        categorical_features = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        target_col = 'Approved_Flag'
        
        # Validation: Ensure all columns exist
        missing_cols = [col for col in numeric_features + categorical_features + [target_col] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

        # Keep only selected columns
        final_cols = numeric_features + categorical_features + [target_col]
        df = df[final_cols].copy()
        print(f" -> Selected {len(final_cols)} features.")

        # Ordinal Encoding (EDUCATION)
        education_map = {
            'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
        }
        df['EDUCATION'] = df['EDUCATION'].map(education_map).astype(int)

        # One-Hot Encoding
        dummy_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        df = pd.get_dummies(df, columns=dummy_cols)

        # Target Encoding (Create & Save Encoder)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        
        # Save the Label Encoder
        os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)
        joblib.dump(le, self.encoder_path)
        print(f" -> LabelEncoder saved to {self.encoder_path}")

        # Train-Test Split
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f" -> Splitting data (test_size={self.test_size}, random_state={self.random_state})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test


def main():
    if not os.path.exists("params.yaml"):
        raise FileNotFoundError("params.yaml is missing!")
        
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Input
    input_path = params['data']['interim']
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found at: {input_path}. Run clean_data.py first.")

    # Outputs
    x_train_path = params['data']['x_train']
    x_test_path = params['data']['x_test']
    y_train_path = params['data']['y_train']
    y_test_path = params['data']['y_test']

    # Process Data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    processor = CreditRiskPreprocessor(params)
    X_train, X_test, y_train, y_test = processor.preprocess(df)

    # Save Processed Data 
    output_dir = os.path.dirname(x_train_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving splits to {output_dir}...")
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    main()