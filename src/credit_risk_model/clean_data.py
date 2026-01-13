from abc import ABC, abstractmethod
import pandas as pd
import yaml
import os

class CleaningStrategy(ABC):
    """
    This is an Abstract Base Class (ABC) that defines the interface
    for different cleaning strategies. Any new cleaning strategy must
    inherit from this class and implement the 'clean' method.
    """
    def __init__(self, params):
        self.params = params
        self.missing_val = params['clean']['missing_value']
        self.threshold = params['clean']['null_threshold']

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Every child class MUST implement this method.
        If they don't, Python will throw an error.
        """
        pass

class ThresholdCleaner(CleaningStrategy):
    """
    This class inherits from CleaningStrategy and implements the specific
    logic to drop columns/rows based on the -99999 threshold.
    """
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Starting Threshold Cleaning (Threshold: {self.threshold})...")
        
        # Identify columns with -99999 counts above threshold
        cols_to_remove = []
        for col in df.columns:
            count = (df[col] == self.missing_val).sum()
            if count > self.threshold:
                cols_to_remove.append(col)
        
        if cols_to_remove:
            print(f" -> Dropping {len(cols_to_remove)} columns: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
        
        # Drop rows which have -99999 
        # Keep rows where ALL columns are NOT equal to missing_val
        initial_count = len(df)
        df_clean = df[(df != self.missing_val).all(axis=1)]
        dropped_count = initial_count - len(df_clean)
        
        print(f" -> Dropped {dropped_count} rows containing {self.missing_val}")
        return df_clean

# The Execution Logic 
def main():
    # Load Params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    cleaner = ThresholdCleaner(params)


    # Extract paths from params file
    bank_path = params['data']['raw_bank']
    cibil_path = params['data']['raw_cibil']


    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Raw Bank data file not found at: {bank_path}")

    if not os.path.exists(cibil_path):
        raise FileNotFoundError(f"CIBIL raw data file not found at: {cibil_path}")


    print("Loading Data...")
    df1 = pd.read_excel(bank_path)
    df2 = pd.read_excel(cibil_path)


    print("\n--- Processing Bank Data ---")
    df1 = cleaner.clean(df1)

    print("\n--- Processing CIBIL Data ---")
    df2 = cleaner.clean(df2)


    print("\nMerging...")
    df_merged = pd.merge(df1, df2, how='inner', on='PROSPECTID')
    
    os.makedirs('data/interim', exist_ok=True)
    df_merged.to_csv('data/interim/merged_data.csv', index=False)
    print(f"Done. Final Shape: {df_merged.shape}")

if __name__ == "__main__":
    main()