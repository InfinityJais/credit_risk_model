from abc import ABC, abstractmethod
import pandas as pd
import joblib
import yaml
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models import infer_signature
from mlflow import pytorch, sklearn, xgboost as mlflow_xgb


class ModelStrategy(ABC):
    """
    Abstract Base Class for all models
    """
    def __init__(self, params):
        self.params = params
        self.random_state = params['base']['random_state']

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model and return the fitted object.
        """
        pass

class RandomForestStrategy(ModelStrategy):
    def train(self, X_train, y_train):
        # Access nested params safely
        n_estimators = self.params['random_forest']['n_estimators']
        print(f"Initializing Random Forest (n_estimators={n_estimators})...")
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        clf.fit(X_train, y_train)
        return clf

class XGBoostStrategy(ModelStrategy):
    def train(self, X_train, y_train):
        print("Initializing XGBoost...")
        
        # we add numeric labels (0, 1, 2, 3) because XGBoost requires it.  
        # We ensure labels are encoded in preprocess.py
        xgb_params = self.params['xgboost']
        
        clf = xgb.XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth=xgb_params['max_depth'],
            random_state=self.random_state,
            objective='multi:softprob',
            num_class=xgb_params['num_class']
        )
        clf.fit(X_train, y_train)
        return clf

# Factory (Decides which model to use) 
class ModelTrainer:
    def __init__(self, params):
        self.params = params
        self.model_type = params['train']['model_type']

    def get_strategy(self) -> ModelStrategy:
        """Factory method to select the correct strategy."""
        if self.model_type == 'random_forest':
            return RandomForestStrategy(self.params)
        elif self.model_type == 'xgboost':
            return XGBoostStrategy(self.params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def run_training(self):

        x_train_path = self.params['data']['x_train']
        y_train_path = self.params['data']['y_train']

        print("Loading processed data...")
        
        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"Training data not found at: {x_train_path}. Run 'python src/preprocess.py' first.")

        if not os.path.exists(y_train_path):
             raise FileNotFoundError(f"Label data not found at: {y_train_path}. Run 'python src/preprocess.py' first.")

        X_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel() 

        # Set the experiment name for MLflow
        experiment_name = self.params.get('mlflow', {}).get('experiment_name', 'Default_Experiment')
        mlflow.set_experiment(experiment_name)
        
        print(f"Starting MLflow Run in experiment: {experiment_name}")
        
        with mlflow.start_run():
            # Log Hyperparameters
            mlflow.log_param("model_type", self.model_type)
            
            # Log only the params relevant to the chosen model
            if self.model_type == 'random_forest':
                mlflow.log_params(self.params['random_forest'])
            elif self.model_type == 'xgboost':
                mlflow.log_params(self.params['xgboost'])

            # Train Model
            strategy = self.get_strategy()
            print(f"Training model: {self.model_type}...")
            model = strategy.train(X_train, y_train)

            try:
                if model is None:
                    raise ValueError("Model training failed - model is None")
                
                signature = infer_signature(X_train, model.predict(X_train))
                
                if self.model_type == 'xgboost':
                    mlflow.log_params(self.params['xgboost'])
                
                if self.model_type == 'random_forest':
                    sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=X_train.iloc[:5]
                    )
                elif self.model_type == 'xgboost':
                    mlflow_xgb.log_model(
                        xgb_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=X_train.iloc[:5]
                    )
                
                print(" -> MLflow logging complete.")
            except Exception as e:
                print(f"MLflow logging failed: {e}")
                raise

        # Save Artifacts
        model_path = self.params['train']['model_path']
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        
        # Save column names Critical for API alignment
        columns_path = self.params['train']['columns_path']
        joblib.dump(X_train.columns.tolist(), columns_path)
        
        print(f"Success! Model saved to models/model.joblib")


if __name__ == "__main__":

    if not os.path.exists("params.yaml"):
        raise FileNotFoundError("params.yaml is missing!")
        
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)


    trainer = ModelTrainer(params)
    trainer.run_training()