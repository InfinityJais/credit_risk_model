from abc import ABC, abstractmethod
import pandas as pd
import joblib
import json
import yaml
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow


class EvaluationStrategy(ABC):
    """
    Abstract Base Class defining the contract for evaluation strategies.
    Every evaluator must implement the 'evaluate' method.
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Calculates metrics and returns them as a dictionary.
        """
        pass

class CreditRiskEvaluator(EvaluationStrategy):
    """
    Specific evaluator for the Credit Risk project (Multi-class).
    Calculates weighted metrics suitable for imbalanced multi-class data.
    """
    def evaluate(self, model, X_test, y_test):
        print("Predicting on test set...")
        y_pred = model.predict(X_test)

        print("Calculating metrics...")
        # 'weighted' average is crucial for multi-class problems with imbalance
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        
        return metrics

class EvaluationPipeline:
    def __init__(self, params):
        self.params = params

    def get_strategy(self) -> EvaluationStrategy:
        """
        Factory method to select the evaluation strategy.
        Currently, we only have one, but this is where you'd switch logic if needed.
        """
        return CreditRiskEvaluator(self.params)

    def run_evaluation(self):

        test_data_path = self.params['data']['x_test']
        test_label_path = self.params['data']['y_test']
        model_path = self.params['train']['model_path']
        # 2. Validation
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at: {test_data_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}. Run 'python src/train.py' first.")

        # Load Artifacts
        print("Loading model and test data...")
        model = joblib.load(model_path)
        X_test = pd.read_csv(test_data_path)
        y_test = pd.Series(pd.read_csv(test_label_path).values.ravel())

        # Select & Execute Strategy
        evaluator = self.get_strategy()
        metrics = evaluator.evaluate(model, X_test, y_test)

        # MLFLOW LOGGING START 
        experiment_name = self.params.get('mlflow', {}).get('experiment_name', 'Credit_Risk_Analysis')
        mlflow.set_experiment(experiment_name)
        
        # We start a run just to log the metrics
        with mlflow.start_run(run_name="Evaluation_Metrics"):
            mlflow.log_metrics(metrics)
            print("Metrics logged to MLflow.")

  
        print(f"Evaluation Complete:")
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" F1 Score: {metrics['f1_score']:.4f}")

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved to 'metrics.json'.")

if __name__ == "__main__":
    if not os.path.exists("params.yaml"):
        raise FileNotFoundError("params.yaml is missing!")
        
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    pipeline = EvaluationPipeline(params)
    pipeline.run_evaluation()