import subprocess
import sys
import os
import yaml

class MLPipeline:
    """
    Orchestrates the execution of the Machine Learning pipeline steps.
    """
    def __init__(self, config_path="params.yaml"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
            
        with open(config_path, "r") as f:
            self.params = yaml.safe_load(f)

        pipeline_conf = self.params.get("pipeline_config", {})
        base_dir = pipeline_conf.get("base_dir", "src") 
        steps = pipeline_conf.get("steps", [])

        # Dynamically build the full paths
        # This handles OS differences (Windows '\' vs Linux '/') automatically
        self.pipeline_steps = [
            os.path.join(base_dir, step) for step in steps
        ]
        
        for step in self.pipeline_steps:
            if not os.path.exists(step):
                print(f"Warning: Pipeline step file not found: {step}")

    def _run_step(self, script_path: str):
        """
        Internal method to run a single python script using subprocess.
        Raises an error if the script fails.
        """
        if not os.path.exists(script_path):
            print(f"Critical Error: Script not found at {script_path}")
            sys.exit(1)

        print(f"-------- Running {script_path} --------")
        
        # Use sys.executable to ensure we use the same python interpreter (venv)
        result = subprocess.run([sys.executable, script_path], capture_output=False)
        
        if result.returncode != 0:
            print(f"Pipeline Failed: {script_path} exited with code {result.returncode}.")
            sys.exit(result.returncode)
            
        print(f"{script_path} completed successfully.\n")

    def run(self):
        """
        Executes the entire pipeline sequentially.
        """
        print("Starting MLOps Pipeline...\n")
        
        for step in self.pipeline_steps:
            self._run_step(step)
            
        print("Pipeline finished successfully!")

if __name__ == "__main__":
    # Instantiate and run
    pipeline = MLPipeline()
    pipeline.run()