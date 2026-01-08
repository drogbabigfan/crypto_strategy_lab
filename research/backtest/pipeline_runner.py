import subprocess
import sys
import logging

class PipelineRunner:
    def __init__(self, config_path, train_script="research/train.py"):
        self.config_path = config_path
        self.train_script = train_script
        self.logger = logging.getLogger(__name__)

    def run_fold(self, fold_idx, start_date, end_date):
        """
        Run a single training fold in a separate subprocess.
        """
        command = [
            sys.executable,
            self.train_script,
            "--config", self.config_path,
            "--fold", str(fold_idx),
            "--start_date", start_date,
            "--end_date", end_date
        ]
        
        self.logger.info(f"Starting Fold {fold_idx} subprocess...")
        
        try:
            # check=True raises CalledProcessError if return code != 0
            result = subprocess.run(
                command,
                capture_output=True, # Optional: Stream output instead for real-time logs
                text=True,
                check=True
            )
            self.logger.info(f"Fold {fold_idx} completed successfully.")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Fold {fold_idx} failed with error:\n{e.stderr}")
            raise e
