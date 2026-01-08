import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from research.backtest.pipeline_runner import PipelineRunner

def test_pipeline_runner_run_fold():
    """Test that PipelineRunner constructs the correct command."""
    
    # Mock subprocess.run to avoid actual execution overhead
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        
        runner = PipelineRunner(
            config_path="config/config.yaml",
            train_script="research/train.py"
        )
        
        # Run Fold 0
        runner.run_fold(fold_idx=0, start_date="2024-01-01", end_date="2024-02-01")
        
        # Verify call
        args, kwargs = mock_run.call_args
        command = args[0]
        
        # Command should look like: [python, train.py, --config, ..., --fold, 0, ...]
        assert command[0] == sys.executable
        assert command[1] == "research/train.py"
        assert "--fold" in command
        assert "0" in command
        assert "--start_date" in command
        assert "2024-01-01" in command
