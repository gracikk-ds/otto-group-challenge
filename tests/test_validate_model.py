from src.models.validate_model import main
from click.testing import CliRunner
from dotenv import load_dotenv

# load env variables
load_dotenv()

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        main,
        [
            "--path_to_dataset",
            "data/processed/test.csv",
            "--path_to_metrics_storage",
            "reports/metrics",
            "--registered_model_name",
            "test_run_model",
            "--experiment_name",
            "test_run_experiment",
        ],
    )
    assert result.exit_code == 0
