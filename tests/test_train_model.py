from src.models.train_model import main
from click.testing import CliRunner

# Initialize runner
гunner = CliRunner()


def test_cli_command():
    result = гunner.invoke(
        main,
        ["args"],
    )
    assert result.exit_code == 0
