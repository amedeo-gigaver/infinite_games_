import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from bittensor.core.metagraph import MetagraphMixin

from neurons.validator.main import main


class TestValidator:
    def test_validator(self):
        file_path = "neurons/validator.py"

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Assert the expected content
        expected_start = [
            "# -- DO NOT TOUCH BELOW - ENV SET --\n",
            "# flake8: noqa: E402\n",
            "import os\n",
            "import sys\n",
            "\n",
            "# Force torch - must be set before importing bittensor\n",
            'os.environ["USE_TORCH"] = "1"\n',
            "\n",
            "# Add the parent directory of the script to PYTHONPATH\n",
            "script_dir = os.path.dirname(os.path.abspath(__file__))\n",
            "parent_dir = os.path.dirname(script_dir)\n",
            "sys.path.append(parent_dir)\n",
            "# -- DO NOT TOUCH ABOVE --\n",
        ]

        # Check the expected lines
        for i, expected_line in enumerate(expected_start):
            assert lines[i] == expected_line

    @pytest.mark.parametrize(
        "config_env,db_path",
        [
            ("prod", "validator.db"),
            ("test", "validator_test.db"),
        ],
    )
    def test_main(self, config_env, db_path):
        # Patch key dependencies inside the method
        with (
            patch("neurons.validator.main.assert_requirements") as mock_assert_requirements,
            patch("neurons.validator.main.get_config", spec=True) as get_config,
            patch("neurons.validator.main.Dendrite", spec=True),
            patch("neurons.validator.main.Wallet", spec=True),
            patch("neurons.validator.main.Subtensor", spec=True) as MockSubtensor,
            patch("neurons.validator.main.IfGamesClient", spec=True) as MockIfGamesClient,
            patch("neurons.validator.main.DatabaseClient", spec=True) as MockDatabaseClient,
            patch("neurons.validator.main.TasksScheduler") as MockTasksScheduler,
            patch("neurons.validator.main.logger", spec=True) as mock_logger,
            patch("neurons.validator.main.ExportPredictions", spec=True),
            patch("neurons.validator.main.ScorePredictions", spec=True),
        ):
            # Mock get_config
            get_config.return_value = MagicMock(), config_env, db_path

            # Mock Subtensor
            mock_subtensor = MockSubtensor.return_value
            mock_subtensor.metagraph.return_value = MagicMock(spec=MetagraphMixin)

            # Mock Database Client
            mock_db_client = MockDatabaseClient.return_value
            mock_db_client.migrate = AsyncMock()

            # Mock TasksScheduler
            mock_scheduler = MockTasksScheduler.return_value
            mock_scheduler.start = AsyncMock(return_value=None)

            # Mock Logger
            mock_logger.start_session = MagicMock()

            # Run the validator
            asyncio.run(main())

            # Verify assert_requirements() was called
            mock_assert_requirements.assert_called_once()

            # Verify start session called
            mock_logger.start_session.assert_called_once()

            # Verify get_config() was called
            get_config.assert_called_once()

            # Verify DatabaseClient args
            MockDatabaseClient.assert_called_once_with(db_path=db_path, logger=mock_logger)

            # Verify IfGamesClient args
            MockIfGamesClient.assert_called_once_with(
                env=config_env, logger=mock_logger, bt_wallet=ANY
            )

            # Verify migrate() was called
            mock_db_client.migrate.assert_awaited_once()

            # Verify start() was called
            mock_scheduler.start.assert_awaited_once()

            # Verify tasks
            assert mock_scheduler.add.call_count == 5

            # Verify logging
            mock_logger.info.assert_called_with(
                "Validator started",
                extra={
                    "validator_uid": ANY,
                    "validator_hotkey": ANY,
                    "bt_network": ANY,
                    "bt_netuid": ANY,
                    "ifgames_env": config_env,
                    "db_path": db_path,
                    "python": ANY,
                    "sqlite": ANY,
                },
            )
