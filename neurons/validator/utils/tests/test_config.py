from unittest.mock import MagicMock, patch

import pytest
from bittensor.core.config import Config

from neurons.validator.utils.config import VALID_NETWORK_CONFIGS, get_config


class TestConfig:
    @pytest.fixture
    def mock_subtensor(self):
        with patch("neurons.validator.utils.config.Subtensor") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_logging_machine(self):
        with patch("neurons.validator.utils.config.LoggingMachine") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_wallet(self):
        with patch("neurons.validator.utils.config.Wallet") as mock:
            # Mock the add_args method
            mock.add_args = MagicMock()
            yield mock

    @pytest.fixture
    def mock_config(self):
        with patch("neurons.validator.utils.config.Config") as mock:
            yield mock

    def create_mock_args(self, netuid=None, network=None, ifgames_env=None):
        """Helper method to create a mock args object with the required attributes"""
        mock_args = MagicMock()

        mock_args.__getattribute__ = MagicMock()
        mock_args.__getattribute__.side_effect = lambda x: {
            "netuid": netuid,
            "subtensor.network": network,
            "ifgames.env": ifgames_env,
        }.get(x)

        return mock_args

    def test_arg_additions(self, mock_subtensor, mock_logging_machine, mock_wallet):
        """Test that all component add_args methods are called"""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=6, network="finney", ifgames_env=None
            )

            config, env, db_path = get_config()

            mock_subtensor.add_args.assert_called_once()
            mock_logging_machine.add_args.assert_called_once()
            mock_wallet.add_args.assert_called_once()

            assert isinstance(config, Config)
            assert env == "prod"
            assert db_path == "validator.db"

    def test_required_args_missing(self):
        """Test behavior when required arguments are missing"""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=None, network=None, ifgames_env=None
            )

            with pytest.raises(ValueError):
                get_config()

    @pytest.mark.parametrize("test_config", VALID_NETWORK_CONFIGS)
    def test_valid_network_configs_constant(self, test_config: dict[str, any]):
        """Test that all configurations in VALID_NETWORK_CONFIGS are properly configured"""

        assert isinstance(test_config, dict)
        assert "subtensor.network" in test_config
        assert "netuid" in test_config
        assert "ifgames.env" in test_config
        assert isinstance(test_config["netuid"], int)
        assert test_config["subtensor.network"] in ["finney", "test", "local"]
        assert test_config["ifgames.env"] in ["prod", "test", None]

    @pytest.mark.parametrize(
        "test_args,expected_valid,expected_env,expected_db_path",
        [
            ((6, "finney", None), True, "prod", "validator.db"),
            ((155, "test", None), True, "test", "validator_test.db"),
            ((6, "local", "prod"), True, "prod", "validator.db"),
            ((155, "local", "test"), True, "test", "validator_test.db"),
            ((7, "finney", None), False, None, None),
            ((6, "invalid", None), False, None, None),
            ((6, "local", None), False, None, None),
            ((155, "local", "invalid"), False, None, None),
        ],
    )
    def test_configurations(
        self,
        test_args: tuple,
        expected_valid: bool,
        expected_env: str | None,
        expected_db_path: str | None,
        mock_config,
    ):
        """Test various network configurations for validity"""
        netuid, network, ifgames_env = test_args
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = self.create_mock_args(
                netuid=netuid, network=network, ifgames_env=ifgames_env
            )

            if expected_valid:
                _, env, db_path = get_config()

                mock_config.assert_called_once()

                # Assert env
                assert env == expected_env

                # Assert db_path
                assert db_path == expected_db_path
            else:
                with pytest.raises(ValueError) as exc_info:
                    get_config()
                assert "Invalid netuid" in str(exc_info.value)
