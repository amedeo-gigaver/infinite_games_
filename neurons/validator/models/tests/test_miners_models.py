from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.miner import MinersModel


class TestMinersModel:
    def test_create_minimal(self):
        model = MinersModel(
            miner_hotkey="hotkey123",
            miner_uid="uid456",
            registered_date=datetime(2024, 1, 1, 10, 0, 0),
        )

        assert model.miner_hotkey == "hotkey123"
        assert model.miner_uid == "uid456"
        assert model.registered_date == datetime(2024, 1, 1, 10, 0, 0)
        # Optional fields
        assert model.node_ip is None
        assert model.last_updated is None
        assert model.blocktime is None
        # Default value
        assert model.blocklisted is False

    def test_create_with_all_fields(self):
        model = MinersModel(
            miner_hotkey="hotkey789",
            miner_uid="uid999",
            node_ip="192.168.1.10",
            registered_date=datetime(2023, 12, 31, 23, 59, 59),
            last_updated=datetime(2024, 1, 2, 9, 0, 0),
            blocktime=123456789,
            blocklisted=True,
        )

        assert model.miner_hotkey == "hotkey789"
        assert model.miner_uid == "uid999"
        assert model.node_ip == "192.168.1.10"
        assert model.registered_date == datetime(2023, 12, 31, 23, 59, 59)
        assert model.last_updated == datetime(2024, 1, 2, 9, 0, 0)
        assert model.blocktime == 123456789
        assert model.blocklisted is True

    def test_blocklisted_int_to_bool(self):
        # blocklisted as integer should convert to bool
        model = MinersModel(
            miner_hotkey="hotkey111",
            miner_uid="uid222",
            registered_date=datetime(2024, 1, 1, 12, 0, 0),
            blocklisted=1,
        )
        assert model.blocklisted is True

        model2 = MinersModel(
            miner_hotkey="hotkey333",
            miner_uid="uid444",
            registered_date=datetime(2024, 1, 1, 12, 0, 0),
            blocklisted=0,
        )
        assert model2.blocklisted is False

    def test_invalid_data_types(self):
        # miner_hotkey must be a string
        with pytest.raises(ValidationError):
            MinersModel(
                miner_hotkey=123,  # invalid type
                miner_uid="uidABC",
                registered_date=datetime(2024, 1, 1),
            )

        # registered_date must be datetime
        with pytest.raises(ValidationError):
            MinersModel(
                miner_hotkey="hotkeyXYZ", miner_uid="uidABC", registered_date="not-a-datetime"
            )

    def test_primary_key_property(self):
        model = MinersModel(
            miner_hotkey="hk", miner_uid="uid", registered_date=datetime(2024, 1, 1)
        )
        assert model.primary_key == ["miner_hotkey", "miner_uid"]
