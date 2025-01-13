from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.prediction import PredictionsModel


class TestPredictionsModel:
    def test_create_minimal(self):
        # Minimal required fields: unique_event_id, minerUid,
        # interval_start_minutes, interval_agg_prediction
        model = PredictionsModel(
            unique_event_id="event123",
            minerUid="minerABC",
            interval_start_minutes=10,
            interval_agg_prediction=0.75,
        )
        assert model.unique_event_id == "event123"
        assert model.minerUid == "minerABC"
        assert model.interval_start_minutes == 10
        assert model.interval_agg_prediction == 0.75
        # Defaults
        assert model.interval_count == 1
        assert model.exported is False
        assert model.minerHotkey is None
        assert model.blocktime is None
        assert model.submitted is None

    def test_create_full(self):
        model = PredictionsModel(
            unique_event_id="uniqueX",
            minerHotkey="hotkey123",
            minerUid="uid456",
            predictedOutcome="some_outcome",
            canOverwrite=True,
            outcome="final_outcome",
            interval_start_minutes=100,
            interval_agg_prediction=1.23,
            interval_count=5,
            submitted=datetime(2024, 1, 1, 12, 30),
            blocktime=987654321,
            exported=True,
        )
        assert model.unique_event_id == "uniqueX"
        assert model.minerHotkey == "hotkey123"
        assert model.minerUid == "uid456"
        assert model.predictedOutcome == "some_outcome"
        assert model.canOverwrite is True
        assert model.outcome == "final_outcome"
        assert model.interval_start_minutes == 100
        assert model.interval_agg_prediction == 1.23
        assert model.interval_count == 5
        assert model.submitted == datetime(2024, 1, 1, 12, 30)
        assert model.blocktime == 987654321
        assert model.exported is True

    def test_exported_int_to_bool(self):
        # If exported is an integer, it should become a bool
        model = PredictionsModel(
            unique_event_id="event123",
            minerUid="minerABC",
            interval_start_minutes=0,
            interval_agg_prediction=0.5,
            exported=1,
        )
        assert model.exported is True

        model2 = PredictionsModel(
            unique_event_id="event124",
            minerUid="minerDEF",
            interval_start_minutes=5,
            interval_agg_prediction=0.3,
            exported=0,
        )
        assert model2.exported is False

    def test_exported_bool_passthrough(self):
        # If exported is already a boolean, no conversion needed
        model = PredictionsModel(
            unique_event_id="event125",
            minerUid="minerGHI",
            interval_start_minutes=15,
            interval_agg_prediction=0.9,
            exported=False,
        )
        assert model.exported is False

    def test_invalid_data_type(self):
        # interval_start_minutes must be an int
        with pytest.raises(ValidationError):
            PredictionsModel(
                unique_event_id="event126",
                minerUid="minerXYZ",
                interval_start_minutes="not-an-int",
                interval_agg_prediction=0.8,
            )

        # interval_agg_prediction must be a float
        with pytest.raises(ValidationError):
            PredictionsModel(
                unique_event_id="event127",
                minerUid="minerABC",
                interval_start_minutes=10,
                interval_agg_prediction="not-a-float",
            )

    def test_unique_constraints_property(self):
        model = PredictionsModel(
            unique_event_id="event128",
            minerUid="minerABC",
            interval_start_minutes=20,
            interval_agg_prediction=0.4,
        )
        assert model.primary_key == ["unique_event_id", "interval_start_minutes", "minerUid"]
