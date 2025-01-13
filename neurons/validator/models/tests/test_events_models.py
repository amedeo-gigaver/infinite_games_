from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.event import EventsModel, EventStatus


class TestEventsModel:
    def test_create_with_enum_status(self):
        event = EventsModel(
            unique_event_id="unique1",
            event_id="event1",
            market_type="truncated_market1",
            event_type="market1",
            description="desc1",
            starts="2024-12-03T00:00:00Z",
            resolve_date="2024-12-04T00:00:00Z",
            outcome="outcome1",
            status=EventStatus.PENDING,
            metadata='{"key": "value"}',
            created_at="2012-12-02T14:30:00+00:00",
            cutoff="2000-12-30T14:30:00+00:00",
        )
        assert event.status == EventStatus.PENDING
        assert isinstance(event.created_at, datetime)
        assert isinstance(event.cutoff, datetime)

    def test_create_with_integer_status(self):
        # Using integer for status (2 is PENDING)
        event = EventsModel(
            unique_event_id="unique2",
            event_id="event2",
            market_type="truncated_market2",
            event_type="market2",
            description="desc2",
            starts="2024-12-03",
            resolve_date="2024-12-04",
            outcome="outcome2",
            status=2,
            metadata='{"key": "value"}',
            created_at="2012-12-02T14:30:00+00:00",
            cutoff="2000-12-30T14:30:00+00:00",
        )
        assert event.status == EventStatus.PENDING  # 2 maps to PENDING

    def test_invalid_status_value(self):
        # Status = 99 is not a valid status
        with pytest.raises(ValidationError) as exc:
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market3",
                description="desc3",
                starts=None,
                resolve_date=None,
                outcome=None,
                status=99,  # invalid
                metadata="{}",
                created_at="2012-12-02T14:30:00Z",
                cutoff=None,
            )
        assert "Invalid status: 99" in str(exc.value)

    def test_processed_and_exported_as_integers(self):
        # processed=1 (True), exported=0 (False)
        event = EventsModel(
            unique_event_id="unique4",
            event_id="event4",
            market_type="truncated_market4",
            event_type="market4",
            description="desc4",
            starts=None,
            resolve_date=None,
            outcome=None,
            status=EventStatus.SETTLED,
            metadata="{}",
            processed=1,  # should become True
            exported=0,  # should become False
            created_at="2012-12-02T14:30:00Z",
            cutoff=None,
        )
        assert event.processed is True
        assert event.exported is False

    def test_optional_fields_none(self):
        event = EventsModel(
            unique_event_id="unique5",
            event_id="event5",
            market_type="truncated_market5",
            event_type="market5",
            description="desc5",
            status=EventStatus.PENDING,
            metadata="{}",
            created_at="2012-12-02T14:30:00Z",
            # Omitting starts, resolve_date, outcome, cutoff, local_updated_at, processed, exported
        )
        assert event.starts is None
        assert event.resolve_date is None
        assert event.outcome is None
        assert event.cutoff is None
        assert event.local_updated_at is None
        # processed and exported default to False if not provided
        assert event.processed is False
        assert event.exported is False
