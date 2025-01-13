from datetime import datetime, timedelta, timezone

CLUSTER_EPOCH_2024 = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES = 60 * 4


def get_interval_start_minutes():
    now = datetime.now(timezone.utc)

    minutes_since_epoch = int((now - CLUSTER_EPOCH_2024).total_seconds()) // 60

    interval_start_minutes = minutes_since_epoch - (
        minutes_since_epoch % (CLUSTERED_SUBMISSIONS_INTERVAL_MINUTES)
    )

    return interval_start_minutes


def get_interval_iso_datetime(interval_start_minutes: int):
    return (CLUSTER_EPOCH_2024 + timedelta(minutes=interval_start_minutes)).isoformat()
