# Release Notes

## [1.3.4] - 2025-01-13
- **Validator Architecture**: Validator architecture re-implemented to handle scaling events.

## [1.3.3] - 2024-12-17
- **Bittensor Upgrade**: Upgraded to Bittensor version 8.5.1
- **Default `netuid` for Validators**: The `--netuid` argument now defaults to 6, ensuring validators register on the intended subnet when the argument is not explicitly provided.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking
- **Resilient Score Exporting**: Re-exporting functionality added to ensure scores from failed or missed past intervals are included in the current interval.

## [1.3.2] - 2024-12-10
- **Bittensor Upgrade**: Upgraded to Bittensor version 7.4.0.
- **Bug Fix**: Patched a recent common `SSLEOFError` encountered during block number retrieval.
- **Query Optimization**: Miners are now queried at a maximum frequency of once every 5 minutes to reduce unnecessary overhead.
- **Database Enhancements**: Improved data insertion logic for the local database, ensuring better reliability and performance.
- **Logging Improvements**: Enhanced logging and error reporting for better debugging and issue tracking.
- **Code Appearance**: Applied formatting and organization improvements using `black` and `isort`.