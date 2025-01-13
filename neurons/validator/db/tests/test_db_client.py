import tempfile
from unittest.mock import MagicMock

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestDbClient:
    @pytest.fixture(scope="function")
    def mocked_logger(self) -> MagicMock:
        # Mock logger

        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture(scope="function")
    async def db_client(self, mocked_logger: MagicMock) -> DatabaseClient:
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        db_path = temp_db.name
        temp_db.close()

        client = DatabaseClient(db_path, mocked_logger)

        # Prepare database schema
        await client.script(
            """
                CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT);
            """
        )

        return client

    async def test_add_column_if_not_exists(self, db_client: DatabaseClient):
        await db_client.add_column_if_not_exists(
            table_name="test_table",
            column_name="created_at",
            column_type="DATETIME",
            default_value=None,
        )

        response = await db_client.add_column_if_not_exists(
            table_name="test_table",
            column_name="created_at",
            column_type="DATETIME",
            default_value=None,
        )

        # No error thrown
        assert response is None

    async def test_insert(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        sql = "INSERT INTO test_table (name) VALUES (?) returning name"
        params = ("test_insert",)
        result = await db_client.insert(sql, params)

        assert result == [("test_insert",)]
        mocked_logger.debug.assert_called()  # Ensure logger was called

    async def test_insert_many(self, db_client: DatabaseClient, mocked_logger):
        sql = "INSERT INTO test_table (name) VALUES (?)"
        params = [("test_insert_1",), ("test_insert_2",)]

        result = await db_client.insert_many(sql, params)
        assert result is None

        sql = "SELECT * FROM test_table"
        result = await db_client.many(sql)

        assert result == [(1, "test_insert_1"), (2, "test_insert_2")]
        assert mocked_logger.debug.call_count == 3  # Ensure logger was called

    async def test_insert_many_no_params(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        sql = "INSERT INTO test_table (name) VALUES (?)"
        params = []

        result = await db_client.insert_many(sql, params)
        assert result is None

        sql = "SELECT * FROM test_table"
        result = await db_client.many(sql)

        assert result == []
        assert mocked_logger.debug.call_count == 3  # Ensure logger was called

    async def test_delete(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        # Insert a row to delete
        await db_client.insert("INSERT INTO test_table (name) VALUES ('test_delete')")

        sql = "DELETE FROM test_table WHERE name = ? returning name"
        params = ("test_delete",)
        result = await db_client.delete(sql, params)

        assert result == [("test_delete",)]

        assert mocked_logger.debug.call_count == 3  # Ensure logger was called

    async def test_update(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        # Insert a row to update
        await db_client.insert("INSERT INTO test_table (name) VALUES ('test_update_1')")

        sql = "UPDATE test_table SET name = ? WHERE name = ? returning name"
        params = ("test_update_2", "test_update_1")
        result = await db_client.update(sql, params)

        assert result == [("test_update_2",)]
        assert mocked_logger.debug.call_count == 3  # Ensure logger was called

    async def test_one(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        # Insert a row for querying
        await db_client.insert("INSERT INTO test_table (name) VALUES ('test_one')")

        sql = "SELECT * FROM test_table WHERE name = ?"
        params = ("test_one",)
        result = await db_client.one(sql, params)

        assert result == (1, "test_one")  # Verify returned row
        assert mocked_logger.debug.call_count == 3  # Ensure logger was called

    async def test_many(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        # Insert multiple rows for querying
        await db_client.insert(
            "INSERT INTO test_table (name) VALUES ('test_many_1'), ('test_many_2')",
        )

        sql = "SELECT * FROM test_table"
        result = await db_client.many(sql)

        assert result == [(1, "test_many_1"), (2, "test_many_2")]  # Verify returned rows
        assert mocked_logger.debug.call_count == 3  # Ensure logger was called
        mocked_logger.warning.assert_not_called()  # No warning since rows are few

    async def test_many_with_warning(self, db_client: DatabaseClient, mocked_logger: MagicMock):
        # Insert many rows for testing pagination warning
        sql = "INSERT INTO test_table (name) VALUES (?)"

        rows_to_insert = 501

        for i in range(rows_to_insert):
            params = (f"Name {i}",)
            await db_client.insert(sql, params)

        sql = "SELECT * FROM test_table"
        result = await db_client.many(sql)

        assert len(result) == rows_to_insert  # Verify returned rows
        mocked_logger.warning.assert_called_with(
            "Query returning many rows", extra={"rows": rows_to_insert}
        )  # Ensure warning logged

    async def test_error(self, db_client: DatabaseClient, mocked_logger: InfiniteGamesLogger):
        with pytest.raises(Exception, match="no such table: fake_table"):
            await db_client.insert(
                "INSERT INTO fake_table (name) VALUES ('test_many_1'), ('test_many_2')",
            )

        assert mocked_logger.debug.call_count == 1  # Only called for creating the table
        mocked_logger.exception.assert_called()

    async def test_migrate(self, db_client: DatabaseClient):
        await db_client.migrate()

        for table_name in ["events", "miners", "predictions"]:
            table = await db_client.one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            )

            assert table is not None  # Check table was created
