import asyncio
from unittest.mock import MagicMock

import pytest

from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.scheduler.tasks_scheduler import TasksScheduler
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TestTasksScheduler:
    @pytest.fixture(scope="class")
    def await_start_with_timeout(self):
        async def _await_start_with_timeout(start_future, timeout):
            try:
                return await asyncio.wait_for(start_future, timeout)  # Run with a timeout
            except asyncio.TimeoutError:
                pass

        return _await_start_with_timeout

    @pytest.fixture
    def logger(self):
        return MagicMock(spec=InfiniteGamesLogger)

    @pytest.fixture(scope="function")
    def scheduler(self, logger):
        return TasksScheduler(logger=logger)

    async def test_scheduler_no_tasks(self, scheduler):
        # Test that the scheduler doesn't crash when no tasks are added
        await scheduler.start()

    async def test_add_task(self, scheduler):
        class TestTask(AbstractTask):
            @property
            def name(self):
                return "Test Task"

            @property
            def interval_seconds(self):
                return 5.0

            async def run(self):
                pass

        task = TestTask()
        scheduler.add(task)

        assert len(scheduler._TasksScheduler__tasks) == 1
        assert scheduler._TasksScheduler__tasks[0].name == "Test Task"

    async def test_schedule_task_execution(self, logger, scheduler, await_start_with_timeout):
        runs = 0

        interval_seconds = 0.5

        class TestTask(AbstractTask):
            @property
            def name(self):
                return "Test Task"

            @property
            def interval_seconds(self):
                return interval_seconds

            async def run(self):
                nonlocal runs
                runs += 1

        task = TestTask()
        scheduler.add(task)

        # Run the scheduler for a short time to allow task to execute
        await await_start_with_timeout(
            start_future=scheduler.start(), timeout=interval_seconds * 2.1
        )

        # Ensure that task iterated 3 times
        assert logger.start_trace.call_count == 3
        assert runs == 3

        # Check info logs
        assert logger.info.call_count == 6

    async def test_tasks_run_concurrently_at_intervals(
        self, logger, scheduler, await_start_with_timeout
    ):
        interval_seconds = 0.5

        runs_task_1 = 0
        runs_task_2 = 0

        class TestTask1(AbstractTask):
            @property
            def name(self):
                return "Test Task 1"

            @property
            def interval_seconds(self):
                return interval_seconds

            async def run(self):
                nonlocal runs_task_1
                runs_task_1 += 1

        class TestTask2(AbstractTask):
            @property
            def name(self):
                return "Test Task 2"

            @property
            def interval_seconds(self):
                return interval_seconds

            async def run(self):
                nonlocal runs_task_2
                runs_task_2 += 1

        task_1 = TestTask1()
        task_2 = TestTask2()

        scheduler.add(task_1)
        scheduler.add(task_2)

        # Run the scheduler for a short time to allow tasks to execute
        await await_start_with_timeout(
            start_future=scheduler.start(), timeout=interval_seconds * 2.5
        )

        # Ensure that the tasks were executed N times
        assert runs_task_1 == 3
        assert runs_task_2 == 3
        assert logger.start_trace.call_count == 6

    async def test_task_execution_error(self, scheduler, await_start_with_timeout, logger):
        runs = 0

        interval_seconds = 0.5

        class TestTask(AbstractTask):
            @property
            def name(self):
                return "Test Task"

            @property
            def interval_seconds(self):
                return interval_seconds

            # Async function that will simulate an error
            async def run(self):
                nonlocal runs
                runs += 1

                raise Exception("Simulated error")

        task = TestTask()
        scheduler.add(task)

        # Run the scheduler for a short time to allow task to execute
        await await_start_with_timeout(
            start_future=scheduler.start(), timeout=interval_seconds * 1.5
        )

        # Ensure that the task was executed by checking the event
        assert runs == 2

        # Check that the task status is "idle" after execution
        assert task.status == "idle"

        # Check logs
        assert logger.info.call_count == 2  # task started logs
        assert logger.exception.call_count == 2  # task errored logs

    async def test_task_with_invalid_status(self, scheduler, await_start_with_timeout):
        # Verify the task is not executed because its status is not "unscheduled"
        runs = 0

        async def dummy_task():
            nonlocal runs
            runs += 1

        class TestTask(AbstractTask):
            @property
            def name(self):
                return "Test Task"

            @property
            def interval_seconds(self):
                return 0.01

            async def run(self):
                nonlocal runs
                runs += 1

        # Create a task and manually set it to "idle"
        task = TestTask()
        task.status = "idle"
        scheduler.add(task)

        # Run the scheduler asynchronously
        await await_start_with_timeout(start_future=scheduler.start(), timeout=1)

        # Verify that the task function is NOT executed because it is already started
        assert runs == 0
        assert task.status == "idle"
