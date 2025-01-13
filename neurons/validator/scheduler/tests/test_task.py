import pytest

from neurons.validator.scheduler.task import AbstractTask


class TestTask:
    def test_task_initialization_valid(self):
        class ValidTask(AbstractTask):
            @property
            def name(self):
                return "Test Task"

            @property
            def interval_seconds(self):
                return 5.0

            async def run(self):
                pass

        task = ValidTask()

        assert task.name == "Test Task"
        assert task.interval_seconds == 5.0
        assert task.status == "unscheduled"
        assert callable(task.run)

    def test_task_initialization_invalid(self):
        class ValidTask(AbstractTask):
            @property
            def name(self):
                return "Valid Task"

            @property
            def interval_seconds(self):
                return 5.0

            async def run(self):
                pass

        class InvalidName(ValidTask):
            @property
            def name(self):
                return None

        class NegativeInterval(ValidTask):
            @property
            def interval_seconds(self):
                return -1.0

        class IntegerInterval(ValidTask):
            @property
            def interval_seconds(self):
                return 5

        class UndefinedRun(AbstractTask):
            @property
            def name(self):
                return "Valid Task"

            @property
            def interval_seconds(self):
                return 5.0

        with pytest.raises(ValueError):
            InvalidName()

        with pytest.raises(ValueError):
            NegativeInterval()

        with pytest.raises(ValueError):
            IntegerInterval()

        with pytest.raises(TypeError):
            UndefinedRun()
