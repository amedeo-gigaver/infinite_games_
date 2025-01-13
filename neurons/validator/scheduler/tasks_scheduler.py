import asyncio
import time

from neurons.validator.scheduler.task import AbstractTask, TaskStatus
from neurons.validator.utils.logger.logger import InfiniteGamesLogger


class TasksScheduler:
    """
    Scheduler for managing and running multiple asynchronous tasks.
    """

    __tasks: list[AbstractTask]
    __logger: InfiniteGamesLogger

    def __init__(self, logger: InfiniteGamesLogger):
        # Validate logger
        if not isinstance(logger, InfiniteGamesLogger):
            raise TypeError("logger must be an instance of InfiniteGamesLogger.")

        self.__tasks = []  # List to store tasks
        self.__logger = logger

    async def __schedule_task(self, task: AbstractTask):
        """
        Private method to manage the execution of a single task.
        :param task: The task to be scheduled and executed.
        """
        while True:  # Continuously execute the task at the specified interval
            start_time = time.time()

            # Start a new trace
            self.__logger.start_trace()

            task.status = TaskStatus.RUNNING  # Mark the task as running

            self.__logger.info("Task started", extra={"task_name": task.name})

            try:
                # Execute the task's run async function
                await task.run()

                elapsed_time_ms = round((time.time() - start_time) * 1000)

                self.__logger.info(
                    "Task finished",
                    extra={"task_name": task.name, "elapsed_time_ms": elapsed_time_ms},
                )

            except Exception:
                # Log any exceptions that occur during task execution
                elapsed_time_ms = round((time.time() - start_time) * 1000)

                self.__logger.exception(
                    "Task errored",
                    extra={"task_name": task.name, "elapsed_time_ms": elapsed_time_ms},
                )

            task.status = TaskStatus.IDLE  # Mark the task as idle after completion

            # Wait for the specified interval before the next execution
            await asyncio.sleep(task.interval_seconds)

    def add(self, task: AbstractTask):
        """
        Add a new task to the scheduler.
        :param task: The Task object to add.
        """
        tasks_names = {item.name for item in self.__tasks}

        if task.name in tasks_names:
            raise ValueError(f"Task '{task.name}' already added")

        self.__tasks.append(task)  # Append the task to the internal list

    async def start(self):
        """
        Start all tasks that are in the "unscheduled" state.
        This method creates a new asyncio task for each task in the scheduler.
        """
        scheduled_tasks = []  # List to hold scheduled asyncio tasks

        for task in self.__tasks:
            if task.status != TaskStatus.UNSCHEDULED:  # Skip tasks that are already scheduled
                continue

            # Schedule and start the task
            scheduled_tasks.append(self.__schedule_task(task))

        # Await all tasks
        await asyncio.gather(*scheduled_tasks)
