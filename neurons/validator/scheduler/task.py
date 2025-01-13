from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


class TaskStatus:
    UNSCHEDULED = "unscheduled"
    IDLE = "idle"
    RUNNING = "running"


@dataclass
class AbstractTask(ABC):
    """
    Represents a task to be scheduled with an execution interval and a run async function.
    """

    status: Literal["unschedule", "idle", "running"] = field(
        init=False, default=TaskStatus.UNSCHEDULED
    )  # Task status

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def interval_seconds(self) -> float:
        pass

    @abstractmethod
    async def run(self) -> None:
        pass

    def __post_init__(self):
        """
        Perform validation after the dataclass initialization.
        :raises ValueError: If any type / value is invalid (e.g., None or negative interval).
        """
        if (
            not self.name
            or not isinstance(self.name, str)
            or not callable(self.run)
            or not isinstance(self.interval_seconds, float)
            or self.interval_seconds < 0.0
        ):
            raise ValueError("Invalid value.")
