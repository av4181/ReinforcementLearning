from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract Environment
    Superclass for all kinds of environments
    """
    # Environment moet na een episode telkens gereset worden
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    # environment moet telkens gerendered worden
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass
    #Gym bibliotheel kan direct ook het aantal states en actions geven met "env.observation_space.n" and "env.action_space.n"
    @property
    @abstractmethod
    def observation_space(self):
        pass

    @property
    @abstractmethod
    def n_actions(self):
        pass

    @property
    @abstractmethod
    def state_size(self):
        pass

    @property
    @abstractmethod
    def isdiscrete(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
