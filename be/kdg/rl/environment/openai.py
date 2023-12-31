from abc import ABC

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from be.kdg.rl.environment.environment import Environment


class OpenAIGym(Environment, ABC):
    """
    Superclass for all kinds of OpenAI environments
    Wrapper for all OpenAI Environments
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._env: TimeLimit = gym.make(name,render_mode='rgb_array')
        # voor een 8x8 omgeving env = gym.make('FrozenLake-v0', map_name='8x8')

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        self._env.update()

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def state_size(self):
        if self.isdiscrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def isdiscrete(self) -> bool:
        return hasattr(self._env.observation_space, 'n')

    @property
    def name(self) -> str:
        return self._name


class FrozenLakeEnvironment(OpenAIGym):

    def __init__(self) -> None:
        super().__init__(name="FrozenLake-v1")


class CartPoleEnvironment(OpenAIGym):

    # STATE SPACE : MOGELIJKE WAARDEN DIE ZICH KUNNEN VOORDOEN BINNEN DEZE OMGEVING
    # ALLE CONTINUE WAARDEN
    # POSITIE
    # SNELHEID
    # HOEK
    # HOEKSNELHEID
    # ACTIONS SPACE IS EEN DISCRETE ACTIE LINKS OF RECHTS  < -- > MOUNTAIN CAR

    def __init__(self) -> None:
        super().__init__(name='CartPole-v1')    # versie v0 is outdated


# TOEVOEGEN VAN DE NONSLIPPERY ENVIRONMENT

class FrozenLakeNotSlipperyEnvironment(OpenAIGym):
    def __init__(self) -> None:
        gym.envs.registration.register(
            id='FrozenLakeNotSlippery-v1',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': False}
        )
        super().__init__(name="FrozenLakeNotSlippery-v1")