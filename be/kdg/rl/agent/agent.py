from abc import abstractmethod
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import os

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner
from be.kdg.rl.utils import parameters
from be.kdg.rl.utils.grafieken import ReturnVisual, QValuesVisual, PolicyVisual


# AAN AGENT CLASS WERD NIETS GEWIJZIGD
class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=50) -> None:
        self.stats = pd.DataFrame({
            "episode_nr": np.arange(1, n_episodes + 2, 1),
            "total_reward": np.empty(n_episodes + 1, dtype=int),
            "avg_reward": np.empty(n_episodes + 1)
        })
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(TabularAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state, _ = self.env.reset()
            # reset the learning strategy
            self.learning_strategy.on_learning_start()

            # while the episode isn't finished by length
            # Implementatie algortime 1 (roept telkens algoritme 3 op)
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action : next state and the corresponding reward
                # step method returns a tuple with values (s', r, terminated, truncated, info)
                t, r, terminated, truncated, info = self.env.step(action)

                # render environment (don't render every step, only every X-th, or at the end of the learning process)
                #self.env.render()

                # create Percept object from observed values state,action,r,s' (SARS') and terminate flag, but
                # ignore values truncated and info
                percept = Percept((state, action, r, t, terminated))

                # add the newly created Percept to the Episode
                episode.add(percept)

                # update Agent's state
                state = percept.next_state

                # learn from Percepts in Episode
                # Agent gaat telkens de learn methode van tabular learner oproepen in geval van frozen lake
                # In het algemeen gaat telkens het algoritme van de van toepassing zijnde learning strategy worden
                # opgeroepen.
                self.learning_strategy.learn(episode)

                # learn from one or more Percepts in the Episode
                self.learning_strategy.learn(episode)

                # update Agent's state
                state = percept.next_state

                # break if episode is over
                if percept.done:
                    self.results()
                    break

            # end episode
            self.episode_count += 1

        self.stats.to_pickle(
            os.path.join(
                parameters.params.get("dirs").get("output"),
                parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
                parameters.current_experiment,
                "results.pkl"))
        self.stats.to_csv(
            os.path.join(
                parameters.params.get("dirs").get("output"),
                parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
                parameters.current_experiment,
                "results.csv"))

        self.env.close()

    def results(self):
        self.stats.at[self.episode_count, 'total_reward'] = self.learning_strategy.total_rewards
        self.stats.at[self.episode_count, 'avg_reward'] = \
            np.round(self.learning_strategy.total_rewards / (self.episode_count + 1) * 100, 1)

        if self.episode_count == 0 or (self.episode_count + 1):
            # print(f'Total rewards after {self.episode_count + 1} episodes: '
            #         f'$$$ {self.stats.total_reward[self.episode_count]} '
            #         f'({self.stats.avg_reward[self.episode_count]}%) $$$'
            # )
            # print(self.learning_strategy.π)
            ReturnVisual.plot(self.stats.episode_nr[:self.episode_count], self.stats.avg_reward[:self.episode_count], self.episode_count)
            QValuesVisual.plot(self.learning_strategy.q_values, self.episode_count)
            PolicyVisual.plot(self.learning_strategy.π, self.episode_count)


class DQNAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=199) -> None:
        self.stats = pd.DataFrame({
            "episode_nr": np.arange(1, n_episodes + 2, 1),
            "timesteps": np.empty(n_episodes + 1, dtype=int),
            "avg_timesteps": np.empty(n_episodes + 1, dtype=int)
        })
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(DQNAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state = self.env.reset()
            # reset the learning strategy
            self.learning_strategy.on_learning_start()

            # Added episode count for easier tracking of episodes
            print(f'\n\nEpisode {self.episode_count + 1}')
            #print(f'=============================')

            # while the episode isn't finished by length
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action: the next_state and the corresponding reward
                observation = self.env.step(action)[:-1]
                # render environment
                #self.env.render()

                # step method returns a tuple with values (s', r, terminated, truncated, info)
                t, r, terminated, truncated, info = self.env.step(action)
                # create Percept from s,a,r,s' and add to Episode

                # create Percept object from observed values state,action,r,s' (SARS') and terminate flag, but
                # ignore values truncated and info
                percept = Percept((state, action, r, t, terminated))
                # percept = Percept((state, action) + observation)
                episode.add(percept)

                # learn from one or more Percepts in the Episode
                self.learning_strategy.learn(episode)

                # update Agent's state
                state = percept.next_state

                # break if episode is over
                if percept.done:
                    self.results()
                    break

            # end episode
            self.episode_count += 1

        self.stats.to_pickle(
            os.path.join(
                parameters.params.get("dirs").get("output"),
                parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
                parameters.current_experiment,
                "results.pkl"))
        self.stats.to_csv(
            os.path.join(
                parameters.params.get("dirs").get("output"),
                parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
                parameters.current_experiment,
                "results.csv"))
        self.env.close()

    def results(self):
        if self.learning_strategy.max_timesteps < self.learning_strategy.t:
            self.learning_strategy.max_timesteps = self.learning_strategy.t

        self.stats.at[self.episode_count, 'timesteps'] = self.learning_strategy.t #total rewards is nr timesteps, the more timesteps in an episode the better
        self.stats.at[self.episode_count, 'avg_timesteps'] = np.round(self.learning_strategy.t / (self.episode_count + 1) * 100, 1)   # total rewards is nr timesteps
        print(f"Timesteps balanced: $$ {self.learning_strategy.t} $$ ")
        print(f"Maximum: == {self.learning_strategy.max_timesteps} ==")
        # self.stats.at[self.episode_count, 'avg_reward'] = \
        #     np.round(self.learning_strategy.total_rewards / (self.episode_count + 1) * 100, 1)

        if self.episode_count == 0 or (self.episode_count + 1) :
            ReturnVisual.plot(self.stats.episode_nr[:self.episode_count],
                              self.stats.timesteps[:self.episode_count],
                              self.episode_count,
                              f"Timesteps (by 10 episodes)"
        )

