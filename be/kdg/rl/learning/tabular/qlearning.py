import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def learn(self, episode: Episode):
        # TODO: COMPLETE THE CODE
        # Zie algoritme 4
        # q waardes updaten, nadien q waardes in de superklasse terug omzetten naar v waardes met evaluate() en tot slot de pi
        # waardes updaten met improve()

        # Formule : q(s,a) = q(s,a) + a*(r + γ * max (q(s',a')) - q(s,a))
        percepts = episode.percepts(1)[0]  # only get the last percept
        s = percepts.state
        a = percepts.action
        r = percepts.reward
        s2 = percepts.next_state
        done = percepts.done

        # state-action function, voor elke s, a berekenen we een utility value met de state action function q
        # die q hoort bij die resp pi tabel
        # 2de soort Bellman vergelijking
        self.q_values[s, a] = self.q_values[s, a] + self.α * \
                              (r + self.γ * (np.max(self.q_values[s2, :]) - self.q_values[s, a]))
        if done:
            self.total_rewards += r

        # compute return
        episode.compute_returns(t=self.t, λ=self.λ)

        super().learn(episode)


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # TODO: COMPLETE THE CODE
        # Zie algoritme 5
        # Je moet de n laatste precepts eruit halen
        # Meerdere q values worden in 1 keer geupdated
        if episode.size >= self.n:
            for p in reversed(episode.percepts(self.n)):
                s = p.state
                a = p.action
                r = p.reward
                s2 = p.next_state
                done = p.done
                # state-action function, voor elke s, a berekenen we een utility value met de state action function q
                # die q hoort bij die resp pi tabel
                # 2de soort Bellman vergelijking
                self.q_values[s, a] = self.q_values[s, a] - self.α * (self.q_values[s, a] -
                                      (r + self.γ * (np.max(self.q_values[s2, :]))))
                if p.done:
                    self.total_rewards += r

        super().learn(episode)

class MonteCarloLearning(TabularLearner):
    # TODO: COMPLETE THE CODE
    # Is hetzelfde als de Nstep methode waarbij n de lengte van de hele episode is
    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        for p in reversed(episode.percepts(episode.size)):
            s = p.state
            a = p.action
            r = p.reward
            s2 = p.next_state
            done = p.done
            self.q_values[s, a] = self.q_values[s, a] - self.α * \
                                  (self.q_values[s, a] - (r + self.γ * (np.max(self.q_values[s2, :]))))
            if p.done:
                self.total_rewards += r
        super().learn(episode)




