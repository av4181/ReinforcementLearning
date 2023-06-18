from abc import abstractmethod

import numpy as np
from numpy import ndarray
import random

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """
    π: ndarray          # policy tabel
    v_values: ndarray
    q_values: ndarray

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        super().__init__(environment, λ, γ, t_max)
        # learning rate alfa = hoeveel moet de originele Q(s,a) veranderd worden.  Evenwicht tussen het belang van het
        # verleden en nieuwe ingewonnen informatie.  alfa =1 te snel, alfa = 0 er wijzigt niets
        self.α = α

        # γ discount factor belang toekomstige rewards vs. directe rewards "beter 1 vogel in de hand dan 2 in de lucht"

        # policy table, initiële policy tabel bevat voor elke state 1/aantal acties.  We hebben BOVEN, BENEDEN, LINKS
        # en RECHTS DUS 0.25 IN DE INITIÊLE TABEL VOOR ELKE STATE
        # shape is het (aantal mogelijk acties, aantal states) en we vullen np.full vvvvvvvv&& alles met 1/4 = 0.25

        self.π = np.full((self.env.n_actions, self.env.state_size), fill_value=1 / self.env.n_actions)

        # state value table, de v() tabel heeft in het begin overal de waarde 0
        self.v_values = np.zeros((self.env.state_size,))

        # state-action table de q() tabel heeft in het begin overal de waarde 0
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))

        # Procedure start met pi - q aanpassen - v aanpassen - terug pi aanpassen

        # total rewards
        self.total_rewards = 0

    @abstractmethod
    def learn(self, episode: Episode):
        # subclasses insert their implementation at this point
        # see for example be\kdg\rl\learning\tabular\qlearning.py
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_learning_start(self):
        # Start episode, teller t telt per episode
        self.t = 0

    def next_action(self, s: int):
        # TODO: COMPLETE THE CODE
        # Implementeer Algoritme 7 of 3 ??
        # Zie ook algoritme 1, agent chooses its next action

        # Gebruik de sample() methode om een random actie te kiezen indien de values in de huidige state nul zijn.
        # Zo niet, dan nemen we de actie met de hoogste value in de huidge state met np.argmax()

        # https://towardsdatascience.com/q-learning-for-beginners-2837b777741

        # Neem de actie met de hoogste value = EXPLOITATION
        # Neem een random actie en probeer nog betere te vinden = EXPLORATION

        # Een afweging tussen deze twee gedragingen is belangrijk: als de agent zich alleen richt op uitbuiting,
        # kan hij geen nieuwe oplossingen uitproberen en leert hij dus niet meer.
        # Aan de andere kant, als de agent alleen willekeurige acties onderneemt, heeft de training geen zin omdat er geen gebruik
        # wordt gemaakt van de Q-tabel.
        # We willen deze parameter dus in de loop van de tijd veranderen: aan het begin van de training willen we de omgeving
        # zoveel mogelijk verkennen.
        # Maar verkenning wordt steeds minder interessant, omdat de agent alle mogelijke staat-actieparen al kent.
        # Deze parameter vertegenwoordigt de mate van willekeur in de actieselectie.

        # Deze techniek wordt gewoonlijk het epsilon-greedy-algoritme genoemd, waarbij epsilon onze parameter is.
        # Het is een eenvoudige maar uiterst efficiënte methode om een goede afweging te maken.
        # Elke keer dat de agent een actie moet ondernemen, heeft hij een kans ε om een willekeurige actie te kiezen,
        # en een kans 1-ε om degene met de hoogste waarde te kiezen.
        # We kunnen de waarde van epsilon aan het einde van elke aflevering verlagen met een vast bedrag (lineair verval),
        # of op basis van de huidige waarde van epsilon (exponentieel verval).

        exploitation_tradeoff = random.uniform(0, 1)
        # als random groter is dan epsilon, neemt de hoogste waarde van uw q tabel in bijhorende state
        if exploitation_tradeoff > self.ε:
            action = np.argmax(self.π[:, s])
        else:
            action = self.env.action_space.sample()  # als random kleiner is dan epsilon, neem een random actie

        return action

    def evaluate(self):
        # TODO: COMPLETE THE CODE
        # Extra groene deel algoritme 3
        # Haal voor elke state in de policy, de actie op en evalueer de Bellman vergelijking
        # v_values[] horen bij die resp. pi tabel
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s, :])

        pass

    def improve(self):
        # TODO: COMPLETE THE CODE
        # De weg terug naar een verbeterde policy met v of q waarden
        # Exploitation vs exploration trade-off
        # a_start is de beste a, dus van welke actie is de bijhorende q waarde de hoogste, dus we zijn geinteresseerd
        # in de index.  Bv. als de eerste q waarde max is, dan is het 0, 2de q waarde 1 etc.
        # tie-breaken
        # nadien pi tabel updaten i.e. kans updaten voor die actie die bij die state hoort
        # als a = a_ster dan doe je 1, als a niet a_ster is dan doe je 2
        # tau is per episode.  Tau blijft dezelfde binnen een episode, verhogen pas na episode zodat epsilon begint
        # te zakken

        # epsilon is de kans op het nemen van een random actie

        for s in range(self.env.state_size):
            best_a = np.argmax(self.q_values[s, :])
            for a in range(self.env.n_actions):
                if best_a == a:
                     self.π[a, s] = 1 - self.ε + self.ε/self.env.n_actions  #greedy
                else:
                     self.π[a, s] = self.ε/self.env.n_actions
        self.decay()


