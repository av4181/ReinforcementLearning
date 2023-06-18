import pygame as pygame

from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning
import gymnasium as gym

if __name__ == '__main__':

# DEZE MAIN METHODE IS OM TE TESTEN OF DE OMGEVING WERKT EN OF ONZE AGENT CORRECT BEWEEGT OVER HET BORD

    pygame.init()
    environment = gym.make("FrozenLake-v1", render_mode="human")
    environment.reset()
    terminated = False

    while True:
        # SPEL MANUEEL SPELEN
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             t, r, terminated, truncated, info = environment.step(0)
        #         if event.key == pygame.K_DOWN:
        #             t, r, terminated, truncated, info = environment.step(1)
        #         if event.key == pygame.K_RIGHT:
        #             t, r, terminated, truncated, info = environment.step(2)
        #         if event.key == pygame.K_UP:
        #             t, r, terminated, truncated, info = environment.step(3)

        # SPEL AUTOMATISCH LATEN LOPEN MET METHODES UIT DE SPACE KLASSE
        action = environment.action_space.sample()

        # t = next state
        # r = reward
        # terminated=True if environment terminates (eg. due to task completion, failure etc.)
        # truncated=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
        t,r, terminated, truncated,info = environment.step(action)

    # Render the environment? toon huidige situatie
        environment.render()
        # Reset wanneer de agent valt of bij het cadeau geraakt.
        if terminated:
            environment.reset()
        # Uittesten of de reward goed wordt teruggegeven bij elke stap
        print(f'Reward = {r}')

    environment.close()




