import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from be.kdg.rl.utils import parameters


class QValuesVisual:
    def plot(q_values, count):
        plt.subplot(1, 2, 1)
        plt.suptitle(f'Episode {count+1}')
        plt.title('q-values')
        sns.heatmap(q_values, cmap="Blues", annot=True, cbar=False, square=False, vmin=0, vmax=1)

class PolicyVisual:
    def plot(policy, count):
        plt.subplot(1, 2, 2)
        plt.title('policy π')
        policy_by_action = np.argmax(np.transpose(policy), 1)
        action_mapping = {
            0: [-1, 0],     # Left
            1: [0, -1],     # Down
            2: [1, 0],      # Right
            3: [0, 1]       # Up
        }

        xy_values = [action_mapping[value] for value in policy_by_action]
        x_values = np.reshape([col[0] for col in xy_values], (4,4))
        y_values = np.reshape([col[1] for col in xy_values], (4,4))
        plt.quiver(x_values, y_values)
        plt.savefig(os.path.join(
            parameters.params.get("dirs").get("output"),
            parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
            parameters.current_experiment,
            parameters.params.get("dirs").get("qval"),
            f"episode_{count+1}.png"))
        plt.clf()

class ReturnVisual:
    def plot(x, y, count, title='Average Reward in % (by 100 episodes)'):
        plt.subplot(1, 1, 1)
        plt.title(title)
        plt.plot(x, y)
        plt.savefig(os.path.join(
            parameters.params.get("dirs").get("output"),
            parameters.params.get("experiment").get(parameters.current_experiment).get("environment"),
            parameters.current_experiment,
            parameters.params.get("dirs").get("reward"),
            f"episode_{count+1}.png"))
        plt.clf()