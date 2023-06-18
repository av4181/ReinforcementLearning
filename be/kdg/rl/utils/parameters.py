import os

# Hier kunnen de verschillende variabelen aangepast worden en kan de keuze van environment, agent en strategy gekozen worden
current_experiment = "experiment08"
n_episodes = 500
output_freq = 10        # met welke frequentie wordt er output gegenereerd
update_interval = 10    # update interval voor deep learning

def init():
    init_folders(current_experiment)


params = {
    # directories voor resultaten
    "dirs": {
        "output": os.path.join("./", "output"),
        "qval": os.path.join("images", "qval"),
        "reward": os.path.join("images", "reward")
    },
    "experiment": {
        "default": {
            'description': 'Experiment QLearning with default settings',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        #  Qlearning
        "experiment01": {
            'description': 'Experiment QLearning with completely random agent',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment02": {
            'description': 'Experiment QLearning',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment03": {
            'description': 'Experiment QLearning with lower discount rate',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment04": {
            'description': 'Experiment QLearning with decreased learning rate',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment05": {
            'description': 'Experiment QLearning with very small learning rate',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.2,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment06": {
            'description': 'Experiment QLearning with large learning rate',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.9,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment07": {
            'description': 'Experiment QLearning with very small discount rate',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.1,
            't_max': 99
        },
        "experiment08": {
            'description': 'Experiment QLearning',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'Qlearning',
            'ddqn': None,
            'n': None,
            'α': 0.7,
            'λ': 0.001,
            'γ': 0.9,
            't_max': 99
        },
        # NStepQLearning #
        "experiment10": {
            'description': 'Experiment NStepQLearning',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '3',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment11": {
            'description': 'Experiment NStepQLearning with lower discount',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '3',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment12": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '7',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.5,
            't_max': 99
        },
        "experiment13": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '2',
            'α': 0.6,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment14": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '2',
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.6,
            't_max': 99
        },
        "experiment15": {
            'description': 'Experiment NStepQLearning with larger step-size',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '2',
            'α': 0.5,
            'λ': 0.0005,
            'γ': 0.4,
            't_max': 99
        },
        "experiment16": {
            'description': 'Experiment NStepQLearning',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'NStepQlearning',
            'ddqn': None,
            'n': '1',
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # Monte Carlo #
        "experiment20": {
            'description': 'Experiment MonteCarloLearning',
            'agent': 'TabularAgent',
            'environment': 'FrozenLakeEnvironment',
            'learning': 'MonteCarloLearning',
            'ddqn': None,
            'n': None,     #nstep
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # DeepQLearning #
        "experiment30": {
            'description': 'Experiment DeepQLearning',
            'agent': 'DQNAgent',
            'environment': 'CartPoleEnvironment',
            'learning': 'DeepQLearning',
            'ddqn': "False",
            'n': '20',       #batch_size
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        "experiment30b": {
            'description': 'Experiment DeepQLearning',
            'agent': 'DQNAgent',
            'environment': 'CartPoleEnvironment',
            'learning': 'DeepQLearning',
            'ddqn': "False",
            'n': '5',  # batch_size
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
        # Double DeepQLearning experiments #
        "experiment40": {
            'description': 'Experiment DoubleDeepQLearning',
            'agent': 'DQNAgent',
            'environment': 'CartPoleEnvironment',
            'learning': 'DeepQLearning',
            'ddqn': "True",
            'n': '50',  # batch_size
            'α': 0.7,
            'λ': 0.0005,
            'γ': 0.9,
            't_max': 99
        },
    },
    "models": {
        "model1": {
            "model_filename": "model1.h5",
            "lr": 0.001,
            "metrics": ["acc"]
        }
    },
    "environments": {
        "FrozenLake": "FrozenLakeEnvironment",
        "CartPole": "CartPoleEnvironment"
    },
    "methods": [
        "Qlearning",
        "NStepQLearning",
        "MonteCarloLearning",
        "DeepQLearning",
        "DoubleDeepQLearning"
    ]
}


def init_folders(experiment):
    path_experiment = os.path.join(
        params.get("dirs").get("output"),
        params.get("experiment").get(current_experiment).get("environment"),
        experiment)
    path_qval = params.get("dirs").get("qval")
    path_reward = params.get("dirs").get("reward")
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
        os.makedirs(os.path.join(path_experiment, path_qval))
        os.makedirs(os.path.join(path_experiment, path_reward))
    return