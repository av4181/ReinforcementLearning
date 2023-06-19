from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Dropout, Dense, Activation
from keras import callbacks

def create_model(selected_model, state_size, action_size):
    # Opbouw ANN
    model = models.Sequential()
    # Input layer 1
    model.add(layers.Dense(24, activation='relu', input_dim=state_size))
    model.add(Dropout(0.5))
    # Input layer 2
    model.add(layers.Dense(24, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(layers.Dense(action_size, activation='linear'))

    model.summary()

    model.compile(loss='mse',optimizer='adam',metrics="acc")

    return model