from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # The number of possible actions: sit, buy, sell
        self.memory = deque(maxlen=1000)  # Memory buffer to store experiences
        self.inventory = []  # Inventory to store bought stocks
        self.model_name = model_name  # Model to load (in evaluation mode)
        self.is_eval = is_eval  # Evaluation mode flag

        # Hyperparameters for Q-learning
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate (starting fully explorative)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay factor for epsilon after each action

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model(self):
        """
        Builds and compiles the neural network model for the agent using Keras Sequential API.
        The model consists of several fully connected (Dense) layers:
        - Input layer has `state_size` input nodes.
        - Hidden layers have 64, 32, and 8 units respectively, all with ReLU activations.
        - Output layer has `action_size` units (for the 3 possible actions) with linear activation.
        The model is compiled with mean squared error loss and Adam optimizer.
        """
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))  # Output layer
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

        return model

    def act(self, state):
        """
        Chooses an action (sit, buy, or sell) based on the current state.
        - If in training mode (not evaluation), the agent explores with probability `epsilon`.
        - Otherwise, it chooses the action with the highest predicted Q-value.
        """
        if not self.is_eval and np.random.rand() <= self.epsilon:
            # Exploration: Choose a random action (sit, buy, sell)
            return random.randrange(self.action_size)

        # Exploitation: Predict Q-values for the current state and choose the action
        # with the highest Q-value
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        """
        Trains the agent using experience replay. A batch of experiences is randomly sampled from
        memory. For each experience, the target Q-value is calculated based on the reward and the
        estimated future reward. The model is then trained on this target Q-value for the given
        action.
        """
        mini_batch = []
        mem_len = len(self.memory)  # Length of the current memory
        # Randomly sample a batch of experiences from memory
        for i in range(mem_len - batch_size + 1, mem_len):
            mini_batch.append(self.memory[i])

        # Iterate through the mini-batch and update the model
        for state, action, reward, next_state, done in mini_batch:
            target = reward  # Initial target is the immediate reward
            if not done:
                # If the episode is not done, update the target with the discounted future
                # reward (Q-learning update)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Get the predicted Q-values for the current state
            target_f = self.model.predict(state)
            # Update the Q-value for the chosen action to the calculated target value
            target_f[0][action] = target
            # Train the model on this experience (state, target Q-values) for one epoch
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decrease exploration rate `epsilon`, but ensure it doesn't go below `epsilon_min`
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
