import sys
from keras.models import load_model
from agent.agent import Agent
from functions import formatPrice, getStockDataVec, getState

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

# Extract stock name and model name from command-line arguments
stock_name, model_name = sys.argv[1], sys.argv[2]
# Load the specified Keras model from the models directory
model = load_model("models/" + model_name)
# Get the input window size from the model's first layer
window_size = model.layers[0].input.shape.as_list()[1]

# Create an instance of Agent with the specified parameters
agent = Agent(window_size, True, model_name)
# Get the stock data (price history) for the given stock name
data = getStockDataVec(stock_name)
num_candles = len(data) - 1  # Length of the stock data (number of candles/trading days)

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(num_candles):
    action = agent.act(state)  # Determine the action based on the current state

    # If nothing happens, defaults to sit
    next_state = getState(data, t + 1, window_size + 1)  # Get the next state
    reward = 0  # Initialize the reward for the current action

    # If the agent decides to buy (action == 1)
    if action == 1:
        agent.inventory.append(data[t])  # Add the current price to the inventory
        print("Buy: " + formatPrice(data[t]))

    # If the agent decides to sell (action == 2) and has something in the inventory
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)  # Get the bought price from the inventory
        reward = max(data[t] - bought_price, 0)  # Calculate reward based on profit
        total_profit += data[t] - bought_price  # Update total profit

        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    # Check if this is the last candle/day
    done = True if t == num_candles - 1 else False

    # Store the experience (state, action, reward, next_state, done) in memory
    agent.memory.append((state, action, reward, next_state, done))

    # Update the state to the next state
    state = next_state

    # If this is the last day, print the total profit
    if done:
        print("--------------------------------")
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")
