import sys
from agent.agent import Agent
from functions import formatPrice, getState, getStockDataVec


# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

# Extract stock name, window size, and episode count from command-line arguments
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

# Initialize the agent with the given window size (number of days in the state)
agent = Agent(window_size)

# Get the stock data (price history) for the given stock name
data = getStockDataVec(stock_name)
num_candles = len(data) - 1  # Length of the stock data (number of candles/trading days)
batch_size = 32  # Size of the mini-batch for experience replay

# Main loop: Iterate over each episode to train the agent
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))

    # Initialize the state for the first day (window_size + 1 days of data)
    state = getState(data, 0, window_size + 1)

    total_profit = 0  # Initialize total profit for this episode
    agent.inventory = []  # Clear the agent's inventory at the beginning of each episode

    # Loop through each trading day within the episode
    for t in range(num_candles):
        # The agent takes an action (0: sit, 1: buy, 2: sell) based on the current state
        action = agent.act(state)

        # Get the next state (stock data for the next day)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0  # Initialize the reward for this step

        # If the agent decides to buy (action == 1)
        if action == 1:
            agent.inventory.append(data[t])  # Add the current price to the inventory
            print("Buy: " + formatPrice(data[t]))

        # If the agent decides to sell (action == 2) and has something in the inventory
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)  # Retrieve the bought price
            reward = max(data[t] - bought_price, 0)  # Calculate reward (profit from sale)
            total_profit += data[t] - bought_price  # Update total profit
            print(
                "Sell: "
                + formatPrice(data[t])
                + " | Profit: "
                + formatPrice(data[t] - bought_price)
            )

        # Check if this is the last candle/day (end of the episode)
        done = True if t == num_candles - 1 else False

        # Store the experience (state, action, reward, next_state, done) in memory
        agent.memory.append((state, action, reward, next_state, done))

        # Update the state to the next state
        state = next_state

        # If this is the last day, print the total profit
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        # If the agent's memory is larger than the batch size,
        # train the agent with experience replay
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    # Every 10 episodes, save the current model to disk
    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))
