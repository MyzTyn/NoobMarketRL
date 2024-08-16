import gymnasium as gym
from gymnasium import spaces
import numpy as np

import matplotlib.pyplot as plt

# Actions
OUT = 0
BUY = 1
SELL = 2

# Order position
SHORT = -1
NONE = 0
LONG = 1

class StaticMarketEnv(gym.Env):
    maxSteps = 500
    steps = 0
    
    isUp = True
    done = False
    
    # Market Price
    current_price = 0
    
    # Order placement: -1 = Short, 0 = None, 1 = Long
    current_position = 0
    
    # Entry Price
    entry_price = np.nan

    # Track cumulative profit
    cumulative_profit = 0.0

    # Previous prices
    history_window_size = 10
    previous_prices = []

    totalrewards = 0

    def __init__(self, config=None):
        super(StaticMarketEnv, self).__init__()
        self.render_mode = 'human'

        # Observation space: Flattened array of price and position
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # Action space: Hold (0), Buy (1), Sell(2)
        self.action_space = spaces.Discrete(3)
        
        # To store price history for visualization
        self.price_history = []
        self.action_history = []
        
        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Price')
        self.ax.set_title('Simple Market Visualization')
        self.ax.grid(True)

        # Initialize plot elements
        self.price_line, = self.ax.plot([], [], 'b-', lw=2, label='Price')
        self.buy_scatter = self.ax.scatter([], [], c='g', marker='^', s=100, label='Long')
        self.sell_scatter = self.ax.scatter([], [], c='r', marker='v', s=100, label='Short')
        self.ax.legend()
        
    def reset(self, seed=None, options=None):
        self.current_price = 0
        self.current_position = NONE
        self.done = False
        self.steps = 0
        self.entry_price = np.nan
        
        # Reset history
        self.price_history = []
        self.action_history = []
        self.cumulative_profit = 0
        self.reward = 0
        self.previous_prices = []

        return np.array([self.current_price, self.current_position], dtype=np.float32)
    
    def step(self, action):
        reward = 0
        
        # Simulate price movement (up or down)
        if self.isUp:
            self.current_price += 0.1
        else:
            self.current_price -= 0.1
            
        self.current_price = np.clip(self.current_price, 0, 1)
        if self.current_price == 0 or self.current_price == 1:
            self.isUp = not self.isUp
        
        # Transaction cost (e.g., 0.1% of the current price)
        transaction_cost = 0.001 * self.current_price

        # Take Action:
        if action == OUT: # Nothing
            if self.current_position == NONE:
                reward = -0.002 # Small penalty 
            else:
                reward = -0.005 # Small rewards for holding
        
        elif action == BUY and self.current_position == NONE:  # Buy (Long)
            self.entry_price = self.current_price # Open the position
            self.current_position = LONG
            # Provide a reward that is inversely proportional to the current price
            reward = -transaction_cost
            self.action_history.append((self.steps, self.current_price, 'Long'))
            

        elif action == SELL and self.current_position == LONG:  # Sell (Short)
            self.current_position = NONE # Close the position
            reward = self.current_price - self.entry_price - transaction_cost
            
            self.cumulative_profit += reward
            self.entry_price = np.nan
            self.action_history.append((self.steps, self.current_price, 'Short'))
            
        # Update history
        self.price_history.append((self.steps, self.current_price))

        if len(self.previous_prices) >= self.history_window_size:
            self.previous_prices.pop(0)
        self.previous_prices.append(self.current_price)

        # Update step count and check if done
        self.steps += 1
        if self.steps >= self.maxSteps:
            self.done = True
        
        self.totalrewards += reward

        return np.array([self.current_price, self.current_position], dtype=np.float32), reward, self.done, False, {}
    
    def render(self):
        if self.render_mode == 'human':
            ## Clear the previous plot
            self.ax.clear()
            self.ax.set_xlabel('Time Steps')
            self.ax.set_ylabel('Price')
            self.ax.set_title('Simple Market Visualization')
            self.ax.grid(True)
            
            # Define the window size for display
            window_size = 50
            steps, prices = zip(*self.price_history) if self.price_history else ([], [])
            if len(steps) > window_size:
                steps = steps[-window_size:]
                prices = prices[-window_size:]
            
            self.price_line, = self.ax.plot(steps, prices, 'b-', lw=2, label='Price')
            
            # Update scatter data for buy/sell actions
            buy_steps, buy_prices = zip(*[(s, p) for s, p, a in self.action_history if a == 'Long']) if any(a == 'Long' for _, _, a in self.action_history) else ([], [])
            sell_steps, sell_prices = zip(*[(s, p) for s, p, a in self.action_history if a == 'Short']) if any(a == 'Short' for _, _, a in self.action_history) else ([], [])
            
            if len(buy_steps) > window_size:
                buy_steps = buy_steps[-window_size:]
                buy_prices = buy_prices[-window_size:]
            if len(sell_steps) > window_size:
                sell_steps = sell_steps[-window_size:]
                sell_prices = sell_prices[-window_size:]
            
            self.buy_scatter = self.ax.scatter(buy_steps, buy_prices, c='g', marker='^', s=100, label='Long')
            self.sell_scatter = self.ax.scatter(sell_steps, sell_prices, c='r', marker='v', s=100, label='Short')
            
            # place a text box in upper left in axes coords
            self.ax.text(0.05, 0.95, "Profit: " + str(self.cumulative_profit) + " Reward:" + str(self.totalrewards), transform=self.ax.transAxes, fontsize=14,
        verticalalignment='top')

            # Redraw the plot
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.legend()
            plt.draw()
            plt.pause(0.01)  # Short pause to update the plot
            
    def close(self):
        plt.close(self.fig)