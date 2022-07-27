# Trade Bot with Deep Q Network
 - This Repository is Trade Bot using Deep Reinforcement Learning Method Deep Q Learning
 
![Tradebot wITH DEEP Q LearnING](https://user-images.githubusercontent.com/67932543/180650176-7628e074-9e30-4b6f-a1d0-0b46c5a0d2c3.png)

# Introduction

Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment, and it can handle problems with stochastic transitions and rewards without requiring adaptations. But in this project, deep learning added to learning process. So that, it becomes Deep Q-learning with model. A core difference between Deep Q-Learning and Q-Learning is the implementation of the Q-table. Critically, Deep Q-Learning replaces the regular Q-table with a neural network. Rather than mapping a state-action pair to a q-value, a neural network maps input states to (action, Q- value) pairs.

![image](https://user-images.githubusercontent.com/67932543/180648947-bf93039e-ceb6-439e-ade6-8c55cce86709.png)

# Quick Start
### Alternative 1
```bash
# For quick action (Conda Environment)
conda create -n tradebot -y && conda activate tradebot && pip install -r requirements.txt && bash Test/test.sh

# For quick action (Pip Environment)
python -m venv tradebot && source tradebot/bin/activate && pip install -r requirements.txt && bash Test/test.sh

## Just give me the linesssss without virtual enviroment
pip install -r requirements.txt && bash Test/test.sh
```
Note: I'm working on better bash code but for now, just tester bash untill next update
### Alternative 2
```bash
# Check before installing. It contains Tensorflow MacOS in requirements.txt
# Python 3.9 or higher suggested
pip install -r requirements.txt
```
```bash
python trader_file.py --stocks AAPL MSI SBUX \
                      --start_date '2021-01-01' \
                      --end_date '2022-07-23'
                      --initial_investment 20000 \
                      --gamma 0.95 \
                      --epsilon 1.0 \
                      --epsilon_min 0.01 \
                      --epsilon_decay 0.995 \
                      --num_episodes 50 \
                      --save_model True
```

# Body
For the beginning, automated stock obtainer function has been written. By using that function, any stock type as daily, can be obtained easily

```python
df = stock_data(stock_name_list = ['AAPL','MSI','SBUX','GME','GOOGL'],
                start_date = '2021-01-01', 
                end_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'))
```

After than, environment has been created. Initial investment set to 20000USD and all environment parameters embedded to trader function. For Deep Network, Keras selected as framework. For flexible use, deep learning method also embedded to environment class.

```python
# Current Model of the Environment(environment.py)
def model_builder(self):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 32, activation='relu', input_dim = self.state_dim))
    model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
    model.add(tf.keras.layers.Dense(units = 128, activation='relu'))
    model.add(tf.keras.layers.Dense(units = len(self.action_list), activation = 'linear'))
    model.compile(loss='mse', optimizer = tf.keras.optimizers.Adam(lr = 0.001))
    return model
```

For this trade-bot, Apple, MSI, Starbucks, GameStop and Google stocks had been selected. Beginning date of the environment selected 2021-01-01 and for the ending 2022-21-07 selected(One day before training). Number of episodes were 50 and rest of the parameters shown as below:
```python
training_output = trader(stocks = ['AAPL','MSI','SBUX','GME','GOOGL'],
                         start_date = '2021-01-01',
                         end_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'),
                         initial_investment = 20000,
                         gamma = 0.95,  # discount rate
                         epsilon = 1.0,  # exploration rate
                         epsilon_min = 0.01,
                         epsilon_decay = 0.995,
                         num_episodes = number_of_episodes,
                         save_model = True)
```

# Conclusion
This work uses a Reinforcement Learning technique called Deep Q-Learning. At any given episode, an agent observes its current state (day window stock price representation), selects and performs an action (buy/sell/hold), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts its parameters based on the gradient of the loss computed. By using this project, day traders or option traders can profit easily. For doing that, model need to be tuned very carefully because of the stock environment is not that stable for develop models and achieve success. Apart from that, maybe success rate not far better from if the model tuned successfully but this technique way safer than trade randomly with very few knowledge of economics who trade randomly in the market(probably represents %97 of the traders in the stock market environment).

# Referances
[1] Mehmet Yasin Ulukus (https://scholar.google.com/citations?user=QHccq8wAAAAJ&hl=en)
