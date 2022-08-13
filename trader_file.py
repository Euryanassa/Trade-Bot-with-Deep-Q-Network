from environment import MultiStockEnv
from datetime import datetime, timedelta
from keras.models import save_model
from stock_obtain import *
import numpy as np 
import pandas as pd
import argparse

def trader(stocks = ['AAPL','MSI','SBUX','GME','GOOGL'],
           start_date = '2021-01-01',
           end_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'),
           initial_investment = 20000,
           gamma = 0.95,  # discount rate
           epsilon = 1.0,  # exploration rate
           epsilon_min = 0.01,
           epsilon_decay = 0.995,
           num_episodes = 50,
           save_model = True):
    data = stock_data(
                    stocks,
                    start_date = start_date,
                    end_date = end_date
                    )

    n_timesteps, n_stocks = data.shape
    n_train = n_timesteps // 2

    train_data = data[:n_train].values
    test_data = data[n_train:].values

    env = MultiStockEnv(train_data,
                        initial_investment = initial_investment)

    env.reset()

    #Train the scaler, the scaler only uses the training set
    state_size = env.state_dim
    action_size = len(env.action_space)

    model = env.model_builder()
    # store the final value of the portfolio (end of episode)

    scaler = MultiStockEnv.get_scaler(env)

    portfolio_value = []

    for i in range(int(num_episodes)):
        done=False
        state = env.reset()
        state = scaler.transform([state])
        while not done:
            action = MultiStockEnv.greedy_policy(state, model, epsilon, action_size)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            reward = float(reward)
            gamma = float(gamma)
            if done:
                target = reward
            else:
                target = reward + gamma * np.amax(model.predict(next_state)[0])
           
            target_full = model.predict(state)
            target_full[0, action] = target

            model.fit(state, target_full, epochs=1, verbose=0)
            
            state = next_state
            
            epsilon_decay = float(epsilon_decay)
            epsilon = float(epsilon)
            epsilon_min = float(epsilon_min)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
        val=info['cur_val']
        print(f"episode: {int(i) + 1}/{int(num_episodes)}, episode end value: {float(val):.2f}, total profit: {-(100 - 100*(float(val)/float(initial_investment))):.2f}%")
        portfolio_value.append(val)

    if save_model:
        model.save('DQN_Trained_Model')

    return portfolio_value

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', nargs='+', default = ['AAPL','MSI','SBUX'], help = 'Stocks shortcut (You can find from google), default is ["AAPL","MSI","SBUX"]')
    parser.add_argument('--start_date', default = '2021-01-01', help = 'Write date in YYYY-DD-MM format, default is 2021-01-01')
    parser.add_argument('--end_date', default = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'), help = 'Write date in YYYY-DD-MM format, default is yesterday')
    parser.add_argument('--initial_investment', default = 20000, help = 'Money amout for your environment, default is 20000(20kUSD)')
    parser.add_argument('--gamma', default = 0.95 , help = 'Discount Rate, default is 0.95')
    parser.add_argument('--epsilon', default = 1.0, help = 'Max limit of exploration rate, default 1.0 (range(0.0 - 1.0))')
    parser.add_argument('--epsilon_min', default = 0.01, help = 'Min limit of exploration rate, default 0.1 (range(0.0 - 1.0))')
    parser.add_argument('--epsilon_decay', default = 0.995, help = 'Decay rate of exploration rate, default is 0.0995')
    parser.add_argument('--num_episodes', default = 50, help = 'Number of episodes for learning, default is 50')
    parser.add_argument('--save_model', default = True, help = 'Save DQN model, default is True')
    opt = parser.parse_args()
    print(opt)
    return opt

def main(opt):
    trader(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    '''
    trader(stocks = ['AAPL','MSI','SBUX','GME','GOOGL'],
           start_date = '2021-01-01',
           end_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'),
           initial_investment = 20000,
           gamma = 0.95,  # discount rate
           epsilon = 1.0,  # exploration rate
           epsilon_min = 0.01,
           epsilon_decay = 0.995,
           num_episodes = 5,
           save_model = True)
    '''
