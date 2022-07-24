from tensorflow.keras.models import load_model
import numpy as np
from environment import MultiStockEnv
import argparse

    #Test the performance of the agent with the test data
def test_trader(test_data,
                initial_investment,
                num_episodes,
                action_size,
                model_path = r'DQN_Tradebot_Project/DQN_Trained_Model',
                epsilon = 0.01
                ):
    env = MultiStockEnv(test_data, initial_investment)
    scaler = MultiStockEnv.get_scaler(env)
    testportfoliovalues=[]
    model = load_model(model_path)
    for i in range(num_episodes):
        state = env.reset()
        state = scaler.transform([state])
        done=False
        while not done:
            action = MultiStockEnv.greedy_policy(state,model,epsilon = epsilon, action_size = action_size)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            state = next_state
        value=info['cur_val']
        print(f"episode: {i + 1}/{num_episodes}, episode end value: {value:.2f}")
        testportfoliovalues.append(value) # append episode end portfolio value
    return testportfoliovalues