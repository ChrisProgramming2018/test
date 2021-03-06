from replay_buffer import ReplayBuffer
from agent_iql import Agent
import sys

import time


def time_format(sec):
    """
    
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train(env, config):
    """

    """
    t0 = time.time()
    memory = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory.load_memory(config["buffer_path"])
    agent = Agent(state_size=8, action_size=4,  config=config) 
    memory_t = ReplayBuffer((8,), (1,), config["expert_buffer_size"], config["device"])
    memory_t.load_memory(config["expert_buffer_path"])
    memory.idx = config["idx"] 
    memory_t.idx = config["idx"] * 4
    print("memory idx ",memory.idx)  
    print("memory_expert idx ",memory_t.idx)
    for idx in range(memory.idx):
        print(memory.actions[idx], memory.rewards[idx])
    for t in range(config["predicter_time_steps"]):
        text = "Train Predicter {}  \ {}  time {}  \r".format(t, config["predicter_time_steps"], time_format(time.time() - t0))
        print(text, end = '')
        agent.learn(memory, memory_t)
        #if t % 1 == 0:
        #print(text)
        if t % 100 == 0:
            agent.save("models/{}-".format(t))
            agent.test_predicter(memory)
            agent.test_q_value(memory)
