#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:19:47 2023

@author: dst
"""

import numpy as np
from environment import Cancer
from agent import DQNAgent
import pickle


def getStateSpace(fb,rb,size_pop):
    stateSpace = []
    xho = np.linspace(fb,rb,size_pop)
    xgs = np.linspace(0,1,size_pop)
    for iho in range(size_pop):
        for igs in range(size_pop):
            xHO, xGS = xho[iho], xgs[igs]
            if 0 < xHO + xGS < 1:
                xRL = 1-(xHO+xGS)
                if 0 < xRL < 1:
                    state = np.array([xHO, xGS, xRL])
                    stateSpace.append(state)
    return stateSpace


def train_agent(seed):
    file = open("data_input",'rb')
    data = pickle.load(file)
    
    sigma = data["sigma"]
    train_each = data["train_each"]
    al = data["al"]
    bl = data["bl"]
    delta_t = data["delta_t"]
    max_step = data["max_steps"]
    penalty = data["penalty"]
    state_size = data["state_size"]
    action_size = data["action_size"]
    EPISODES = data["EPISODES"]
    
    env = Cancer(al, bl, delta_t, max_step, sigma, penalty)
    agent = DQNAgent(state_size, action_size, max_step, EPISODES) 
       
    train_iter = 0
    
    scores, eps_hist, steps = [], [], []
    deads, recos, unrecos = [], [], []
    actions_0,actions_1,actions_2,actions_3 = [], [], [], []
    actions_4,actions_5,actions_6,actions_7,actions_8 = [], [], [], [], [] 
    
    for e in range(EPISODES):
        done = False
        score = 0.0
        action_0,action_1,action_2,action_3 = 0, 0, 0, 0
        action_4,action_5,action_6,action_7,action_8 = 0, 0, 0, 0, 0
        
        state = env.reset()
                
        while not done:
            action = agent.get_action(np.reshape(state, [1, state_size]))
            state_, reward, done, info = env.step(state, action) 
            
            agent.append_sample(np.reshape(state, [1, state_size]), action, reward, np.reshape(state_, [1, state_size]), done)
            score += reward
            
            train_iter += 1
            if not train_iter % train_each: agent.train_model()
        
            state = state_
            
            if action == 0: action_0 += 1
            elif action == 1: action_1 += 1
            elif action == 2: action_2 += 1
            elif action == 3: action_3 += 1
            elif action == 4: action_4 += 1
            elif action == 5: action_5 += 1
            elif action == 6: action_6 += 1
            elif action == 7: action_7 += 1
            else: action_8 += 1
            
        agent.update_target_model()
        agent.update_epsilon() 
                
        if state[0] > env.rb: dead, reco, unreco = 0, 1, 0
        elif state[0] < env.fb: dead, reco, unreco = 1, 0, 0
        else: dead, reco, unreco = 0, 0, 1
                
        scores.append(score)
        eps_hist.append(agent.epsilon)
        steps.append(env.step_cntr)
        deads.append(dead)
        recos.append(reco)
        unrecos.append(unreco)
        actions_0.append(action_0)
        actions_1.append(action_1)
        actions_2.append(action_2)
        actions_3.append(action_3)
        actions_4.append(action_4)
        actions_5.append(action_5)
        actions_6.append(action_6)
        actions_7.append(action_7)
        actions_8.append(action_8)
        
        if not(e % 1000) or e==EPISODES-1:
            # Save results  
            agent.save("DDQN_weights_" + str(seed) + "_" + str(e)) # Save NN weights  
            
            results ={'scores': scores,
                      'eps_hist': eps_hist,
                      'steps': steps,
                      'deads': deads,
                      'recos': recos,
                      'unrecos': unrecos,
                      'actions_0': actions_0,
                      'actions_1': actions_1,
                      'actions_2': actions_2,
                      'actions_3': actions_3,
                      'actions_4': actions_4,
                      'actions_5': actions_5,
                      'actions_6': actions_6,
                      'actions_7': actions_7,
                      'actions_8': actions_8}
            pickle_out = open(f"DDQN_training_{seed}_{e}", "wb")
            pickle.dump(results, pickle_out, pickle.HIGHEST_PROTOCOL)
            pickle_out.close()
        
        
        

def validate_agent(seed):
    file = open("data_input",'rb')
    data = pickle.load(file)
    
    sigma = data["sigma"]
    train_each = data["train_each"]
    al = data["al"]
    bl = data["bl"]
    delta_t = data["delta_t"]
    max_step = data["max_steps"]
    penalty = data["penalty"]
    state_size = data["state_size"]
    action_size = data["action_size"]
    EPISODES = data["EPISODES"]
    
    env = Cancer(al, bl, delta_t, max_step, sigma, penalty)
    agent = DQNAgent(state_size, action_size, max_step, EPISODES) 
    agent.load(f"DDQN_weights_{seed}_{EPISODES-1}")
    save_epsilon = agent.epsilon
    agent.epsilon = 0 
    
    size_pop = 32
    size_pop = 2 # CAMBIA ESTOS..................................!!!!!!!!!!!!
    states = getStateSpace(env.fb,env.rb,size_pop)
    
    scores = []
    deads, recos, unrecos = [], [], []
    steps = []
    
        
    for state in states:
        done = False
        env.step_cntr = 0
        
        score = 0.0
                
        while not done:
            action = agent.get_action(np.reshape(state, [1, state_size]))
            state_, reward, done, info = env.step(state, action)  
            state = state_
            score += reward

        if state[0] > env.rb:
            dead, reco, unreco = 0, 1, 0
        elif state[0] < env.fb:
            dead, reco, unreco = 1, 0, 0
        else:
            dead, reco, unreco = 0, 0, 1
        deads.append(dead)
        recos.append(reco)
        unrecos.append(unreco)
        
        scores.append(score)
        steps.append(env.step_cntr)
        
        
    results = {"final_cost": np.sum(scores),
               "scores": scores,
               "deads": deads,
               "recos": recos,
               "unrecos": unrecos,
               "steps": steps,
               }
    
    pickle_out = open("DDQN_validate_" + str(seed), "wb")
    pickle.dump(results, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close() 
    
    agent.epsilon = save_epsilon
    
    
    
    