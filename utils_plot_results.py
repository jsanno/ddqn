#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:36:56 2023

@author: len0x
"""

import pickle
import matplotlib.pyplot as plt
import ternary
from environment import Cancer
from agent import DQNAgent
from utils_train_ddqn import getStateSpace
import numpy as np


def view_DDQN_policy(policy,weights):
    
    size_pop = 32
    n = 20
    
    pickle_in = open('data_input', 'rb')
    data = pickle.load(pickle_in) 
    
    sigma = data["sigma"]    
    al = data["al"]
    bl = data["bl"]
    delta_t = data["delta_t"]
    max_step = data["max_steps"]
    penalty = data["penalty"]
    
    state_size = data['state_size']
    action_size = data['action_size']
      
    
    EPISODES = data['EPISODES']
    
    
    
    pickle_in = open(policy, "rb")
    data = pickle.load(pickle_in)   
    actions0 = data['actions_0']
    actions1 = data['actions_1']
    actions2 = data['actions_2']
    actions3 = data['actions_3']
    actions4 = data['actions_4']
    actions5 = data['actions_5']
    actions6 = data['actions_6']
    actions7 = data['actions_7']
    actions8 = data['actions_8']
    
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    if actions0 != []: tax.scatter(actions0, marker="s", color='tab:blue', label='1') 
    if actions1 != []: tax.scatter(actions1, marker="s", color='tab:orange', label='2')
    if actions2 != []: tax.scatter(actions2, marker="s", color='tab:green', label='3')
    if actions3 != []: tax.scatter(actions3, marker="s", color='tab:red', label='4')
    if actions4 != []: tax.scatter(actions4, marker="s", color='tab:purple', label='5')
    if actions5 != []: tax.scatter(actions5, marker="s", color='tab:brown', label='6')
    if actions6 != []: tax.scatter(actions6, marker="s", color='tab:pink', label='7')
    if actions7 != []: tax.scatter(actions7, marker="s", color='tab:gray', label='8')
    if actions8 != []: tax.scatter(actions8, marker="s", color='tab:olive', label='9')

    
    tax.legend(title="Actions")
    
    env = Cancer(al, bl, delta_t, max_step, sigma, penalty)
    agent = DQNAgent(state_size, action_size, max_step, EPISODES) 
    agent.load(weights)
    save_epsilon = agent.epsilon
    agent.epsilon = 0 
    
    initial_states = getStateSpace(env.fb,env.rb,size_pop)
    
    i = 0
    while i < len(initial_states):
        state = initial_states[i]
        done = False
        trajectory = []
        trajectory.append(state)
        
        while not done:
            action = agent.get_action(np.reshape(state, [1, state_size]))
            state_, reward, done, info = env.step(state, action)
            state = state_
            trajectory.append(state)
        
        tax.plot(trajectory, color='black')  
        i += n
    
    agent.epsilon = save_epsilon
    
    
    plt.annotate('$x_{R^L}$',xy=(-0.023,-0.053))
    plt.annotate('$x_{H^O}$',xy=(0.98,-0.053))
    plt.annotate('$x_{G^S}$',xy=(0.529,0.853))
    
    tax.show()
    
    policy_cost = data['average_score']*delta_t
    

    print('*------------------------------------------------*')
    print(f'DDQN cost: {policy_cost}')
    print('------------------------------------------------/n')      


    
    
    
def ddqn_vs_conventional(init_state,weights):
    pickle_in = open('data_input', 'rb')
    data = pickle.load(pickle_in) 
    
    sigma = data["sigma"]    
    al = data["al"]
    bl = data["bl"]
    delta_t = data["delta_t"]
    max_step = data["max_steps"]
    penalty = data["penalty"]
    state_size = data["state_size"]
    action_size = data["action_size"]
    EPISODES = data["EPISODES"]
    
    env = Cancer(al, bl, delta_t, max_step, sigma, penalty, real_cost=True)
    agent = DQNAgent(state_size, action_size, max_step, EPISODES) 
    
    agent.load(weights)
    save_epsilon = agent.epsilon
    agent.epsilon = 0 
    
    actions = []
    trayectory = []
    trayectory.append(init_state)
    done = False
    score = 0.0
    state = init_state
    while not done:
        action = agent.get_action(np.reshape(state, [1, state_size]))
        state_, reward, done, info = env.step(state, action) 
        score += reward
        state = state_
        trayectory.append(state)
        actions.append(action)
    
    agent.epsilon = save_epsilon
    
    num0, num1, num2, num3, num4, num5, num6, num7, num8 = 0,0,0,0,0,0,0,0,0    
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    
    for i in range(len(actions)):
        action = actions[i]
        new_traj = trayectory[i:i+2]
        if action == 0: 
            col = 'tab:blue'
            num0 += 1
        elif action == 1: 
            col = 'tab:orange'
            num1 += 1
        elif action == 2:
            col = 'tab:green'
            num2 += 1
        elif action == 3:
            col = 'tab:red'
            num3 +=1
        elif action == 4:
            col = 'tab:purple'
            num4 += 1
        elif action == 5:
            col = 'tab:brown'
            num5 += 1
        elif action == 6:
            col = 'tab:pink'
            num6 += 1
        elif action == 7:
            col = 'tab:gray'
            num7 += 1
        else: 
            col = 'tab:olive'
            num8 += 1
        tax.plot(new_traj, linewidth=2.0 , color=col)
        tax.scatter(new_traj, s=20 ,color=col)
    tax.show()
            
    
    print('*----------------------------------------------*')
    print('                   DDQN Report                  ')
    print('------------------------------------------------')
    print(f'trajectory cost: {score}')
    print(f'number of steps: {env.step_cntr}')
    print('------------------------------------------------/n')       
    
    
    #--------------------------------------------------------------------------
    # CONVENTIONAL TRAJECTORY
    #--------------------------------------------------------------------------
    env.reset()
    actions = []
    trayectory = []
    trayectory.append(init_state)
    done = False
    score = 0.0
    state = init_state
    while not done:
        xGS, xRL = state[1], state[2]
        if xGS > xRL: action = 7#7
        else: action = 5
        state_, reward, done, info = env.step(state, action) 
        score += reward
        state = state_
        trayectory.append(state)
        actions.append(action)
    
    num0, num1, num2, num3, num4, num5, num6, num7, num8 = 0,0,0,0,0,0,0,0,0    
    for i in range(len(actions)):
        action = actions[i]
        new_traj = trayectory[i:i+2]
        if action == 0: 
            col = 'tab:blue'
            num0 += 1
        elif action == 1: 
            col = 'tab:orange'
            num1 += 1
        elif action == 2:
            col = 'tab:green'
            num2 += 1
        elif action == 3:
            col = 'tab:red'
            num3 +=1
        elif action == 4:
            col = 'tab:purple'
            num4 += 1
        elif action == 5:
            col = 'tab:brown'
            num5 += 1
        elif action == 6:
            col = 'tab:pink'
            num6 += 1
        elif action == 7:
            col = 'tab:gray'
            num7 += 1
        else: 
            col = 'tab:olive'
            num8 += 1
        tax.plot(new_traj, linewidth=2.0 , color=col, linestyle="--")
        tax.scatter(new_traj, s=20 ,color=col)

            
    
    print('*----------------------------------------------*')
    print('        Conventional Trajectory Report          ')
    print('------------------------------------------------')
    print(f'trajectory cost: {score}')
    print(f'number of steps: {env.step_cntr}')
    print('------------------------------------------------/n')    
    
    
    
    #--------------------------------------------------------------------------
    # NO THERAPY TRAJECTORY
    #--------------------------------------------------------------------------
    env.reset()
    actions = []
    trayectory = []
    trayectory.append(init_state)
    done = False
    score = 0.0
    state = init_state
    while not done:
        xGS, xRL = state[1], state[2]
        action = 0
        state_, reward, done, info = env.step(state, action) 
        score += reward
        state = state_
        trayectory.append(state)
        actions.append(action)
    
    
    print('*----------------------------------------------*')
    print('               No therapy                       ')
    print('------------------------------------------------')
    print(f'trajectory cost: {score}')
    print(f'number of steps: {env.step_cntr}')
    
    tax.scatter(trayectory, s=20 ,color='tab:blue')
    tax.plot(trayectory, linewidth=2.0 , color='tab:blue', linestyle="--")
    tax.ticks(axis='lbr', multiple=0.1, linewidth=1, tick_formats="%.1f", offset=0.02)
    tax.gridlines(multiple=0.1, color="black")
    tax.get_axes().axis('off')
    tax.show()



def view_HJB_policy(dir_std):
    num_jobs = 10
    num_traj = 60

    rb, fb = 0.9, 0.1
    states, actions = [], []
    gab = 1e-5
    qg_max, qg_min = 102.06, 51.03
    ql_max, ql_min = 0.0054, 0.0027
    
    for job in range(num_jobs+1):
        for traj in range(num_traj):
            try:
                dir_states = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.states'
                dir_doses = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.controls' 
                
                count = 0
                f = open(dir_states,'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    xho, xgs, xrl = float(fields[0]), float(fields[1]), float(fields[2])
                    if xho < fb or xho > rb: break
                    else:
                        state = np.array([xho, xgs, xrl])
                        states.append(state)
                        count += 1 

                step = 0
                f = open(dir_doses, 'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    qg, ql = float(fields[0]), float(fields[1])
                    if 0.0-gab <= qg <= 0.0+gab and 0.0-gab <= ql <= 0.0+gab: action = 0
                    elif 0.0-gab <= qg <= 0.0+gab and ql_min-gab <= ql <= ql_min+gab: action = 1
                    elif 0.0-gab <= qg <= 0.0+gab and ql_max-gab <= ql <= ql_max+gab: action = 2
                    elif qg_min-gab <= qg <= qg_min+gab and 0.0-gab <= ql <= 0.0+gab: action = 3
                    elif qg_min-gab <= qg <= qg_min+gab and ql_min-gab <= ql <= ql_min+gab: action = 4
                    elif qg_min-gab <= qg <= qg_min+gab and ql_max-gab <= ql <= ql_max+gab: action = 5
                    elif qg_max-gab <= qg <= qg_max+gab and 0.0-gab <= ql <= 0.0+gab: action = 6
                    elif qg_max-gab <= qg <= qg_max+gab and ql_min-gab <= ql <= ql_min+gab: action = 7
                    elif qg_max-gab <= qg <= qg_max+gab and ql_max-gab <= ql <= ql_max+gab: action = 8
                    else: print(f'Error job {job} traj {traj}')
                    actions.append(action)
                    step += 1
                    if step == count: break
                    else: pass
                               
            except:
                print(f'Trajectory {job}-{traj} not found ...')
        
    
    states_0, states_1, states_2, states_3, states_4, states_5 = [], [], [], [], [], []
    states_6, states_7, states_8 = [], [], []
    for index in range(len(states)):
        state = states[index]
        action = actions[index]
        if action == 0: states_0.append(state)
        elif action == 1: states_1.append(state)
        elif action == 2: states_2.append(state)
        elif action == 3: states_3.append(state)
        elif action == 4: states_4.append(state)
        elif action == 5: states_5.append(state)
        elif action == 6: states_6.append(state)
        elif action == 7: states_7.append(state)
        elif action == 8: states_8.append(state) 
        
        
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    if states_0 != []: tax.scatter(states_0, marker="s", color='tab:blue', label='1') 
    if states_1 != []: tax.scatter(states_1, marker="s", color='tab:orange', label='2')
    if states_2 != []: tax.scatter(states_2, marker="s", color='tab:green', label='3')
    if states_0 != []: tax.scatter(states_0, marker=".", color='tab:blue') 
    if states_3 != []: tax.scatter(states_3, marker="s", color='tab:red', label='4')
    if states_4 != []: tax.scatter(states_4, marker="s", color='tab:purple', label='5')
    if states_5 != []: tax.scatter(states_5, marker="s", color='tab:brown', label='6')
    if states_6 != []: tax.scatter(states_6, marker="s", color='tab:pink', label='7')
    if states_7 != []: tax.scatter(states_7, marker="s", color='tab:gray', label='8')
    if states_8 != []: tax.scatter(states_8, marker="s", color='tab:olive', label='9')
    
    tax.legend(title="Actions")
    
    
    for job in range(0,11):
        traj = 0
        while traj <= 51:  
            try:
                dir_states = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.states'
                dir_doses = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.controls'
                
                states = []
                f = open(dir_states,'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    xho, xgs, xrl = float(fields[0]), float(fields[1]), float(fields[2])
                    if xho > rb or xho < fb: break
                    else:
                        state = np.array([xho, xgs, xrl])
                        states.append(state)
                        
                tax.plot(states, color='black')  
            except:
                pass
            
            traj += 20
    
    plt.annotate('$x_{R^L}$',xy=(-0.023,-0.053))
    plt.annotate('$x_{H^O}$',xy=(0.98,-0.053))
    plt.annotate('$x_{G^S}$',xy=(0.529,0.853))
    tax.show()
    
    

def view_hjb_policy(num_jobs,num_traj,dir_std):
    rb, fb = 0.9, 0.1
    states, actions = [], []
    gab = 1e-5
    qg_max, qg_min = 102.06, 51.03
    ql_max, ql_min = 0.0054, 0.0027
    
    for job in range(num_jobs+1):
        for traj in range(num_traj):
            try:
                dir_states = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.states'
                dir_doses = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.controls' 
                
                count = 0
                f = open(dir_states,'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    xho, xgs, xrl = float(fields[0]), float(fields[1]), float(fields[2])
                    if xho < fb or xho > rb: break
                    else:
                        state = np.array([xho, xgs, xrl])
                        states.append(state)
                        count += 1 

                step = 0
                f = open(dir_doses, 'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    qg, ql = float(fields[0]), float(fields[1])
                    if 0.0-gab <= qg <= 0.0+gab and 0.0-gab <= ql <= 0.0+gab: action = 0
                    elif 0.0-gab <= qg <= 0.0+gab and ql_min-gab <= ql <= ql_min+gab: action = 1
                    elif 0.0-gab <= qg <= 0.0+gab and ql_max-gab <= ql <= ql_max+gab: action = 2
                    elif qg_min-gab <= qg <= qg_min+gab and 0.0-gab <= ql <= 0.0+gab: action = 3
                    elif qg_min-gab <= qg <= qg_min+gab and ql_min-gab <= ql <= ql_min+gab: action = 4
                    elif qg_min-gab <= qg <= qg_min+gab and ql_max-gab <= ql <= ql_max+gab: action = 5
                    elif qg_max-gab <= qg <= qg_max+gab and 0.0-gab <= ql <= 0.0+gab: action = 6
                    elif qg_max-gab <= qg <= qg_max+gab and ql_min-gab <= ql <= ql_min+gab: action = 7
                    elif qg_max-gab <= qg <= qg_max+gab and ql_max-gab <= ql <= ql_max+gab: action = 8
                    else: print(f'Error job {job} traj {traj}')
                    actions.append(action)
                    step += 1
                    if step == count: break
                    else: pass
                               
            except:
                #print(f'Trajectory {job}-{traj} not found ...')
                pass
        
    
    states_0, states_1, states_2, states_3, states_4, states_5 = [], [], [], [], [], []
    states_6, states_7, states_8 = [], [], []
    for index in range(len(states)):
        state = states[index]
        action = actions[index]
        if action == 0: states_0.append(state)
        elif action == 1: states_1.append(state)
        elif action == 2: states_2.append(state)
        elif action == 3: states_3.append(state)
        elif action == 4: states_4.append(state)
        elif action == 5: states_5.append(state)
        elif action == 6: states_6.append(state)
        elif action == 7: states_7.append(state)
        elif action == 8: states_8.append(state) 
        
        
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    if states_0 != []: tax.scatter(states_0, marker="s", color='tab:blue', label='1') 
    if states_1 != []: tax.scatter(states_1, marker="s", color='tab:orange', label='2')
    if states_2 != []: tax.scatter(states_2, marker="s", color='tab:green', label='3')
    if states_0 != []: tax.scatter(states_0, marker=".", color='tab:blue') 
    if states_3 != []: tax.scatter(states_3, marker="s", color='tab:red', label='4')
    if states_4 != []: tax.scatter(states_4, marker="s", color='tab:purple', label='5')
    if states_5 != []: tax.scatter(states_5, marker="s", color='tab:brown', label='6')
    if states_6 != []: tax.scatter(states_6, marker="s", color='tab:pink', label='7')
    if states_7 != []: tax.scatter(states_7, marker="s", color='tab:gray', label='8')
    if states_8 != []: tax.scatter(states_8, marker="s", color='tab:olive', label='9')
    
    tax.legend(title="Actions")
    
    
    for job in range(0,11):
        traj = 0
        while traj <= 51:  
            try:
                dir_states = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.states'
                dir_doses = dir_std + f'trajectory_{job}_{traj}/simulatedTrajectory.controls'
                
                states = []
                f = open(dir_states,'r')
                for line in f.readlines():
                    fields = line.split('\t')
                    xho, xgs, xrl = float(fields[0]), float(fields[1]), float(fields[2])
                    if xho > rb or xho < fb: break
                    else:
                        state = np.array([xho, xgs, xrl])
                        states.append(state)
                        
                tax.plot(states, color='black')  
            except:
                pass
            
            traj += 20
    
    plt.annotate('$x_{R^L}$',xy=(-0.023,-0.053))
    plt.annotate('$x_{H^O}$',xy=(0.98,-0.053))
    plt.annotate('$x_{G^S}$',xy=(0.529,0.853))
    tax.show()    