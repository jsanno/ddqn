#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils_plot_results import view_DDQN_policy, ddqn_vs_conventional, view_HJB_policy
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == "__main__":
    init_state = np.array([0.3, 0.6, 0.1])
    
    ## Scenario 1
    al, bl = 1.7e3, 9.8e3 #[uM]
    data = {"al": al,
            "bl": bl,
            "delta_t": 2.0,
            "max_steps": 300,   
            "sigma": 0.01,
            "penalty": -1000, 
            "state_size": 3,
            "action_size": 9,
            "EPISODES": 90_000,
            "train_each": 10}    
    pickle_out = open("data_input", "wb")
    pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    
    policy = 'DDQN_policy_scenario_1' 
    weights = 'DDQN_weights_scenario_1'    
    view_DDQN_policy(policy,weights)
   
    ddqn_vs_conventional(init_state,weights)   
    
    hjb_policy = 'HJB_policy_scenario_1'
    hjb_trajectories = 'HJB_trajectories_scenario_1'
    view_HJB_policy(hjb_policy,hjb_trajectories)
    
    pickle_in = open('HJB_trajectory_costs', 'rb')
    hjb_costs = pickle.load(pickle_in) 
    delta_t = data['delta_t']
    print(f'HJB cost: {np.mean(hjb_costs)*delta_t}')
    pickle_in = open("DDQN_policy_scenario_1", "rb")
    data = pickle.load(pickle_in)
    ddqn_scores= data['scores']
    ddqn_costs = [-score for score in ddqn_scores] 
    
    fig = plt.figure()
    plt.plot(ddqn_costs, label='DDQN')
    plt.plot(hjb_costs, '--', label='HJB')
    plt.legend()
    plt.show()
    
    stat = stats.ttest_ind(ddqn_costs, hjb_costs, equal_var=False)
    print(f'T_test p-value: {stat[1]}')  


    
    ## Scenario 2
    al, bl = 2e3, 9.5e3 #[uM]
    data = {"al": al,
            "bl": bl,
            "delta_t": 2.0,
            "max_steps": 300,   
            "sigma": 0.01,
            "penalty": -1000, 
            "state_size": 3,
            "action_size": 9,
            "EPISODES": 90_000,
            "train_each": 10}
    
    pickle_out = open("data_input", "wb")
    pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    policy = 'DDQN_policy_scenario_2' 
    weights = 'DDQN_weights_scenario_2'
    
    view_DDQN_policy(policy,weights) 
    ddqn_vs_conventional(init_state,weights)
    
    
    
    ## Scenario 3
    al, bl = 1.7e3, 21.3e3 #[uM]
    data = {"al": al,
            "bl": bl,
            "delta_t": 2.0,
            "max_steps": 300,   
            "sigma": 0.01,
            "penalty": -1000, 
            "state_size": 3,
            "action_size": 9,
            "EPISODES": 90_000,
            "train_each": 10}
    
    pickle_out = open("data_input", "wb")
    pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    policy = 'DDQN_policy_scenario_3' 
    weights = 'DDQN_weights_scenario_3'
    
    view_DDQN_policy(policy,weights) 
    ddqn_vs_conventional(init_state,weights)
    
    
    ## Scenario 4
    al, bl = 1.7e3, 28.3e3 #[uM]
    data = {"al": al,
            "bl": bl,
            "delta_t": 2.0,
            "max_steps": 300,   
            "sigma": 0.01,
            "penalty": -1000, 
            "state_size": 3,
            "action_size": 9,
            "EPISODES": 90_000,
            "train_each": 10}
    
    pickle_out = open("data_input", "wb")
    pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    policy = 'DDQN_policy_scenario_4' 
    weights = 'DDQN_weights_scenario_4'
    
    view_DDQN_policy(policy,weights)    
    ddqn_vs_conventional(init_state,weights)
