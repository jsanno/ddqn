#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:16:02 2023

@author: dst
"""

import numpy as np
import pickle
import random
import os
from utils import train_agent, validate_agent
from joblib import Parallel, delayed



if __name__ == "__main__":
    train_flag = False
    validate_flag = True
    show_best = True
    
    # Fig 1a 1c
    al, bl = 1.7e3, 9.8e3 #[uM]
    
    # # Fig 1b 1d
    # al, bl = 2e3, 9.5e3 #[uM]
    
    # Fig 2a 2c
    # al, bl = 1.7e3, 21.3e3 #[uM]
    
    # Fig 2b 2d
    # al, bl = 1.7e3, 28.3e3 #[uM]
   
    data = {"al": al,
            "bl": bl,
            "delta_t": 2.0,
            "max_steps": 300,   
            "sigma": 0.01,
            "penalty": -1000, 
            "state_size": 3,
            "action_size": 9,
            #"EPISODES": 90_000,
            "EPISODES": 10,
            "train_each": 10,        
            "seed": None}
    
    pickle_out = open("data_input", "wb")
    pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()
    
  
    n_seeds = 10
    num_cores = n_seeds
    
    if train_flag:
        _ = Parallel(n_jobs=num_cores, verbose=5)(delayed(train_agent)(seed=seed) for seed in range(n_seeds))  # Train
    if validate_flag:
        _ = Parallel(n_jobs=num_cores, verbose=5)(delayed(validate_agent)(seed=seed) for seed in range(n_seeds))  # Validate


    if show_best:
        scores = []
        for seed in range(n_seeds):
            pickle_in = open("DDQN_validate_" + str(seed), "rb")
            data = pickle.load(pickle_in)
            score = data['final_cost']
            scores.append(score)
        best_seed = np.argmax(scores)
        pickle_in = open("DDQN_validate_" + str(best_seed), 'rb')
        data = pickle.load(pickle_in)
        scores = data['scores']
        deads = data['deads']
        recos = data['recos']
        unrecos = data['unrecos']
        steps = data['steps']
       
        print('\n*--------------------------------------------------------*')        
        print(f'                BEST RESULTS: seed {best_seed}')
        print('*--------------------------------------------------------*') 
        print(f'average optimal policy cost: {np.mean(scores)}')
        print(f'maximum optimal policy cost: {max(scores)}')
        print(f'minimun optimal policy cost: {min(scores)}')
        print('----------------------')
        print(f'Num of recos with optimal policy: {np.sum(recos)}')
        print(f'Num of unrecos with optimal policy: {np.sum(unrecos)}')
        print(f'Num of deads with optimal policy: {np.sum(deads)}')
        print('----------------------')
        print(f'Maximum trajectory steps: {max(steps)}')
        print(f'Minimun trajectory steps: {min(steps)}')
        print(f'Average trajectory steps: {np.mean(steps)}')
        print('----------------------------------------------------------\n')
        
        
        