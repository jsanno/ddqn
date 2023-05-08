#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pickle
from utils_train_ddqn import train_agent, validate_agent
from joblib import Parallel, delayed



if __name__ == "__main__":
    train_flag = True
    validate_flag = True
    show_best = True
    
    ## Scenario 1
    al, bl = 1.7e3, 9.8e3 #[uM]
    
    ## Scenario 2
    # al, bl = 2e3, 9.5e3 #[uM]
    
    ## Scenario 3
    # al, bl = 1.7e3, 21.3e3 #[uM]
    
    ## Scenario 4
    # al, bl = 1.7e3, 28.3e3 #[uM]
   
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
    
  
    n_seeds = 10
    num_cores = n_seeds
    
    if train_flag:
        _ = Parallel(n_jobs=num_cores, verbose=5)(delayed(train_agent)(seed=seed) for seed in range(n_seeds))  # Train
    if validate_flag:
        _ = Parallel(n_jobs=num_cores, verbose=5)(delayed(validate_agent)(seed=seed) for seed in range(n_seeds))  # Validate


    if show_best:
        pickle_in = open("data_input", "rb")
        data = pickle.load(pickle_in)
        e = data['EPISODES']-1
        scores = []
        for seed in range(n_seeds):
            pickle_in = open(f"DDQN_policy_{seed}_{e}", "rb")
            data = pickle.load(pickle_in)
            score = data['average_score']
            scores.append(score)
        best_seed = np.argmax(scores)
        pickle_in = open(f"DDQN_policy_{best_seed}_{e}", 'rb')
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
        print(f'Num of recovered patients: {np.sum(recos)}')
        print(f'Num of patients not recovered: {np.sum(unrecos)}')
        print(f'Num of deceased patients: {np.sum(deads)}')
        print('----------------------')
        print(f'Maximum trajectory steps: {max(steps)}')
        print(f'Minimun trajectory steps: {min(steps)}')
        print(f'Average trajectory steps: {np.mean(steps)}')
        print('----------------------------------------------------------\n')             
        
