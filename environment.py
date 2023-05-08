#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.integrate import odeint


class Cancer():
    def __init__(self, al, bl, delta_t, max_step, sigma, penalty, real_cost=False):
        self.al = al #[uM]
        self.bl = bl 
        self.delta_t = delta_t
        self.max_step = max_step
        self.real_cost = real_cost
        self.sigma = sigma
        self.penalty = penalty
        
        self.step_cntr = False
               
        self.kho =  14.37 #[uM]
        self.kgs = 5e3 #[uM]
        self.krl = 6.5e3 #[uM]
        self.so = 3.82 #[uM]
        self.sg = 17100 #[uM]
        self.Lsup = 10_000 #[uM]
        self.theta = 1e-8 #[uM^{-1}]
        self.rb = 0.9
        self.fb = 0.1
     
        
        """GENISTEIN"""
        self.kg = 7.0 #[uM]
        self.qg_min = 51.03
        self.qg_max = 102.06
        
        """AR-C155858"""
        self.kl = 2.3e-3 #[uM]
        self.ql_min = 0.0027
        self.ql_max = 0.0054
        
       
       
    def reset(self):
        self.step_cntr = 0
        
        xrl = -1
        
        while xrl < 0.0:
            xho = random.uniform(self.fb, self.rb)
            xgs = random.uniform(0.0, 1.0)
            xrl = 1.0-(xho+xgs)
                
        state = np.array([xho,xgs,xrl])
        return state
        
    
    
    def dynamic(self,state,t, al,bl, kg,kl, qg,ql, kho,kgs,krl,
                so,sg,Lsup,theta):
        
        xho, xgs, xrl = state[0], state[1], state[2]
        
        sl = al+bl*xgs
        if sl < Lsup: fho = so/(so+kho)
        else: fho = so/(so+kho) -theta*(sl-Lsup) 
        
        fgs = sg*kg/(sg*kg+kgs*(kg+qg))
        frl = sl*kl/((sl+krl)*(kl+ql))
        
        F=fho*xho+fgs*xgs+frl*xrl # Overall fitness
    
        d_xho = xho*(fho-F) 
        d_xgs = xgs*(fgs-F) 
        d_xrl = xrl*(frl-F)     
        
        state_ = np.array([d_xho,d_xgs,d_xrl])
    
        return state_
        
   
    def step(self,state,action):
        if action == 0:   qg, ql = 0.0,         0.0
        elif action == 1: qg, ql = 0.0,         self.ql_min
        elif action == 2: qg, ql = 0.0,         self.ql_max
        
        elif action == 3: qg, ql = self.qg_min, 0.0
        elif action == 4: qg, ql = self.qg_min, self.ql_min
        elif action == 5: qg, ql = self.qg_min, self.ql_max
        
        elif action == 6: qg, ql = self.qg_max, 0.0
        elif action == 7: qg, ql = self.qg_max, self.ql_min
        else:             qg, ql = self.qg_max, self.ql_max

                    
        t = np.array([0.0, self.delta_t])
        sol = odeint(self.dynamic, state, t, 
                     args=(self.al,self.bl,self.kg,self.kl,
                           qg,ql,self.kho,self.kgs,self.krl,self.so,self.sg,self.Lsup,self.theta
                           ,),atol=1.0e-32, rtol=1.0e-13)
        
        state_ = sol[1,:]
        
        if state[0] >= self.rb:
            reward = 0.0
            done = True
        elif state_[0] < self.fb:
            reward = self.penalty
            done = True
            self.step_cntr += 1
        else:
            reward = -(self.sigma + qg/self.qg_max + 10.0*ql/self.ql_max)
            done = False
            self.step_cntr += 1        
            if self.real_cost: 
                reward *= self.delta_t
            
        
        info = []
        
        return state_, reward, done, info
