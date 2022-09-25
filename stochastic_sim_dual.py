#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:33:17 2019

@author: dalitengelhardt
"""
import numpy as np
import math

GI_EPS = 0.000001


def growth_calc(beta,gi1,gi2,delta,i1,i2):
    return beta/(1+i1/gi1+i2/gi2)-delta

def noise_calc(beta,gi1,gi2,delta,i1,i2):
    return beta/(1+i1/gi1+i2/gi2)+delta

def sde_sim(params, cur_pops, Tau, eps, mut_prob):
    tcur = 0
    delta, beta, gi_a, gi_b, cap, inhibA, inhibB = params

    cap_term = (1-sum(cur_pops)/cap)

    birth_rate_wt = beta[0]/(1+inhibA/gi_a[0]+inhibB/gi_b[0])
        
    while tcur < Tau:
        x_o = cur_pops
        dt = 0.01
        sqrtdt=np.sqrt(dt)
        N_mut_possible = 0
        if (birth_rate_wt>0 and x_o[0]>1):
            time_scale = 1/(birth_rate_wt*x_o[0])
            N_mut_possible = int(round(dt/time_scale))
            
        for ind0 in range(N_mut_possible):
            if (np.random.random_sample()<=mut_prob):
               # x_o[0]-=1
                pop_chosen = np.random.randint(1, 4)
                #print('pop_chosen = ',pop_chosen)
                if (pop_chosen == 1):
                    x_o[1]+=1
                if (pop_chosen == 2):
                    x_o[2]+=1
                if (pop_chosen == 3):
                    x_o[3]+=1
        
        x_n_list = []
        for ind in range(len(x_o)):
            x_n_i = x_o[ind] + dt * growth_calc(beta[ind], gi_a[ind], gi_b[ind], delta, inhibA, inhibB)*x_o[ind]*cap_term + np.sqrt(max(0,noise_calc(beta[ind], gi_a[ind], gi_b[ind], delta, inhibA, inhibB)*x_o[ind]*cap_term)) * sqrtdt *np.random.randn()
            if (x_n_i < 10 and x_n_i >= 1):
                prand = np.random.random_sample()
                if (prand<=0.5):
                    x_n_i = math.floor(x_n_i)
                elif (prand>0.5):
                    x_n_i = math.ceil(x_n_i)
            elif (x_n_i < 1):
                x_n_i = 0.
            x_n_list.append(x_n_i)
        
              
        tcur = tcur + dt


        cur_pops = x_n_list

        all_zero = True
        for i in range(len(cur_pops)):
            if (cur_pops[i] >= eps):
                all_zero = False
        
        if (all_zero):
            break
        
    return cur_pops
        
