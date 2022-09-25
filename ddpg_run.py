#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:11:34 2018

@author: dalitengelhardt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ddpg_model import ActorNet
import torch
from ddpg_run_env import Environment 



if __name__ == "__main__":
    
    trial = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default = "saves/ddpg-thisrun/actor_130000.000.dat", help="Model file to load")
    #parser.add_argument("-m", "--model", default = "result_paper_version.dat", help="Model file to load")
    
    args = parser.parse_args()
    
    env = Environment()
    state0, *_ = env.reset()
    obs_size = state0.size
    
    max_inhibit = env.max_inhibit
    base_inhibit = env.base_inhibit
    min_inhibit = env.min_inhibit
    max_time = env.max_time
    time_int = env.time_int
    eps_bar = env.eps_bar
    cap = env.cap
    action_size = env.action_size
    
    net=ActorNet(obs_size,base_inhibit,min_inhibit,max_inhibit,action_size)
    net.load_state_dict(torch.load(args.model))
        
    action_seqA = []
    action_seqB = []
    tot_inhibit = np.zeros(action_size)
    tot_inhibit = np.zeros(2)
    total_reward=0.
    cur_time=0.
    
        
    state, pop_init, tot_init_growth = env.reset()
    mut_will_happen = False
    mut_happened = False
    mut_desc_list = []
    is_done = False
    pop_seq = []
    pop_seqA = []
    pop_seqB = []
    pop_seqC = []
    pop_seqD = []
    tlist=[]
    mut_desc_list = []

    
    rand = np.random.random_sample()
    already_happened=False
    eps_bar= 1.
    
    mut_will_happen_t=max_time+time_int
    if (eps_bar >= rand):
        mut_will_happen = True
        
    while True:
        torch_action = net(torch.from_numpy(state).float()) 
        action = torch_action.detach().numpy() #action is now a numpy array
        action_list=action.tolist()
        action_seqA.append(action_list[0])
        action_seqB.append(action_list[1])

        new_state, tot_inhibit, is_done, cur_time, mut_happened, mut_desc = env.step(action, state, tot_inhibit, cur_time,  pop_init, tot_init_growth, mut_will_happen, mut_happened)

        pop_seq.append([(cap)**new_state.item(0)-1,(cap)**new_state.item(1)-1,(cap)**new_state.item(2)-1])
        pop_seqA.append((cap)**new_state.item(0)-1)
        pop_seqB.append((cap)**new_state.item(1)-1)
        pop_seqC.append((cap)**new_state.item(2)-1)
        pop_seqD.append((cap)**new_state.item(3)-1)
        tlist.append(cur_time)
        mut_desc_list.append(mut_desc)
        #total_reward += reward
        if is_done:
            break
        state = new_state
    #print(pop_seq)
    tarray = np.array(tlist)
    popAarray = np.array(pop_seqA)
    popBarray = np.array(pop_seqB)
    popCarray = np.array(pop_seqC)
    popDarray = np.array(pop_seqD)
    inhibArrayA = np.array(action_seqA)
    inhibArrayB = np.array(action_seqB)
    
    tot_inhib=0
    for i in range(len(action_seqA)):
        tot_inhib += action_seqA[i]+action_seqB[i]
    tot_inhib = tot_inhib*4.
    epis_time = cur_time

    
    plt.tight_layout()
    figI, axI = plt.subplots(figsize=(4,3))
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_yscale('log')
    epis_time_formatted="{:.0f}".format(epis_time)
   # ax.set(xlabel='Time (h)', ylabel='Cell Concentration (CFU/mL)')
    ax.set_xlim([0,24*7])
    ax.set_ylim([1,10**7])
    ax.axhline(y=10**6, color='gray',linestyle='dashed')
    #ax.text=('Episode time = '+ epis_time_formatted + ' h')
    ax.text(115, 2*10**6, '$T_e$ = '+ epis_time_formatted + ' h', fontsize=12)
   # ax.set_title('Episode time = ', epis_time)
    ax.plot(tarray,popAarray, color = 'black')
    ax.plot(tarray,popBarray, color = 'blue')
    ax.plot(tarray,popCarray, color='green')
    ax.plot(tarray,popDarray, color='red')
    

    axI.set_ylim([0,5])
    axI.set_xlim([0,24*7])
    
    tot_inhibit_formatted="{:.2f}".format(sum(tot_inhibit))
    #axI.set(xlabel='Time (h)', ylabel='Inhibitor Concentration ($\mu$g/mL)')
    axI.text(82, 10.2, '$I_{tot}$ = '+ tot_inhibit_formatted + ' $\mu$g/mL', fontsize=12)
    #axI.set(xlabel='Time (h)', ylabel='Inhibitor Concentration ($\mu$g/mL)',title = 'Total dosage = %.3f $\mu$g/mL'% (float("{0:.2f}".format(tot_inhibit))) )
    axI.plot(tarray,inhibArrayA, color='orange')
    axI.plot(tarray,inhibArrayB, color='magenta')
    
    """
    name_fig = "pop_%.3f.png" % trial
    name_fig_I = "inhibit_%.3f.png" % trial
    fig.savefig(name_fig,dpi=300)
    figI.savefig(name_fig_I,dpi=300)
    """
    
    #print('total_reward = ',total_reward)
    #print('action_sequence = ',action_seq)
    #print('pop_sequence = ',pop_seq)
    #print('total inhibitor used = ',tot_inhibit)
    #print('end of episode reward = ',reward)
    
    if (mut_happened):
        print('mut_desc = [time,species,num] = ',mut_desc_list)
    else:
        print('mut did not happen')