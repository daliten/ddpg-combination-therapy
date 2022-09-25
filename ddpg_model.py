#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:25:29 2018

@author: dalitengelhardt
"""
import torch
import torch.nn as nn


HIDDEN1=60
HIDDEN2=45




def init_weights(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(bias_val)
     #   m.weight.data.fill_(0.01)
    #    print(m.bias)
    
def init_bias_zero(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(0.)
        
class ActorNet(nn.Module):
                
    def __init__(self, obs_size, base_inhibit, min_val_final, max_val_final, act_size):
        super(ActorNet, self).__init__()
        
        hidden1=HIDDEN1
        hidden2=HIDDEN2
        
        global bias_val 
        bias_val = base_inhibit


        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2),#extra
            nn.ReLU(),#extra
           # nn.Linear(hidden2, hidden2),#extra
           # nn.ReLU(),#extra
            nn.Linear(hidden2, act_size),
            nn.Hardtanh(min_val_final,max_val_final)
        )
        self.net.apply(init_weights)
        

    def forward(self, x):
        return self.net(x)
    
    
    
class CriticNet(nn.Module):
    def __init__(self, obs_size, act_size):
        super(CriticNet, self).__init__()

        hidden1=HIDDEN1
        hidden2=HIDDEN2
        
        self.obs_net = nn.Sequential(
                nn.Linear(obs_size,hidden1),
                nn.ReLU()
        )
        
        self.out_net = nn.Sequential(
                nn.Linear(hidden1 + act_size, hidden2),
                nn.ReLU(),
               # nn.Linear(hidden2, hidden2),#extra
               # nn.ReLU(),#extra
                nn.Linear(hidden2, 1)
        )
        
    def forward(self,x,act):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, act], dim = 1))
    
