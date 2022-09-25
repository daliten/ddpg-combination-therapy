#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:14:12 2018

@author: dalitengelhardt
"""



import random
import argparse
import collections
import numpy as np
from copy import deepcopy
import os
import datetime

import warnings

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter


from ddpg_env import Environment
from ddpg_model import ActorNet, CriticNet


#Hyperparameters
GAMMA = 0.99 #TD
ACTOR_LR = 0.00001
CRITIC_LR = 0.0001
REPLAY_SIZE=1000000
BATCH_SIZE= 512
REPLAY_NEED_SIZE=BATCH_SIZE*1.2
TAU=0.001 #target soft update parameter
NOISESIGMA=0.3 #standard deviation for OU noise


TEST_EPISODES = 150000 





class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)
        
    def normalize(self,sample_array):
        min_el=np.amin(sample_array, axis=0)
        max_el=np.amax(sample_array, axis=0)
        return (-0.5+(sample_array - min_el)/(max_el - min_el))
        

    def sample(self, batch_size):
        if (len(self.buffer)<batch_size):
            warnings.warn('Batch size is smaller than required sample size')
            return None
        else:
            sample_batch = random.sample(self.buffer, batch_size)
            states = [el[0] for el in sample_batch]
            actions = [el[1] for el in sample_batch]
            rewards = [el[2] for el in sample_batch]
            dones = [el[3] for el in sample_batch]
            new_states = [el[4] for el in sample_batch]
            
        if (len(self.buffer)>=REPLAY_SIZE):
            self.buffer.popleft()
            
       # states_norm = self.normalize(states)
       # actions_norm = self.normalize(actions)
       # rewards_norm = self.normalize(rewards)
       # new_states_norm = self.normalize(new_states)
        
        return states, actions, rewards, dones, new_states

        
# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
    
def target_update(tgt_net,net,tau): #soft update; make it a hard update by setting TAU=1
    for tgt_param,param in zip(tgt_net.parameters(),net.parameters()):
        tgt_param.data.copy_(tgt_param.data*(1.0-tau)+param.data*tau) 
    
def addnoise(noiseless_action): 
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_size),sigma=NOISESIGMA) 
    noise = ou_noise()
    noise_action1 = np.clip(noiseless_action.item(0) + noise.item(0), min_inhibit, max_inhibit)
    noise_action2 = np.clip(noiseless_action.item(1) + noise.item(1), min_inhibit, max_inhibit)
    return np.array([noise_action1,noise_action2])

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="thisrun")
    parser.add_argument("--actormodel", default = "actor_init.dat", help="Model file to load")
    parser.add_argument("--criticmodel", default = "critic_init.dat", help="Model file to load")
    args = parser.parse_args()
    save_path = os.path.join('saves', 'ddpg-' + args.name)
    os.makedirs(save_path, exist_ok=True)
    
    writer = SummaryWriter()    
    env = Environment()
    buffer = ReplayBuffer(REPLAY_SIZE)
    
    tau = TAU
    gamma = GAMMA
    state0, *_ = env.reset()
    obs_size = state0.size
    max_inhibit = env.max_inhibit
    base_inhibit = env.base_inhibit
    min_inhibit = env.min_inhibit
    max_time = env.max_time
    time_int = env.time_int
    learn_factor = max_time/time_int
    eps_bar = env.eps_bar
    action_size = env.action_size
    cap = env.cap
    
    actor_net = ActorNet(obs_size,base_inhibit,min_inhibit,max_inhibit,action_size)
    actor_net.load_state_dict(torch.load(args.actormodel))
    actor_tgt_net = deepcopy(ActorNet(obs_size,base_inhibit,min_inhibit,max_inhibit,action_size))
    actor_tgt_net.load_state_dict(torch.load(args.actormodel))
    
    critic_net = CriticNet(obs_size,action_size)
    critic_net.load_state_dict(torch.load(args.criticmodel))
    critic_tgt_net = deepcopy(CriticNet(obs_size,action_size))
    critic_tgt_net.load_state_dict(torch.load(args.criticmodel))
    
    #torch.save(actor_net.state_dict(), "actor_init.dat")
    #torch.save(critic_net.state_dict(), "critic_init.dat")

    actor_opt = optim.Adam(actor_net.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic_net.parameters(), lr=CRITIC_LR)
    

    highest_reward=0.
    highest_end_epis_rew=0.
    
    step_counter = 0

    
    for e in range(TEST_EPISODES):
                    
        total_reward = 0.0

        state, pop_init, tot_init_growth = env.reset() #state is a numpy array
        
        mut_will_happen = False
        mut_happened = False
        mut_desc_list = []
       
        action_seq = []
        pop_seq = []
       # tot_inhibit = np.zeros(action_size)
        tot_inhibit = np.zeros(2)
        cur_time = 0.
        
        rand = np.random.random_sample()
        if (eps_bar >= rand):
            mut_will_happen = True

           
        while True:
            is_done = False            
           
            torch_action = actor_net(torch.from_numpy(state).float()) #exploration noise added below
            action = torch_action.detach().numpy() #action is now a numpy array
            noisy_action = addnoise(action) 
            
            new_state, reward, tot_inhibit, is_done, cur_time, mut_happened, mut_desc = env.step(noisy_action, state, tot_inhibit, cur_time,  pop_init, tot_init_growth, mut_will_happen, mut_happened)
            
            action_seq.append(noisy_action.tolist())
            pop_seq.append([cap*new_state.item(0),cap*new_state.item(1),cap*new_state.item(2)])
            mut_desc_list.append(mut_desc)
            
            done_neg_mask=np.array([not is_done])
            done_neg_mask=done_neg_mask.astype(np.int)
            
            buffer.append((state.tolist(),noisy_action.tolist(),[reward], done_neg_mask.tolist(), new_state.tolist()))
           
           
            #Note that if done the is_done_neg is false
            #NOTE: Will have to change torch.Tensor([action]) if we have multiple inhibitors to torch.from_numpy(action) where action is a numpy array
            total_reward += reward

            if is_done:
                end_epis_rew = reward
                writer.add_scalar("end_of_epis_reward",end_epis_rew,e)
                writer.add_scalar("step_counter",step_counter,e)


            if len(buffer)>REPLAY_NEED_SIZE:
                #print('training from replay buffer')
                #This part happens when the replay buffer is full (has more than REPLAY_NEED_SIZE samples)
                #we then sample BATCH_SIZE from it, calculate the target, calculate the loss, 
                #and update Q via SGD by minimizing the loss wrt to model parameters
                
                states_batch, actions_batch, rewards_batch, dones_neg_batch, new_states_batch = buffer.sample(BATCH_SIZE) #get the unpacked batches
                    
                states_batch = torch.Tensor(states_batch) 
                actions_batch = torch.Tensor(actions_batch)
                new_states_batch = torch.Tensor(new_states_batch) 
                rewards_batch = torch.Tensor(rewards_batch)
                dones_neg_batch = torch.Tensor(dones_neg_batch)
                
                q_pred = critic_net(states_batch, actions_batch) 
                q_pred_mean = np.mean(q_pred.detach().numpy())
                writer.add_scalar("q_pred_mean = ", q_pred_mean, e)

                new_actions_batch = actor_tgt_net(new_states_batch) 
                new_crit_val = critic_tgt_net(new_states_batch, new_actions_batch)
                q_tgt = torch.addcmul(rewards_batch, gamma, dones_neg_batch, new_crit_val)
                q_tgt_mean = np.mean(q_tgt.detach().numpy())
                writer.add_scalar("q_tgt_mean = ", q_tgt_mean, e)
                
                #Then we update/train the critic
                critic_opt.zero_grad()
                critic_loss=F.mse_loss(q_pred,q_tgt.detach())
                writer.add_scalar("MSE_loss", critic_loss.detach().numpy().item(0), e)
                critic_loss.backward()
                critic_opt.step()
                
                #Then we update/train the actor
                actor_opt.zero_grad()
                actor_loss = -critic_net(states_batch,actor_net(states_batch)).mean() #this is the policy loss
                #actor_loss = -torch.mean(critic_net(states_batch,actor_net(states_batch))) #this is the policy loss
                writer.add_scalar("policy_loss", actor_loss.detach().numpy().item(0), e)
                actor_loss.backward()
                actor_opt.step()
                                       
                #soft update the target nets
                target_update(actor_tgt_net, actor_net, tau)
                target_update(critic_tgt_net, critic_net, tau) 
                
                step_counter +=1

            if is_done:
                break
            state = new_state

        
        if (e % 25 == 0):
            print('step number ',step_counter)
            print('episode',e,' reward',total_reward)
            print('end of epis reward', end_epis_rew)
            print('total inhibitor used = ',tot_inhibit)
            print('mut_happened?',mut_happened)
            print('mut_desc = [time,species,num] = ',mut_desc_list)
            print(action_seq)
           # print(pop_seq)
            print('--------')

        
        if ((e <= 600 and e % 20 == 0) or (e > 600 and e % 100 == 0)):
            name_act_tgt = "actor_target_%.3f.dat" % e
            name_act = "actor_%.3f.dat" % e
            name_crit_tgt = "critic_target_%.3f.dat" % e
            name_crit = "critic_%.3f.dat" % e
            fname_act_tgt = os.path.join(save_path, name_act_tgt)
            fname_act = os.path.join(save_path, name_act)
            fname_crit_tgt = os.path.join(save_path, name_crit_tgt)
            fname_crit = os.path.join(save_path, name_crit)
            torch.save(actor_tgt_net.state_dict(), fname_act_tgt) #should it be the target net?
            torch.save(actor_net.state_dict(), fname_act)
            torch.save(critic_tgt_net.state_dict(), fname_crit_tgt) #should it be the target net?
            torch.save(critic_net.state_dict(), fname_crit)
            print('buffer length is ', len(buffer), ' epis = ',e)

            
    name = "fin_target_{date:%Y-%m-%d_%H-%M-%S}.dat".format( date=datetime.datetime.now())
    fname = os.path.join(save_path, name)
    torch.save(actor_tgt_net.state_dict(), fname) #should it be the target net?   
    
    writer.close()  
    f.close()

