#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:20:33 2018

@author: dalitengelhardt
"""


import math
import numpy as np
import stochastic_sim_dual as ssim
import random

NUM_SPECIES = 4
POP_MAX = 10**7
POP_MIN = 10**6
NUM_START_POSSIBILITIES = 10
CAP = 1.2*10.**7
START_POPS_MAX = [POP_MAX,0,0,0]
BETA = [0.5,0.4,0.4,0.3]
DELTA=0.2

#first-use drug
GI_A = [0.1,0.8,10000,10000]

#second-use drug
GI_B = [0.1,0.8,2.,5.]


EPS_POP = 0.000001





TINT=4 #decision time step (hours)
MAXTIME=7*24.
MAX_INHIBIT = max(GI_B)*8.
NUM_DRUGS = 2

EPS_BAR=0.3
MUT_PROB = 10**(-6)

GAMMA_SHAPING = 0.99
REW_SCALE = 10
ACTION_REW_SCALE_A = 16
ACTION_REW_SCALE_B = ACTION_REW_SCALE_A*3
OVERALL_SCALE=1



class Environment:
  def __init__(self):
      self.is_done=False
      self.o_state = self.reset()
      self.max_inhibit = MAX_INHIBIT
      self.min_inhibit = 0.
      self.cap = CAP
      self.max_time = MAXTIME
      self.time_int = TINT
      self.eps_bar = EPS_BAR
      self.action_size = NUM_DRUGS
      self.num_species = NUM_SPECIES

      index = 0
      growth_numerator = 0
      base_inhib = 0
      while (index < self.num_species):
          init_growth = (BETA[index] - DELTA)*(1-sum(START_POPS_MAX)/self.cap)
          growth_numerator += init_growth*START_POPS_MAX[index]
          if (START_POPS_MAX[index] > EPS_POP):
              base_inhib_new = GI_A[index]*(-1+BETA[index]*MAXTIME/(DELTA*MAXTIME - math.log(START_POPS_MAX[index]*(self.cap-0.98)/(0.98*(self.cap-START_POPS_MAX[index])))))
              if (base_inhib_new > base_inhib):
                  base_inhib = base_inhib_new
          index += 1

      self.lo_growth = -DELTA
      self.base_inhibit = min(5*base_inhib,0.75*self.max_inhibit)


  def shape_function(self, state, tot_init_growth):
      
      growth_rate_scaled = state.item(-1)
      growth_rate = (tot_init_growth-self.lo_growth)*growth_rate_scaled+self.lo_growth

      shape_reward = REW_SCALE*(-growth_rate/tot_init_growth)

      return shape_reward
  
          

  def step(self, action, o_state, tot_inhibit, cur_time, pop_init, tot_init_growth, mut_can_happen = False,  mut_occurred = False):
      is_done=False
      
      curtime = cur_time + TINT
      
      mut_desc = []

      o_pop_levels = []
      for i in range(self.num_species):
          o_pop_levels.append((self.cap)**o_state.item(i)-1)
          
      actionA = action.item(0)
      actionB = action.item(1)      

      n_tot_inhibit = np.array([tot_inhibit.item(0) + actionA,tot_inhibit.item(1) + actionB])

      params = [DELTA, BETA, GI_A, GI_B, CAP, actionA, actionB]
      #mut_prob = np.random.normal(10**(-7), 5*10**(-8), 1).item(0)
      #mut_prob = random.randint(1, 10)*MUT_PROB
      soln_next = ssim.sde_sim(params, o_pop_levels, TINT, 0.1/CAP, MUT_PROB)


      n_pop_level = sum(soln_next) 


      if (n_pop_level > EPS_POP):
          n_growth_list = []
          for ind in range (self.num_species):
              n_growth = (-DELTA+BETA[ind]/(1+actionA/GI_A[ind]+actionB/GI_B[ind]))*soln_next[ind]#*(1-n_pop_level/self.cap)/n_pop_level
              n_growth_list.append(n_growth)
          n_growth_tot = sum(n_growth_list)*(1-n_pop_level/self.cap)/n_pop_level
      else:
          n_growth_tot = 0

      n_state  = np.zeros(NUM_SPECIES+1)
      for ind in range (NUM_SPECIES):
          n_state[ind] = math.log(soln_next[ind]+1)/math.log(CAP)
      n_state[-1] = (n_growth_tot-self.lo_growth)/(tot_init_growth-self.lo_growth)
      
      if (curtime>=MAXTIME):
          is_done = True
          
      if (n_pop_level >= 1 and curtime < MAXTIME):
          shaping_rew = GAMMA_SHAPING*self.shape_function(n_state,tot_init_growth) - self.shape_function(o_state,tot_init_growth)
          inhibitor_penalty_A = ACTION_REW_SCALE_A*((actionA/MAX_INHIBIT)**2)
          inhibitor_penalty_B = ACTION_REW_SCALE_B*((actionB/MAX_INHIBIT)**2)
          if (n_pop_level<=1.0001*pop_init):
              rew=shaping_rew - 10*(inhibitor_penalty_A+inhibitor_penalty_B)
          else:
              rew=shaping_rew -0.5*(inhibitor_penalty_A+inhibitor_penalty_B)
      elif (n_pop_level >= 1 and curtime >= MAXTIME):
          rew = -20 - 20*math.log(1+n_pop_level/pop_init)
      elif (n_pop_level<1):
          rew = 40*(1-(n_tot_inhibit.item(0)+(ACTION_REW_SCALE_B/ACTION_REW_SCALE_A)*n_tot_inhibit.item(1))/(2*MAX_INHIBIT*MAXTIME/TINT))
          is_done = True
      else:
          print('Exception: something went wrong, pop_level is ', n_pop_level)

      return n_state, rew, n_tot_inhibit, is_done, curtime, mut_occurred, mut_desc #new_state is numpy array, the rest are scalars except is_done is bool

  def reset(self): #return observation of initial state
  
      init_state  = np.zeros(NUM_SPECIES+1)
      pop_wt = random.randrange(POP_MIN,POP_MAX,(POP_MAX - POP_MIN)/NUM_START_POSSIBILITIES)
      init_state[0]=math.log(pop_wt+1)/math.log(CAP)
      pop_init_list = [math.log(pop_wt+1)/math.log(CAP)]
      for ind in range (NUM_SPECIES-1):
          init_state[ind+1] = 0
          pop_init_list.append(init_state[ind+1])
      init_state[-1] = 1
      pop_init = sum(pop_init_list)

      growth_numerator = 0
      for i in range(NUM_SPECIES):
          init_growth = (BETA[i] - DELTA)*(1-pop_init/CAP)
          growth_numerator += init_growth*pop_init_list[i]
      tot_init_growth = growth_numerator/pop_init
      return init_state, pop_init, tot_init_growth
     
     