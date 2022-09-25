Code associated with Engelhardt D. "Dynamic Control of Stochastic Evolution: A Deep Reinforcement Learning Approach to Adaptively Targeting Emergent Drug Resistance" JMLR 21 203:1-30 (arXiv:1903.11373).

A deep deterministic policy gradient (DDPG)-trained combination therapy dosing optimization algorithm using stochastic simulations of cell population dynamics. Uses PyTorch.

ddpg_train.py: training code
ddps_run.py: test code using saved net configurations
stochastic_sim_dual.py: stochastic simulation code