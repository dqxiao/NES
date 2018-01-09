import numpy as np 
from ESUtil import *

class NoveltyGA:
  '''Simple Genetic Algorithm.'''
  def __init__(self, num_params,      # number of model parameters
               sigma_init=0.1,        # initial standard deviation
               sigma_decay=0.999,     # anneal standard deviation
               sigma_limit=0.01,      # stop annealing if less than this
               popsize=255,           # population size
               novel_ratio=0.1,       # percentage of the novel 
               elite_ratio=0.1,       # percentage of the elists 
               forget_best=False,     # forget the historical best elites
               forget_history=False,  # forget the historical novel instance
               done_threshold=1e-6,   # threshold when we say we are done
               behavior_size=100      # bevavior_size 
               ):  

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize
    self.bevavior_size=int(behavior_size)   # the observation of bevavior 


    #keep tracking of novel params
    self.novel_ratio = novel_ratio
    self.novel_popsize = int(self.popsize * self.novel_ratio) 
    self.novel_obs= np.zeros((self.novel_popsize,self.bevavior_size))
    self.novel_params = np.zeros((self.novel_popsize, self.num_params))
    self.novelty  = np.zeros(self.novel_popsize)


    #keep tracking of the elists
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.elite_params  = np.zeros((self.elite_popsize,self.num_params))  
    self.elite_rewards = np.zeros(self.elite_popsize)

    #other setting 

    self.sigma = self.sigma_init
    self.first_iteration = True
    self.forget_best = forget_best
    self.done_threshold = done_threshold

    #additional 
    self.forget_history = forget_history




  def rms_stdev(self):
    return self.sigma # same sigma for all parameters.



  def tell(self,reward_table_result):
    #keep_tracking the best params 
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    if (self.forget_best or self.first_iteration):
      reward = reward_table_result
      solution = self.solutions
    else:
      reward = np.concatenate([reward_table_result, self.elite_rewards])
      solution = np.concatenate([self.solutions, self.elite_params])

    idx = np.argsort(reward)[::-1][0:self.elite_popsize]

    self.elite_rewards = reward[idx]
    self.elite_params = solution[idx]

    # 
    self.curr_best_reward = self.elite_rewards[0] 


    if self.first_iteration or (self.curr_best_reward > self.best_reward):
      self.first_iteration = False
      self.best_reward = self.elite_rewards[0]
      self.best_param = np.copy(self.elite_params[0])

    #decay Variance seems hand-coded 
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay
    

  def tellMore(self,observation):
    #keep_tracking the novel params 

    assert(len(observation)==self.popsize), "Inconsistent observation reported"


    if (self.forget_history or self.first_iteration):
      obs = observation
      solutions = self.solutions
    else:
      obs = np.concatenate([observation, self.novel_obs]) 
      solutions = np.concatenate([self.solutions,self.novel_params])

    # do you think the t-sne will help us to find right centers of species 
    distance=compute_novelty_obs(obs) # done 

    idx=np.argsort(distance)[::-1][0:self.novel_popsize]  # keep the representative anchors 

    self.novel_obs = obs[idx]
    self.novel_params = solutions[idx]
    self.novelty     =distance[idx]



  def ask(self): 
    """
    Generate next generation for anchor examples 
    """

    self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma

    solutions=[] 

    def mate(a, b):
      c = np.copy(a)
      idx = np.where(np.random.rand((c.size)) > 0.5)
      c[idx] = b[idx]
      return c


    #novelty-measure is based on its relative position on the current population 
    curr_novelty = self.novelty   
    # print(self.nov)
    novelty_range = [item for item in range(self.novel_popsize)] 
    choices=np.random.choice(novelty_range,self.popsize+1,list(curr_novelty))


    novel_params = self.novel_params
    solutions=[mate(novel_params[choices[i]], novel_params[choices[i+1]])+ self.epsilon[i] for i in range(self.popsize)] 

    solutions = np.array(solutions)
  
    self.solutions=solutions  # current solution 

    return solutions


  def best_param(self):
    return self.best_param

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)

  def current_param(self):
    return self.elite_params[0]