from ESUtil import * 
import torch.cuda as torch_c
import torch 
import numpy as np 

class OpenESCuda:
  ''' Basic Version of OpenAI Evolution Strategies.'''
  def __init__(self, num_params,             # number of model parameters
               sigma_init=0.1,               # initial standard deviation
               sigma_decay=0.999,            # anneal standard deviation
               sigma_limit=0.01,             # stop annealing if less than this
               learning_rate=0.01,          # learning rate for standard deviation
               learning_rate_decay = 0.9999, # annealing the learning rate
               learning_rate_limit = 0.001,  # stop annealing learning rate
               popsize=255,                  # population size
               antithetic=False,             # whether to use antithetic sampling
               forget_best=True            # forget historical best
               ):            

    self.num_params = num_params
    self.sigma_decay = sigma_decay
    self.sigma = sigma_init
    self.sigma_limit = sigma_limit
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    if self.popsize %2 ==1:
        self.popsize +=1 
        
    
    self.antithetic = antithetic
    self.half_popsize = int(self.popsize/2)
   
    self.reward = np.zeros(self.popsize)
    self.best_mu = torch_c.FloatTensor(self.num_params).fill_(0.0)
    self.mu = torch_c.FloatTensor(self.num_params).fill_(0.0) # I believe borrow the DNN initial val will help, try it later 
    self.best_reward = 0
    self.first_interation = True
    self.forget_best = forget_best
    self.diversity_base = 0 
    self.mdb = 0 

    print(self.sigma,self.learning_rate)
    
  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    
#     if self.antithetic:
#       self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
#       self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
#     else:
#       self.epsilon = np.random.randn(self.popsize, self.num_params)
#     self.epsilon = torch_c.FloatTensor(self.popsize, self.num_params).normal_() 

#     self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma
    self.epsilon_half = torch_c.FloatTensor(self.half_popsize, self.num_params).normal_()*self.sigma 
    self.epsilon = torch.cat((self.epsilon_half, -1*self.epsilon_half))
    self.solutions = self.mu.expand(self.epsilon.size())+ self.epsilon*self.sigma 
    return self.solutions

  def tell(self, reward):
    # input must be a numpy float array
    assert(len(reward) == self.popsize), "Inconsistent reward_table size reported."
    
    
#     reward = torch_compute_centered_ranks(reward,True)
#     reward=reward.cuda()
#     idx = np.argsort(reward)[::-1]
    y, idx = torch.sort(reward,0) 
    idx = reverse(idx)
    idx = idx.type(torch.LongTensor)
    
    
    # using the information over the all information: Simple 
    best_reward = reward[idx[0]]
    best_mu = self.solutions[idx[0]]
    
#     print(reward, best_reward)
    
    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # main bit:
    # standardize the rewards to have a gaussian distribution
    normalized_reward = (reward - torch.mean(reward)) / reward.std()
    normalized_reward = normalized_reward.view(1,self.popsize)
    self.mu += self.learning_rate/(self.popsize*self.sigma)*torch.mm(normalized_reward,self.epsilon)
#     self.mu += self.learning_rate/(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)

 

    # adjust sigma according to the adaptive sigma calculation
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

    if (self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

#   def done(self):
#     return False

  def current_param(self):
    return self.curr_best_mu
 

  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)

  def name(self):
    return "OES"
        