from ESUtil import * 
import torch 





class PEPGTorch:
    def __init__(self, num_params,             # number of model parameters
               sigma_init=0.10,              # initial standard deviation
               sigma_alpha=0.20,             # learning rate for standard deviation
               sigma_decay=0.999,            # anneal standard deviation
               sigma_limit=0.01,             # stop annealing if less than this
               learning_rate=0.01,           # learning rate for standard deviation
               learning_rate_decay = 0.9999, # annealing the learning rate
               learning_rate_limit = 0.001,  # stop annealing learning rate
               popsize=255,                  # population size
               done_threshold=1e-6,          # threshold when we say we are done
               average_baseline=True,        # set baseline to average of batch
               weight_decay=0.01,            # weight decay coefficient
               rank_fitness=True,            # use rank rather than fitness numbers
               forget_best=True):            # don't keep the historical best solution

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
          assert (self.popsize % 2 == 0), "Population size must be even"
          self.batch_size = int(self.popsize / 2)
        else:
          assert (self.popsize & 1), "Population size must be odd"
          self.batch_size = int((self.popsize - 1) / 2)
        self.forget_best = forget_best

        #


        self.batch_reward = torch.zeros(self.batch_size*2)
        self.mu = torch.zeros(self.num_params)
        self.sigma = torch.ones(self.num_params).mul_(self.sigma_init)
        self.curr_best_mu = torch.zeros(self.num_params)
        self.best_mu = torch.zeros(self.num_params) 
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
          self.forget_best = True # always forget the best one if we rank
        self.done_threshold = done_threshold

    def rms_stdev(self):
        sigma = self.sigma 
        return torch.mean(torch.sqrt(sigma*sigma)) 
    

    def ask(self):
        '''returns a list of parameters'''
    # antithetic sampling
    # self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon = torch.randn(self.batch_size,self.num_params)
        self.epsilon.mul_(self.sigma.expand(self.batch_size,self.num_params)) 
        self.epsilon_full = torch.cat((self.epsilon, -1*self.epsilon))
        
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            epsilon = torch.cat((torch.zeros(1,self.num_params),self.epsilon_full)) 
        solutions = self.mu.expand(epsilon.size())+epsilon
        self.solutions = solutions
        #print(epsilon)
        return solutions

    def tell(self, reward_table_result):
        reward_table = reward_table_result 
        reward_offset = 1
        if self.rank_fitness:
            reward_table = torch_compute_centered_ranks(reward_table,False)

        #done 
        if self.average_baseline:
            b = torch.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0] # baseline

        reward = reward_table[reward_offset:]
        y,idx =  torch.sort(reward,0)
        idx = reverse(idx)

        # print(idx.type())
        idx = idx.type(torch.LongTensor)

        best_reward = reward[idx[0]]
        # print(best_reward)
        # print(reward_table.type())
        #print("self.mu:{}".format(self.mu.size()))

        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        #print("generate:{}".format(self.curr_best_mu.size()))

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
          if self.forget_best or (self.curr_best_reward > self.best_reward):
            self.best_mu = best_mu
            self.best_reward = self.curr_best_reward

        # adaptive sigma
        # normalization
        stdev_reward = reward.std()
        epsilon = self.epsilon
        sigma = self.sigma
        S = ((epsilon * epsilon - (sigma * sigma).expand(epsilon.size())) / sigma.expand(epsilon.size()))
        reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
        rS = reward_avg - b
        #rS = torch.from_numpy(rS).view(1,self.batch_size).float()
        rS = rS.view(1,self.batch_size) 
        # print(rS.type())
        # print(S.type())
        delta_sigma = torch.mm(rS,S) / (2 * self.batch_size * stdev_reward)

        #     # move mean to the average of the best idx means
        rT = (reward[:self.batch_size] - reward[self.batch_size:])
        #rT =torch.from_numpy(rT).float().view(1,self.batch_size)
        rT = rT.view(1,self.batch_size)
        change_mu = self.learning_rate * torch.mm(rT,epsilon)
        #print(change_mu.size())
        self.mu =self.mu + change_mu.view(self.mu.size())
        
        # print(torch.sum(change_mu))
        # self.mu.add_(change_mu)  

        #     print(change_mu[0])

        #     # adjust sigma according to the adaptive sigma calculation
        change_sigma = self.sigma_alpha * delta_sigma
        change_sigma = torch.min(change_sigma, self.sigma)
        change_sigma = torch.max(change_sigma, - 0.5 * self.sigma)
        #print(change_sigma)
        # self.sigma.add_(change_sigma)
        self.sigma = self.sigma+ change_sigma.view(self.sigma.size())

        #     print(self.sigma)
        self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
          self.learning_rate *= self.learning_rate_decay

    def done(self):
        return (self.rms_stdev() < self.done_threshold)

    def current_param(self):
        return self.curr_best_mu

    def best_param(self):
        return self.best_mu

    def result(self): # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma, self.rms_stdev())

    def name(self):
        return "PEPG"
