import torch as t
import math
import random
import copy 
import numpy as np 
import torch.nn.functional as F
from ESUtil import * 





def cal_nparams(model):
    orig_params=[] 
    model_shapes=[]

    for param in model.parameters():
        p = param.data.cpu().numpy()
        model_shapes.append(p.shape)
        orig_params.append(p.flatten())
    orig_params_flat = np.concatenate(orig_params)
    NPARAMS = len(orig_params_flat)

    return int(NPARAMS), model_shapes



#done 
class SGDmomentum():
    def __init__(self,net,momentum,lr=0.001):
        self.lr=lr 
        self.momentum=momentum 
        self.momentum_buffer={} 
        for n,p in net.named_parameters():
            self.momentum_buffer[n]=t.zeros_like(p.data)


    def step(self,net,data,target,base):
        #
        momentum=self.momentum
        lr=self.lr 
        for n,p in net.named_parameters():
            d_p=p.grad.data
            self.momentum_buffer[n].mul_(momentum).add_(d_p) #update momentum_buff 
            p.data.add_(-1*lr*self.momentum_buffer[n])

        return 1,0,self.momentum

    
class SGD_OES_NN():
    # momentum: uniform distribution 
    # 
    def __init__(self,net,lr=0.001, momentum=0.99):
        self.lr = lr 
        self.v = {}
        self.popsize = 8
        self.sigma = 0.1 
        self.hlr = 0.1   # hyper-learning rate
        self.gsigma= 0.2 
        self.momentum = momentum 

        for name, p in net.named_parameters():
            self.v[name] = t.zeros_like(p.data)
        self.base  = self.alpha_reTrans(momentum)
        self.mu = np.array([self.base,1])
        self.num_params =2 


    def ask(self):

        self.epsilon = np.zeros([self.popsize,2])
        self.epsilon[:,0] = np.random.normal(0,self.sigma,self.popsize) # uniformal distribution 
        self.epsilon[:,1] = np.random.normal(0,self.gsigma,self.popsize) #normal distribution 


    def alpha_trans(self,i):
        return 1-np.power(0.1,i)
    

    def alpha_reTrans(self,val):
        return -1*np.log10(1-val)
        
    def step(self, net, data, target, base):
        lossnum= -1 
        loss_min = base  
        lr = self.lr 

      
        self.ask()
        reward = np.zeros(self.popsize) 
        
        for i in range(self.popsize):

            for name, p in net.named_parameters():
                d_p =  p.grad.data
                temp =  self.v[name]*(self.alpha_trans(self.mu[0]+self.epsilon[i][0]))
                temp += d_p * (self.mu[1]+self.epsilon[i][1]) 

                p.data.add_(-1*lr*temp)

            output=net(data)
            loss = F.nll_loss(output, target)  

            reward[i] = -loss.data[0] 
            # find the best index 
            if (loss.data < loss_min).all:
                loss_min=loss.data   
                lossnum=i
            # recover 
            for name, p in net.named_parameters():
                d_p =  p.grad.data
                temp =  self.v[name]*(self.alpha_trans(self.mu[0]+self.epsilon[i][0]))
                temp += d_p * (self.mu[1]+self.epsilon[i][1]) 

                p.data.add_(lr*temp)

        #reward = compute_centered_ranks(reward)   

        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        #update the expectation of 1 
        self.mu[0] += self.hlr/(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)[0] # update the beta 
        self.mu[1] += self.hlr/(self.popsize*self.gsigma)*np.dot(self.epsilon.T, normalized_reward)[1]
        
        self.mu[0] = max(1,self.mu[0])
        self.mu[0] = min(4,self.mu[0])
        c_momentum= self.mu[0]
        for name, p in net.named_parameters():
            d_p = p.grad.data
            self.v[name].mul_(self.alpha_trans(self.mu[0])).add_(d_p*self.mu[1])
            p.data.add_(-1*lr*self.v[name])
        

        return loss_min[0], np.std(reward), self.alpha_trans(c_momentum)

class SGD_OES():
    
    def __init__(self,net,lr=0.001, momentum=0.99):
        self.lr = lr 
        self.v = {} # dict for recording 
        self.var = {}
        self.popsize = 8
        self.sigma = 0.01 
        self.gsigma = 0.2 
        self.hlr = 0.1   # hyper-learning rate 
       
        for name, p in net.named_parameters():
            self.v[name]=t.zeros_like(p.data)
            self.v[name+"g"]=1 
            self.var[name]=np.random.normal(0,self.gsigma,self.popsize) 
           
        self.v['momentum']= momentum 
        self.momentum = momentum
        self.var['momentum']=np.random.normal(0,self.sigma,self.popsize)         
        
        
    def step(self, net, data, target, base):
        lossnum= -1 
        loss_min = base  
        lr = self.lr 

        reward = np.zeros(self.popsize) 
        
  
        # using this object 
        for i in range(self.popsize):

            for name, p in net.named_parameters():
                d_p =  p.grad.data
                temp =  self.v[name]*(self.v['momentum']+self.var['momentum'][i])
                
                temp =  temp + d_p*(self.v[name+"g"]+self.var[name][i])
                p.data.add_(-1*lr*temp)

            output=net(data)
            loss = F.nll_loss(output, target)  

            reward[i] = -loss.data[0] 
            # find the best index 
            if (loss.data < loss_min).all:
                loss_min=loss.data   
                lossnum=i
            # recover 
            for name, p in net.named_parameters():
                d_p  = p.grad.data
                temp =  self.v[name]*(self.v['momentum']+self.var['momentum'][i])
                temp =  temp + d_p*(self.v[name+"g"]+self.var[name][i])
                p.data.add_(lr*temp)


        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        #update the expectation of 1, momentum 
        self.v['momentum'] += self.hlr/(self.popsize*self.sigma)*np.dot(self.var['momentum'].T, normalized_reward)
        self.v['momentum'] = min(1, self.v['momentum'])
        self.v['momentum'] = max(0, self.v['momentum'])
        c_momentum= self.v['momentum']
        for name, p in net.named_parameters():
            d_p = p.grad.data
            self.v[name+"g"] += self.hlr/(self.popsize*self.gsigma)*np.dot(self.var[name].T, normalized_reward)
            self.v[name].mul_(self.v['momentum']).add_(d_p*(self.v[name+"g"]))
            p.data.add_(-1*lr*self.v[name]) 
            self.v[name+"g"] = 1 

        self.v['momentum'] = 0.99
        
        return loss_min[0], np.std(reward), c_momentum
             

class SGD_PEPG():
    # let us consider single g 
    def __init__(self,net,lr=0.001, momentum=0.99):
        self.lr = lr 
        self.v = {} # dict for recording 

        self.popsize = 8
        self.sigma_init = 0.01 
        self.hlr = 0.1    # hyper-learning rate 
        self.hlr_limit = 0.01 
        self.sigma_alpha = 0.2 

        for name, p in net.named_parameters():
            self.v[name]=t.zeros_like(p.data)
        

        self.batch_size = int((self.popsize)/2)  
        self.num_params = 2 
        self.momentum = momentum
        self.mu = np.array([momentum,1]) # 
        self.sigma = np.ones(self.num_params)*self.sigma_init 

    def ask(self):
        # anthithetic sampling 

        self.epsilon = np.random.randn(self.batch_size,self.num_params)*self.sigma.reshape(1,self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, -self.epsilon])

    def step(self, net, data, target, base):
        loss_min=base 
        lossnum= -1 
        reward = np.zeros(self.popsize)
        lr = self.lr 
        self.ask()
        #print(self.epsilon)
        
        
        for name, p in net.named_parameters():
            d_p = p.grad.data
            temp = self.v[name]*(self.mu[0])
            temp = temp + d_p 
            p.data.add_(-1*lr*temp)
        
        output = net(data)
        loss = F.nll_loss(output, target)
        b = -loss.data[0]
        
        for name, p in net.named_parameters():
            d_p = p.grad.data
            temp = self.v[name]*(self.mu[0])
            temp = temp + d_p 
            p.data.add_(lr*temp)
        
        
        for i in range(self.popsize):
            #move one noisy step 
            for name, p in net.named_parameters():
                d_p =  p.grad.data
                temp =  self.v[name]*(self.mu[0]+self.epsilon_full[i,0]) 
                temp =  temp + d_p*(self.mu[1]+self.epsilon_full[i,1])
                p.data.add_(-1*lr*temp)

            output=net(data)
            loss = F.nll_loss(output, target)  

            reward[i] = -loss.data[0] 
            # find the best index 
            if (loss.data < loss_min).all:
                loss_min=loss.data   
                lossnum=i
            # recover the model 
            for name, p in net.named_parameters():
                d_p =  p.grad.data
                temp =  self.v[name]*(self.mu[0]+self.epsilon_full[i,0]) 
                temp =  temp + d_p*(self.mu[1]+self.epsilon_full[i,1])
                p.data.add_(lr*temp)
                
#         normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        # update the sigma and mu 
#         b= -1*base
        stdev_reward = reward.std() 
        epsilon = self.epsilon
        sigma = self.sigma
        S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
        reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
        rS = reward_avg - b
        delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)


        rT = (reward[:self.batch_size] - reward[self.batch_size:])

        # update the mu 
        change_mu = self.hlr * np.dot(rT, epsilon)
        self.mu += change_mu
        self.mu[0] = min(1, self.mu[0])
        self.mu[0] = max(0, self.mu[0])

        # update the sigma 
        change_sigma = self.sigma_alpha * delta_sigma

#         change_sigma = np.minimum(change_sigma, self.sigma)
#         change_sigma = np.maximum(change_sigma, - 0.5 * self.sigma)
        change_sigma [1] = max(0,change_sigma[1])
        self.sigma += change_sigma

        # one real step forward  
        for name, p in net.named_parameters():
            d_p =  p.grad.data
            self.v[name].mul_(self.mu[0]).add_(d_p*self.mu[1])

            p.data.add_(-1*lr*self.v[name])
        #reset-the g multilper 
        self.mu[1]= 1
        c_momentum= self.mu[0]
        
        return loss_min[0], self.sigma[1], c_momentum

    
        
        
    
    
    
# class SGD_GA():

#     def __init__(self,net,lr=0.001,momentum=0.99):
#         self.lr=lr 
#         self.momentum=0.99 
#         self.momentum_buffer={} #the mean 
#         self.sigma_buffer={}    #the variance 
#         for n,p in net.named_parameters():
#             self.momentum_buffer[n]=t.zeros_like(p.data)  # the same idea 
#             self.sigma_buffer[n]=t.zeros_like(p.data)   

#         self.popsize = 8 
#         self.first_iteration = True 
        


#     def step(self,net,data,target,base):

#         popsize=self.popsize
#         momentum=self.momentum
#         lr=self.lr 
#         lossnum = None 
#         loss_min = None 
#         rewards = np.zeros(popsize)

#         deviation =self.ask()


#         for num in range(popsize):
#             momentum_buffer=copy.deepcopy(self.momentum_buffer)
#             # momentum_buffer=copy.copy(self.momentum_buffer)
#             for n, p in net.named_parameters():
#                 d_p=p.grad.data
#                 momentum_buffer[n].add_(self.sigma_buffer[n])
#                 momentum_buffer[n].mul_(momentum)
#                 momentum_buffer[n].add_(deviation[num]*d_p) 
             
#                 #update gradient val 
#                 p.data.add_(-1*lr*momentum_buffer[n])


#             output=net(data)
#             loss = F.nll_loss(output, target)  #cal_entropy 
#             rewards[num] =-1*loss.data[0] 
            
#             if (loss_min is None) or (loss.data < loss_min).all:
#                 loss_min=loss.data   # loss
#                 lossnum=num

#             if loss.data[0]>-1*rewards[1]:
#                 deviation[num]=1 

            

#             for n,p in net.named_parameters():
#                 p.data.add_(lr*momentum_buffer[n]) #recover gradient 





#         # #Move to the next step 
#         # momentum_buffer=copy.copy(self.momentum_buffer)
#         if loss_min[0]<base or random.random()>0.5:

#             for n,p in net.named_parameters():
#                 d_p=p.grad.data
#                 momentum_buffer[n].add_(self.sigma_buffer[n])
#                 momentum_buffer[n].mul_(momentum)
#                 momentum_buffer[n].add_(deviation[lossnum]*d_p) 


#                 p.data.add_(-1*lr*momentum_buffer[n])

#                 self.sigma_buffer[n]=d_p
#                 self.sigma_buffer[n].mul_(self.momentum) 

#                 self.momentum_buffer[n].add_(self.sigma_buffer[n])
#                 self.momentum_buffer[n].mul_(momentum)
#                 self.momentum_buffer[n].add_(np.mean(deviation)*d_p)
#         # else:
#         # for n,p in net.named_parameters():
#         #     d_p=p.grad.data
#         #     self.momentum_buffer[n].mul_(momentum).add_(d_p) #update momentum_buff 
#         #     p.data.add_(-1*lr*self.momentum_buffer[n])






        

#         return -1,0



#     def ask(self):

#         popsize=self.popsize
#         deviation=np.zeros(popsize)
#         deviation[1]=1
#         deviation[2:]=0.001*np.random.randn(popsize-2)+1


#         return deviation







        

        


