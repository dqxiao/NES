import torch as t
import math
import random
import copy 
import numpy as np 

def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks


class SGDS_ES():
    
    def __init__(self, net,lr=0.001,momentum=0.99):
        self.num_list=range(8)
#        self.momentum_list=[]
#        for i in range(3,11):
#            self.momentum_list.append(1-1.0/(2**i))
        self.lr=lr
        self.momentum=momentum
        self.V={}
        
        for num in self.num_list:
            self.V[num]={}
            for name,parameters in net.named_parameters():
                self.V[num][name]=(t.zeros_like(parameters.data))
           
   
 
        
    
    def step(self,net,inputs,labels,criterion,flag):
        loss_min=None
        lossnum=None
       
        reward =np.zeros(8)
        # keep elite and remove 
        num_list=self.num_list
        
        for num in num_list:
            for name,parameters in net.named_parameters():
                
                self.V[num][name]=self.momentum*self.V[num][name]+parameters.grad.data # update history 
                parameters.data=parameters.data-self.V[num][name]*self.lr
            
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            reward[num] = -1*(loss.data[0]) # larger reward means less loss 
            
            
            if (loss_min is None) or (loss.data[0] < loss_min).all():
                loss_min=loss.data
                lossnum=num
            for name,parameters in net.named_parameters():
                parameters.data=parameters.data+self.V[num][name]*self.lr
        
        
        #print(reward)  
        def mate(a,b):
            vsample=copy.copy(self.V[a])
            for name,parameters in net.named_parameters():
                
                c=np.random.rand()>0.5 
                if c:
                    vsample[name]=self.V[b][name]
            return vsample
        
        def mute(sample):
            for name,parameters in net.named_parameters():
                data=sample[name].cpu().numpy()
                print(data.shape)
                ep =np.random.randn(data.shape)*0.001
                data +=ep 
                sample[name]=torch.from_numpy(data).float()
                
       
        for name,parameters in net.named_parameters():
            parameters.data=parameters.data-self.V[lossnum][name]*self.lr
        elite_popsize=4
        reward=np.array(reward) 
        
        print(reward.std())
        elite_size=idx = np.argsort(reward)[::-1][0:elite_popsize]  
        elite_params=[]
        for item in elite_size:
            elite_params.append(self.V[item])  
        elite_reward=reward[elite_size]
        
        elite_range=[item for item in range(elite_popsize)]
        popsize=8-elite_popsize
        
        choices=np.random.choice(elite_range,2*popsize,list(elite_reward))
        solutions=[mate(choices[2*i],choices[2*i+1]) for i in range(popsize)] #cross-over 
        _=[mute(s) for s in solutions]
        
        elite_params+=solutions
        
        for idx, num in enumerate(num_list):
            self.V[num]=elite_params[idx]