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


class SGDmomentum():
    def __init__(self,net,lr=0.001,momentum=0.99):
        self.lr=lr 
        self.momentum=0.99 
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
            #self.momentum_buffer[n]=self.momentum_buffer[n]*momentum+d_p
            # p.data=p.data -lr*self.momentum_buffer[n] 
            #move to next 

            p.data.add_(-1*lr*self.momentum_buffer[n])

        return 1,0 


class SGD_OES():
    
    def __init__(self,net,lr=0.001, momentum=0.99):
        self.lr = lr 
        self.v = {} # dict for recording 
        self.var = {}
        self.popsize = 8
        self.sigma = 0.001 
        # setting up the \mu 
        for name, p in net.named_parameters():
            self.v[name]=t.zeros_like(p.data)
            self.var[name]= [t.zeros_like(p.data).normal_()*self.sigma for i in range(self.popsize)]
        self.v['momentum']= momentum 
        self.var['momentum']=[2*self.sigma*(random.random()-0.5)+momentum for i in range(self.popsize)]          
    
    def step(self, net, data, target, base):
        best_index= -1 
        lr = self.lr 
        for i in range(self.popsize):
            for n, p in net.named_parameters():
                    d_p =  p.grad.data
                    temp = self.v[n]*self.var['momentum']+d_p
                    p.data.add_(-1*lr*temp)
             
            
        
        
        
        
    
    
    
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







        

        


