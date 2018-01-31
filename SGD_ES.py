import torch as t
import math
import random
import copy 
import numpy as np 

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
        choiceList=random.sample(self.num_list,8)
        reward = np.zeros(8)
        # let us assume this es

        for num in choiceList:
            for name,parameters in net.named_parameters():
                #self.V[num][name]=self.momentum*self.V[num][name]+parameters.grad.data # update history 
                parameters.data=parameters.data-(self.momentum*self.V[num][name]+parameters.grad.data)*self.lr
            
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            reward[num]=-1*loss 
            
            if (loss_min is None) or (loss.data[0] < loss_min).all():
                loss_min=loss.data
                lossnum=num
            for name,parameters in net.named_parameters():
                parameters.data=parameters.data+(self.momentum*self.V[num][name]+parameters.grad.data)*self.lr
        
        self.popsize=len(choiceList) 
        # def submate()
        def mate(a,b):
            vsample=copy.copy(self.V[a])
            for name,parameters in net.named_parameters():
                c=np.random.rand()>0.5 
                if c:
                    vsample[name]=self.V[b][name]
            return vsample
        #
        #keep the elite samples and random-cross over to generate the other variants 
        #add simple random noise 
        choices=np.random.choice(self.num_list,2*self.popsize,list(reward))     
        solutions=[mate(choices[2*i],choices[2*i+1]) for i in range(self.popsize)]
         
        # for i in range(self.popsize):
        #     if num!=lossnum:
        #         self.V[num]=solutions[i]
        i=0
        for num in choiceList:
            if num!=lossnum:
                self.V[num]=solutions[i]
            else:
                self.V[num]=
            i+=1

       
        #moving around the best_mu 
        #cross-over the history 
        for name,parameters in net.named_parameters():
            parameters.data=parameters.data-self.V[lossnum][name]*self.lr