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




class SGDS_ES():
    
    def __init__(self,net,lr=0.001,momentum=0):
        
        self.lr=lr
        self.momentum=momentum
        self.popsize = 8
        self.num_list=range(self.popsize)
        self.num_params,self.model_shapes=cal_nparams(net)
        self.sigma=0.001
        self.mu = np.zeros(self.num_params)
        #self.solutions =np.zeros((self.popsize,self.num_params))
        #self.epsilon = np.random.randn(self.popsize, self.num_params)

        # self.V=np.zeros((self.popsize,self.NPARAMS))
        # self.V=np.random.randn(self.popsize,self.NPARAMS) * self.sigma 
        




    
    def updateModel(self,model,momentum):
        idx =0 
        i =0 
        model_shapes=self.model_shapes
    
        for param in model.parameters():
            delta = np.product(model_shapes[i])
            block = momentum[idx:idx+delta]
            block = np.reshape(block, model_shapes[i])
            i += 1
            idx += delta
            block_data = t.from_numpy(block).float()
            # if args.cuda:
            #   block_data = block_data.cuda() # using cuda 
            param.data -= block_data*self.lr 

    def recoverModel(self,model,momentum):

        idx =0 
        i =0 
        model_shapes=self.model_shapes
        #
        for param in model.parameters():
            delta = np.product(model_shapes[i])
            block = momentum[idx:idx+delta]
            block = np.reshape(block, model_shapes[i])
            i += 1
            idx += delta
            block_data = t.from_numpy(block).float() 
            param.data=param.data+block_data*self.lr 

 
        
    
    def step(self,net,data,target,base):
        loss_min=None
        lossnum=None
       
        reward =np.zeros(self.popsize)
        # keep elite and remove 
        num_list=self.num_list


        grad=[]
        for p in net.parameters():
            p = p.grad.data.cpu().numpy()
            grad.append(p.flatten()) #done 

        gradFlat=np.concatenate(grad)

        #solutions=self.solutions
        self.epsilon = np.random.randn(self.popsize, self.num_params)
        solutions=self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        #print(solutions.std())
        val=0       
        for num in num_list:
       
            newH=self.momentum*solutions[num]+gradFlat #update moment
            self.updateModel(net,newH)
            output = net(data)
            loss = F.nll_loss(output, target) 
            reward[num] = -(loss.data[0])
            
            if (loss_min is None) or (loss.data[0] < loss_min):
                loss_min=loss.data[0] # loss
                lossnum=num

            self.recoverModel(net,newH)
            # if loss.data[0]>base:
            #     val+=1 


            #solutions[num]=newH

            if loss.data[0]>base:
                #print("kidding ---> <----")
                if np.random.randn()>0.5:
                    reward[num]=-1*base
                else:
                    solutions[num]=newH
            else:
                #val+=1
                solutions[num]=newH

            # if loss.data[0]<base:
            #     if np.random.randn()>0.9:
            #         solutions[num]=newH
            #     else:
            #         reward[num]=-1*loss.data[0]
            # else:
            #     if np.random.randn()>0.1:
            #         solutions[num]=newH

            # solutions[num]=newH

        #moving to the next step
        # print("percent:{}".format(float(val)/8)) 

        self.updateModel(net,solutions[lossnum])
        
        def mate(a,b):

            c=copy.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        
       
        _std=reward.std()
        idx = np.argsort(reward)[::-1] 
        best_reward = reward[idx[0]]
        # print("{}{}".format(lossnum,idx[0]))
        # best_mu = solutions[idx[0]]
        # #
        # # reward = compute_centered_ranks(reward)
        # # l2_decay = compute_weight_decay(0.1, solutions)
        # # reward += l2_decay

        epsilon=solutions-self.mu*self.momentum-gradFlat
        # if _std==0:
        if _std==0:
            _std=0.000000000001 
            self.sigma*=0.9999
            # sigma*=0.99 #done 
        normalized_reward = (reward - np.mean(reward))/ _std
        deleta_Mu=0.001/(self.popsize*self.sigma)*np.dot(epsilon.T, normalized_reward)
        self.mu =(self.momentum*self.mu)+gradFlat+deleta_Mu
        #done 
        # self.mu=self.mu*self.momentum+gradFlat
        # self.sigma *=0.9999

        #self.mu=solutions[lossnum]
        # self.mu=solutions[lossnum] # the best one 

        # if self.sigma> 0.00001:
        # self.sigma*=0.99

        # self.mu+=gradFlat

        # h=solutions[lossnum]*self.momentum+gradFlat
        # self.updateModel(net,h)
        # others=np.argsort(reward)[::-1][elite_popsize:]
        # 
        # # reward = compute_centered_ranks(reward)
        # # l2_decay = compute_weight_decay(0.1, self.V)
        # # reward += l2_decay
        # # reward
        # elite_popsize=3
        # elite_idx=np.argsort(reward)[::-1][0:elite_popsize]  
        # other_idx=np.argsort(reward)[::-1][elite_popsize:]
      
        # elite_reward=reward[elite_idx] 

        # popsize=8-elite_popsize
        # choices=np.random.choice(num_list,2*popsize,list(reward))
        # crossOver=[mate(solutions[choices[2*i]],solutions[choices[2*i+1]]) for i in range(popsize)]
        # crossOver=np.array(crossOver)

        # solutions[other_idx]=crossOver
        # if _std==0:
        #     solutions+=np.random.randn(self.popsize, self.num_params)*self.sigma
        #     self.sigma*=0.999
        #self.mu=np.mean(solutions,axis=0)






      
        # # # #print(self.V[0].shape)
        # self.V[others]=solutions #update the remaing ones 
        # # if reward.std()>0.0001:
        # self.sigma*=0.9999
        # # if _std==0.0:
        # #     self.lr*=0.99
        # # else:
        # #     self.sigma=0.0001
        # # self.V=solutions
        # # self.updateModel(net,np.mean(self.V,axis=0))
        return best_reward,_std
        

        


