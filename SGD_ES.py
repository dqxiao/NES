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


class SGD_GA():

    def __init__(self,net,lr=0.001,momentum=0.99):
        self.lr=lr 
        self.momentum=0.99 
        self.momentum_buffer={} #the mean 
        self.sigma_buffer={}    #the variance 
        for n,p in net.named_parameters():
            self.momentum_buffer[n]=t.zeros_like(p.data)  # the same idea 
            self.sigma_buffer[n]=t.zeros_like(p.data)   

        self.popsize = 8 
        self.first_iteration = True 
        


    def step(self,net,data,target,base):

        popsize=self.popsize
        momentum=self.momentum
        lr=self.lr 
        lossnum = None 
        loss_min = None 
        rewards = np.zeros(popsize)

        deviation =self.ask()


        for num in range(popsize):
            momentum_buffer=copy.deepcopy(self.momentum_buffer)
            # momentum_buffer=copy.copy(self.momentum_buffer)
            for n, p in net.named_parameters():
                d_p=p.grad.data
                momentum_buffer[n].add_(self.sigma_buffer[n])
                momentum_buffer[n].mul_(momentum)
                momentum_buffer[n].add_(deviation[num]*d_p) 
             
                #update gradient val 
                p.data.add_(-1*lr*momentum_buffer[n])


            output=net(data)
            loss = F.nll_loss(output, target)  #cal_entropy 
            rewards[num] =-1*loss.data[0] 
            
            if (loss_min is None) or (loss.data < loss_min).all:
                loss_min=loss.data   # loss
                lossnum=num

            if loss.data[0]>-1*rewards[1]:
                deviation[num]=1 

            

            for n,p in net.named_parameters():
                p.data.add_(lr*momentum_buffer[n]) #recover gradient 





        # #Move to the next step 
        # momentum_buffer=copy.copy(self.momentum_buffer)
        if loss_min[0]<base or random.random()>0.5:

            for n,p in net.named_parameters():
                d_p=p.grad.data
                momentum_buffer[n].add_(self.sigma_buffer[n])
                momentum_buffer[n].mul_(momentum)
                momentum_buffer[n].add_(deviation[lossnum]*d_p) 


                p.data.add_(-1*lr*momentum_buffer[n])

                self.sigma_buffer[n]=d_p
                self.sigma_buffer[n].mul_(self.momentum) 

                self.momentum_buffer[n].add_(self.sigma_buffer[n])
                self.momentum_buffer[n].mul_(momentum)
                self.momentum_buffer[n].add_(np.mean(deviation)*d_p)
        # else:
        # for n,p in net.named_parameters():
        #     d_p=p.grad.data
        #     self.momentum_buffer[n].mul_(momentum).add_(d_p) #update momentum_buff 
        #     p.data.add_(-1*lr*self.momentum_buffer[n])






        

        return -1,0



    def ask(self):

        popsize=self.popsize
        deviation=np.zeros(popsize)
        deviation[1]=1
        deviation[2:]=0.001*np.random.randn(popsize-2)+1


        return deviation






# class SGDS_GA():

#     def __init__(self,net,lr=0.001, momentum=0.99):
#         self.lr=lr 
#         self.momentum =momentum




# class SGDS_ES():
    
#     def __init__(self,net,lr=0.001,momentum=0):
        
#         self.lr=lr
#         self.momentum=momentum
#         self.popsize = 15
#         self.num_list=range(self.popsize)
#         self.num_params,self.model_shapes=cal_nparams(net)
#         self.batch_size= 7
#         self.average_baseline= False 

#         self.sigma= np.ones(self.num_params)*0.001
#         self.mu = np.zeros(self.num_params) # the history 
#         self.first_interation=True 
#         self.learning_rate =0.0001 
#         self.sigma_alpha=0.2
#         self.forget_best =True 


#     def ask(self):
#         self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
#         self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])

#         if self.average_baseline:
#             epsilon = self.epsilon_full
#         else:
#       # first population is mu, then positive epsilon, then negative epsilon
#             epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])

#         solutions = self.mu.reshape(1, self.num_params) + epsilon
#         self.solutions = solutions
#         return solutions


#     def tell(self, reward_table_result):
    
#         assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

#         reward_table = np.array(reward_table_result)
    
#         # if self.rank_fitness:
#         #     reward_table = compute_centered_ranks(reward_table)
    
#         # if self.weight_decay > 0:
#         #     l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
#         #     reward_table += l2_decay

#         reward_offset = 1

#         # 
#         if self.average_baseline:
#           b = np.mean(reward_table)
#           reward_offset = 0
#         else:
#           b = reward_table[0] # baseline
          
#         reward = reward_table[reward_offset:]
#         idx = np.argsort(reward)[::-1]

#         best_reward = reward[idx[0]]
#         if (best_reward > b or self.average_baseline):
#           best_mu = self.mu + self.epsilon_full[idx[0]]
#           best_reward = reward[idx[0]]
#         else:
#           best_mu = self.mu
#           best_reward = b

#         self.curr_best_reward = best_reward
#         self.curr_best_mu = best_mu

#         if self.first_interation:
#           self.first_interation = False
#           self.best_reward = self.curr_best_reward
#           self.best_mu = best_mu
#         else:
#           if self.forget_best or (self.curr_best_reward > self.best_reward):
#             self.best_mu = best_mu
#             self.best_reward = self.curr_best_reward

#         #update 
#         stdev_reward = reward.std()
#         epsilon = self.epsilon
#         sigma = self.sigma
#         S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
#         reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
#         rS = reward_avg - b
#         delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

#         # move mean to the average of the best idx means
#         rT = (reward[:self.batch_size] - reward[self.batch_size:])
#         change_mu = self.learning_rate * np.dot(rT, epsilon)
#         self.mu += change_mu

#         #print(change_mu[0])

#         # adjust sigma according to the adaptive sigma calculation
#         change_sigma = self.sigma_alpha * delta_sigma
#         change_sigma = np.minimum(change_sigma, self.sigma)
#         change_sigma = np.maximum(change_sigma, - 0.5 * self.sigma)
#         self.sigma += change_sigma


#     def move(self,gradFlat):
#         self.mu=self.momentum*self.mu+gradFlat

#         #done 


        




    
#     def updateModel(self,model,momentum):
#         idx =0 
#         i =0 
#         model_shapes=self.model_shapes
    
#         for param in model.parameters():
#             delta = np.product(model_shapes[i])
#             block = momentum[idx:idx+delta]
#             block = np.reshape(block, model_shapes[i])
#             i += 1
#             idx += delta
#             block_data = t.from_numpy(block).float()
#             # if args.cuda:
#             #   block_data = block_data.cuda() # using cuda 
#             param.data -= block_data*self.lr 

#     def recoverModel(self,model,momentum):

#         idx =0 
#         i =0 
#         model_shapes=self.model_shapes
#         #
#         for param in model.parameters():
#             delta = np.product(model_shapes[i])
#             block = momentum[idx:idx+delta]
#             block = np.reshape(block, model_shapes[i])
#             i += 1
#             idx += delta
#             block_data = t.from_numpy(block).float() 
#             param.data=param.data+block_data*self.lr 

 
        
    
#     def step(self,net,data,target,base):
#         loss_min=None
#         lossnum=None
       
#         reward =np.zeros(self.popsize)
#         # keep elite and remove 
#         num_list=self.num_list


#         grad=[]
#         for p in net.parameters():
#             p = p.grad.data.cpu().numpy()
#             grad.append(p.flatten()) #done 

#         gradFlat=np.concatenate(grad)

        

#         solutions=self.ask() #minor variance 

#         val=0       
#         for num in num_list:
       
#             newH=self.momentum*solutions[num]+gradFlat #update moment
#             self.updateModel(net,newH)
#             output = net(data)
#             loss = F.nll_loss(output, target) 
#             reward[num] = -(loss.data[0])
            
#             if (loss_min is None) or (loss.data[0] < loss_min):
#                 loss_min=loss.data[0] # loss
#                 lossnum=num

#             self.recoverModel(net,newH)

#             solutions[num]=newH

#             # if loss.data[0]>base:
#             #     #print("kidding ---> <----")
#             #     if np.random.randn()>0.5:
#             #         reward[num]=-1*base
#             #     else:
#             #         solutions[num]=newH
#             # else:
#             #     #val+=1
#             #     solutions[num]=newH

#             # if loss.data[0]<base:
#             #     if np.random.randn()>0.9:
#             #         solutions[num]=newH
#             #     else:
#             #         reward[num]=-1*loss.data[0]
#             # else:
#             #     if np.random.randn()>0.1:
#             #         solutions[num]=newH

#             # solutions[num]=newH
#             # done 
#         #self.updateModel(net,solutions[lossnum])
#         self.tell(reward)
#         self.move(self.mu)


#         def mate(a,b):

#             c=copy.copy(a)
#             idx = np.where(np.random.rand((c.size)) > 0.5)
#             c[idx] = b[idx]
#             return c

        
       
#         _std=reward.std()
#         idx = np.argsort(reward)[::-1] 
#         best_reward = reward[idx[0]]
#         # self.sigma =self.momentum*gradFlat # for next step 



#         # print("{}{}".format(lossnum,idx[0]))
#         # best_mu = solutions[idx[0]]
#         # #
#         # # reward = compute_centered_ranks(reward)
#         # # l2_decay = compute_weight_decay(0.1, solutions)
#         # # reward += l2_decay

#         # OES version 
#         # epsilon=solutions-self.mu*self.momentum-gradFlat
#         # # if _std==0:
#         # # if _std==0:
#         # #     _std=0.000000000001 
#         # #     self.sigma*=0.9999
#         # #     # sigma*=0.99 #done 
#         # # normalized_reward = (reward - np.mean(reward))/ _std
#         # # deleta_Mu=0.001/(self.popsize*self.sigma)*np.dot(epsilon.T, normalized_reward)
#         # # self.mu =(self.momentum*self.mu)+gradFlat+deleta_Mu




       

#         # h=solutions[lossnum]*self.momentum+gradFlat
#         # self.updateModel(net,h)
#         # others=np.argsort(reward)[::-1][elite_popsize:]
#         # 
#         # # reward = compute_centered_ranks(reward)
#         # # l2_decay = compute_weight_decay(0.1, self.V)
#         # # reward += l2_decay
#         # # reward
#         # elite_popsize=3
#         # elite_idx=np.argsort(reward)[::-1][0:elite_popsize]  
#         # other_idx=np.argsort(reward)[::-1][elite_popsize:]
      
#         # elite_reward=reward[elite_idx] 

#         # simple GA version 

#         # popsize=8-elite_popsize
#         # choices=np.random.choice(num_list,2*popsize,list(reward))
#         # crossOver=[mate(solutions[choices[2*i]],solutions[choices[2*i+1]]) for i in range(popsize)]
#         # crossOver=np.array(crossOver)

#         # solutions[other_idx]=crossOver
#         # if _std==0:
#         #     solutions+=np.random.randn(self.popsize, self.num_params)*self.sigma
#         #     self.sigma*=0.999
#         #self.mu=np.mean(solutions,axis=0)






      
#         # # # #print(self.V[0].shape)
#         # self.V[others]=solutions #update the remaing ones 
#         # # if reward.std()>0.0001:
#         # self.sigma*=0.9999
#         # # if _std==0.0:
#         # #     self.lr*=0.99
#         # # else:
#         # #     self.sigma=0.0001
#         # # self.V=solutions
#         # # self.updateModel(net,np.mean(self.V,axis=0))
#         return best_reward,_std
        

        


