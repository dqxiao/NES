import numpy as np
import math
import multiprocessing as mp
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


from collections import namedtuple
from PIL import Image
import os
import os.path
import errno
import codecs
import copy

from SimpleGA import * 
from NoveltyGA import *
from PEPG import *
from NetworkModel import * 
from ESUtil import * 
from XNES import *
import pickle 

import time 
from RewardEvaluator import *


def pickle_write(data,method, fname):
    pickle_out=open(method+fname+".pickle","wb")
    pickle.dump(data,pickle_out)
    pickle_out.close()


def createNGA():

	es= NoveltyGA(NPARAMS, 
		popsize=NPOPULATION,
		forget_best=False,
		forget_history=False
		)

	return es 

def createGA():
	es = SimpleGA(NPARAMS,
              popsize=NPOPULATION,
              forget_best=False,
              sigma_init=0.01,
              sigma_decay=0.9999,
              sigma_limit=0.01
             )
	return es


def createPEPG():
	es = PEPG(    NPARAMS,
              popsize=NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=0.1,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False
             )
	return es 


def createXNES():
	es = XNES(NPARAMS,
              popsize=NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=0.1,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False,
             )
	return es 


def noiseIn(target, muteRate):
    if np.random.randn(1)<muteRate:
        target[target==8]=11
        target[target==3]=12
        target[target==11]=3 
        target[target==12]=8 


def trainES(epoch, printTrain=True):
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        if epoch>10:
            noiseIn(target,muteRate)
        data, target = Variable(data), Variable(target)
        solutions = es.ask() 
        reward = np.zeros(es.popsize)
        
        for i in range(es.popsize):
            update_model(solutions[i], model, model_shapes)
            output=model(data)
            loss = F.nll_loss(output, target)
            reward[i] = - loss.data[0]
        best_raw_reward = reward.max()
        reward = compute_centered_ranks(reward) # already consider the 
        l2_decay = compute_weight_decay(weight_decay_coef, solutions)
        reward += l2_decay

        es.tell(reward)

        result = es.result()
        if (batch_idx %50== 0):
          print(epoch, batch_idx, best_raw_reward, result[0].mean())
    curr_solution = es.current_param()
    update_model(curr_solution, model, model_shapes)
    # done 


def configRun():
	"""
	just for debugging 
	"""
	global NPARAMS
	global NPOPULATION
	global args 
	global model_shapes
	global weight_decay_coef 


	torch.manual_seed(0)
	np.random.seed(0)
	NPOPULATION=101  
	weight_decay_coef = 0.1

	Args = namedtuple('Args', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'cuda', 'seed', 'log_interval'])
	args = Args(batch_size=100, test_batch_size=1000, epochs=50, lr=0.001, cuda=False, seed=0, log_interval=10)

def dataFeed():
	global train_loader
	global valid_loader
	global test_loader

	kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

	train_loader = torch.utils.data.DataLoader(datasets.MNIST('MNIST_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
  			batch_size=args.batch_size, shuffle=True, **kwargs)

	valid_loader = train_loader

	test_loader = torch.utils.data.DataLoader(
  		datasets.MNIST('MNIST_data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
  			batch_size=args.batch_size, shuffle=False, **kwargs)


if __name__=="__main__":

	configRun()
	global model
	global es
	global muteRate

	training_log=[]
	dataFeed()
	model=Net()
	NPARAMS,model_shapes=cal_nparams(model)

	d=dict()




	# xneslog=dict()

	# for noise in [0.1,0.3,0.5,0.7]:
	# 	muteRate=noise
	# 	testLog=[]
	# 	es = XNES(NPARAMS,
	# 	          popsize=NPOPULATION,
	# 	          sigma_init=0.01,
	# 	          sigma_decay=0.999,
	# 	          sigma_alpha=0.2,
	# 	          sigma_limit=0.01,
	# 	          learning_rate=0.01,            # learning rate for standard deviation
	# 	          learning_rate_decay = 0.9999, # annealing the learning rate
	# 	          learning_rate_limit = 0.01,   # stop annealing learning rate
	# 	          average_baseline=False,
	# 	         )
	# 	for epoch in range(1,100):
	# 		trainES(epoch,printTrain=True)
	# 		test_acc,test_loss=evaluate(model, test_loader, print_mode=True)
	# 		valid_acc,valid_loss=evaluate(model, valid_loader,print_mode=True)
	# 		testLog.append([test_acc,valid_acc,test_loss,valid_loss])
	# 	xneslog[noise]=copy.copy(testLog)  
	# pickle_write(xneslog,"xNES-noise-adaptive-dict-OG","-NN")

	






