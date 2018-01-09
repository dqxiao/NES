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



def testRuns(training_log, trainLog=True):
	best_valid_acc = 0
	start=time.time() # just for timing
	for epoch in range(1, args.epochs + 1):
	  # train loop
	  model.eval()
	  for batch_idx, (data, target) in enumerate(train_loader):
	    data, target = Variable(data), Variable(target)
	    solutions = es.ask() 
	    reward = np.zeros(es.popsize)

	    for i in range(es.popsize):
	      update_model(solutions[i], model, model_shapes)
	      output = model(data)
	      loss = F.nll_loss(output, target) # loss function
	      reward[i] = - loss.data[0]
	    best_raw_reward = reward.max()
	    reward = compute_centered_ranks(reward)
	    l2_decay = compute_weight_decay(weight_decay_coef, solutions)
	    reward += l2_decay
	    es.tell(reward)
	    result = es.result()
	    if (batch_idx % 50 == 0):
	    	print(epoch, batch_idx, best_raw_reward,result[1])	    
	  curr_solution = es.current_param()
	  update_model(curr_solution, model, model_shapes)

	  valid_acc,valid_loss = evaluate(model, valid_loader, print_mode=False)
	  #epoch 
	  if trainLog: 
	  	training_log.append([valid_acc,valid_loss]) 

	  print('valid_acc', valid_acc * 100.)
	  if valid_acc >= best_valid_acc:
	    best_valid_acc = valid_acc
	    best_model = copy.deepcopy(model)
	    print('best valid_acc', best_valid_acc * 100.)
	evaluate(best_model, test_loader, print_mode=True)


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

	training_log=[]
	dataFeed()
	model=Net()
	NPARAMS,model_shapes=cal_nparams(model)
	
	d=dict()


	es= createXNES()
	print("Debug xNES function")
	training_log=[] 
	testRuns(training_log,trainLog=True)
	if len(training_log)!=0:
		pickle_write(np.array(training_log),"xNES-v2","-NN")

	






