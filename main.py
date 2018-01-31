import numpy as np
import math
import multiprocessing as mp
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


from collections import namedtuple
import copy

# import sys
import argparse

#model setting up 
from model.vgg import * 
from model.NetworkModel import * 
from ES import *    # the creator function for NES methods 
from fileIO import * 
import os 




def testRuns(training_log, trainLog=True,rewardShaping=False):
	best_valid_acc = 0
	# start=time.time() # just for timing
	for epoch in range(1, args.epochs + 1):
	# train loop
		model.eval()
		resultLogs=np.zeros(5)
		for batch_idx, (data, target) in enumerate(train_loader):
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			#done 

			solutions = es.ask() 
			reward = np.zeros(es.popsize)

			for i in range(es.popsize):
				update_model(solutions[i], model, model_shapes)
				output = model(data)
				loss = F.nll_loss(output, target) # loss function
				reward[i] = - loss.data[0]
			best_raw_reward = reward.max()
			if rewardShaping:
				reward = compute_centered_ranks(reward)
				l2_decay = compute_weight_decay(weight_decay_coef, solutions)
				reward += l2_decay
			es.tell(reward)
			result = es.result()
			tempLog=np.array([abs(result[1]),abs(reward.mean()),calEntropy(result[3]),abs(reward.std()),result[-1]])
			resultLogs+=tempLog

			if (batch_idx % 50 == 0):
				print(epoch, batch_idx, best_raw_reward,result[1],result[-1])	    
			curr_solution = es.current_param()
			update_model(curr_solution, model, model_shapes)

		valid_acc,valid_loss = evaluate(model,valid_loader, print_mode=False)
		test_acc, test_loss =evaluate(model,test_loader,print_mode=False)

		if trainLog: 
			resultLogs/=batch_idx
			training_log.append([valid_acc,valid_loss,test_acc,test_loss]+list(resultLogs))

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

	global weight_decay_coef 


	torch.manual_seed(0)
	np.random.seed(0)
	
	weight_decay_coef = 0.1

	# Args = namedtuple('Args', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'cuda', 'seed', 'log_interval'])
	# args = Args(batch_size=100, test_batch_size=1000, epochs=50, lr=0.001, cuda=True, seed=0, log_interval=10)

def cifar10Feed():
	global train_loader
	global valid_loader
	global test_loader
	
	transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	valid_loader = train_loader


	testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	return "CIFAR10"

def mnistFeed():
	global train_loader
	global valid_loader
	global test_loader

	kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

	train_loader = torch.utils.data.DataLoader(datasets.MNIST('MNIST_data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
  			batch_size=args.batch_size, shuffle=True, **kwargs)

	valid_loader = train_loader

	test_loader = torch.utils.data.DataLoader(
  		datasets.MNIST('MNIST_data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
  			batch_size=args.batch_size, shuffle=True, **kwargs)

	return "MNIST"

if __name__=="__main__":

	
	
	global model
	global es
	global diversity_base 
	global NPARAMS
	global NPOPULATION
	global args 
	global model_shapes

	configRun()

	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--model',default="VGG16",help='DNN model')
	parser.add_argument('--optimizer',default='PEPG',help='SGD or ES methods used')
	parser.add_argument('--popsize',default=0, type=int, help='ES popsize')
	parser.add_argument('--opt',default=1, type=int, help='ES sigma option')
	parser.add_argument('--diversity_base',default=0.0, type=float, help='diversity up bounded')
	parser.add_argument('--cuda',default=False,type=bool,help='use cuda or not')
	parser.add_argument('--epochs',default=100, type=int, help='the number of iteration')
	parser.add_argument('--batch_size',default=100,type=int,help='batch size')
	# parser.add_argument()
	
	args = parser.parse_args()

	datasetName=mnistFeed()
	
	
	

	model=MLPNet()

	if args.cuda:
		model.cuda()
	NPARAMS,model_shapes=cal_nparams(model)




	if args.popsize==0:	
		NPOPULATION = int(4+3*np.ceil(np.log(NPARAMS)))
		NPOPULATION = int(NPOPULATION/2)*2+1
	else:
		NPOPULATION = args.popsize

	ea=ESArgs(NPARAMS=NPARAMS, NPOPULATION=NPOPULATION,diversity_base=args.diversity_base, opt=args.opt,lr=args.lr) 
	esCreate={
		"XNESVar": createXNESVar(ea),
		"XNESSA": createXNESSA(ea),
		"PEPG": createPEPG(ea),
		"PEPGVar": createPEPGVar(ea)
	}

	es=esCreate[args.optimizer]



	print("Debug {} function".format(es.name()))
	print("with popsize:{}, db:{}, Opt:{}".format(NPOPULATION, args.diversity_base,args.opt))

	training_log=[] 
	if args.popsize==0:
		fname = "{}-{}-{}".format(args.optimizer,args.opt,args.diversity_base)
	else:
		fname = "BP-{}-{}-{}".format(args.optimizer,args.opt,args.diversity_base)
	folder ="{}/{}/".format(datasetName,model.name())

	testRuns(training_log)
	if not os.path.exists("./"+folder):
		os.makedirs(folder)
	pickle_write(np.array(training_log),folder+fname,"")
	






