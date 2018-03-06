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
# from model.vgg import * 
from model.NetworkModel import * 
from ES import *    # the creator function for NES methods 
from fileIO import * 
import os 
import sys 
from SGD_ES import * 

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

#done 


def train(epoch,printTrain=True):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
	
		data, target= Variable(data), Variable(target)
		output = model(data)
		model.zero_grad()
		loss = F.nll_loss(output, target) 
		loss.backward()
		best_reward,reward_std=optimizer.step(model,data,target,loss.data[0])
		if batch_idx % 100 ==0:
			print('{}\t{}\t{}'.format(loss.data[0],-1*best_reward,reward_std)) 

		# if batch_idx % 1000 == 0 and printTrain:
		# 	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		# 		epoch, batch_idx * len(data), len(train_loader.dataset),
		# 		100. * batch_idx / len(train_loader), loss.data[0]))
	#done 



if __name__=="__main__":

	# configRun()
	global model
	global es
	# global muteRate
	global optimizer
	global args

	
	

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
	parser.add_argument('--momentum',default=0.99,type=float,help='momentum')
	# parser.add_argument()
	
	args = parser.parse_args()
	mnistFeed()


	model = MLPNet()
	# NPARAMS,model_shapes=cal_nparams(model)
	momentum =args.momentum
	#optimizer = SGDS_ES(model,momentum=momentum)
	# optimizer= SGDmomentum(model,momentum=momentum)
	optimizer = SGD_GA(model,momentum=momentum)


	#

	training_log=[]
	for epoch in range(1,100):
		train(epoch,printTrain=True)
		test_acc,test_loss=evaluate(model, test_loader, print_mode=False)
		valid_acc,valid_loss=evaluate(model, valid_loader,print_mode=False)
		training_log.append([valid_acc,valid_loss,test_acc,test_loss])
		print("{}\t{}\t{}".format(epoch,valid_loss,test_loss))

	fname = "{}-{}".format("SGDM-oes",momentum)
	folder ="{}/{}/".format("MNIST","MLP") 
	pickle_write(np.array(training_log),folder+fname,"-NN")

	