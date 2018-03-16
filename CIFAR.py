import math
import multiprocessing as mp
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


from collections import namedtuple
import copy
import torch.optim as optim

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
	reward_min = None 
	for epoch in range(1, args.epochs + 1):
	# train loop
		model.eval()
		# resultLogs=np.zeros(5)
		running_loss =0.0 
		for batch_idx, (data, target) in enumerate(train_loader):
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			#done 

			solutions = es.ask() 
			# reward = np.zeros(es.popsize)
			reward = torch.zeros(es.popsize)
			if args.cuda:
				reward=reward.cuda()

			pop_loss =0.0 
			for i in range(es.popsize):
				update_model(solutions[i], model, model_shapes)
				output = model(data)            
				loss = F.nll_loss(output, target) # loss function
# 				loss,acc= loss_acc_evaluate(output,target) 
# 				reward[i] = acc             # get the value 
				reward[i] = -loss.data[0]             # get the value 
				pop_loss += loss.data[0]
			best_raw_reward = reward.max()
			pop_loss/=es.popsize
			running_loss+= pop_loss
			# if True:
			# 	reward = compute_centered_ranks(reward)
			# 	l2_decay = compute_weight_decay(weight_decay_coef, solutions)
			# 	reward += l2_decay
			es.tell(reward)
			result = es.result()
		# 	#tempLog=np.array([abs(result[1]),abs(reward.mean()),calEntropy(result[3]),abs(reward.std()),result[-1]])
		# 	#resultLogs+=tempLog

			if (batch_idx % 100 == 0):
				print(epoch, batch_idx,best_raw_reward,es.diversity_base,result[-1])	    
			curr_solution = es.current_param()
			update_model(curr_solution, model, model_shapes)
		running_loss/=batch_idx
		print("{}\{}".format(epoch,running_loss))
		if args.autotuning:
# 			print("{}\{}".format(epoch,running_loss))
			db=math.pow(running_loss/2.33,0.5)*args.diversity_base
			mdb=db/running_loss 
			es.set_diversity_base(db) 
			es.set_mu_diversity_base(-1*0.0002*mdb) # done 
		test_acc,test_loss=evaluate(model, test_loader, print_mode=True,cuda=args.cuda)       
		if trainLog:
			training_log.append([running_loss,test_acc,test_loss])
	# 	valid_acc,valid_loss = evaluate(model,valid_loader, print_mode=False,cuda=args.cuda)
	# 	test_acc, test_loss =evaluate(model,test_loader,print_mode=False,cuda=args.cuda)

	# 	if trainLog: 
	# 		# resultLogs/=batch_idx  
	# 		training_log.append([valid_acc,valid_loss,test_acc,test_loss])

	# 	print('valid_acc', valid_acc * 100.)
	# 	if valid_acc >= best_valid_acc:
	# 		best_valid_acc = valid_acc
	# 		best_model = copy.deepcopy(model)
	# 		print('best valid_acc', best_valid_acc * 100.)


def configRun():
	"""
	just for debugging 
	"""

	global weight_decay_coef 


	torch.manual_seed(0)
	#np.random.seed(0)
	
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
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

	valid_loader = train_loader


	testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
	parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
	parser.add_argument('--model',default="Net",help='DNN model')
	parser.add_argument('--optimizer',default='PEPG',help='SGD or ES methods used')
	parser.add_argument('--popsize',default=0, type=int, help='ES popsize')
	parser.add_argument('--opt',default=1, type=int, help='ES sigma option')
	parser.add_argument('--diversity_base',default=0.0, type=float, help='diversity up bounded')
	parser.add_argument('--cuda',default=False,type=bool,help='use cuda or not')
	parser.add_argument('--epochs',default=500, type=int, help='the number of iteration')
	parser.add_argument('--batch_size',default=100,type=int,help='batch size')
	parser.add_argument('--autotuning',default=False,type=bool,help='auto tune hyperparameter')   
	parser.add_argument('--sigma_init',default=0.1,type=float,help='sigma init for es')     
	# parser.add_argument()
	
	args = parser.parse_args()

	datasetName=cifar10Feed()

	
	

	# model=VGG(args.model) 
	if args.model=="Net":
		model = Net()# 
# 		model = CNNNet()#
	else:
		model = VGG(args.model)
	if args.cuda:
		torch.cuda.manual_seed(0)
		model.cuda()
	NPARAMS,model_shapes=cal_nparams(model)

        
# 	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
	if args.cuda:
		torch.cuda.manual_seed(0)
		model.cuda()


	if args.popsize==0:	
		NPOPULATION = int(4+3*np.ceil(np.log(NPARAMS)))
		NPOPULATION = int(NPOPULATION/2)*2+1
	else:
		NPOPULATION = args.popsize

	ea=ESArgs(NPARAMS=NPARAMS, NPOPULATION=NPOPULATION,diversity_base=args.diversity_base, opt=args.opt,lr=args.lr,sigma_init=args.sigma_init) 
	if args.cuda:
		# es = createPEPGCuda(ea)
		esCreate={
			"PEPGVar": createPEPGVarCuda(ea),
			"PEPG": createPEPGCuda(ea)
		}
		es= esCreate[args.optimizer]

	else:
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
	lrtag="" 
	if args.lr!=0.01:
		lrtag="slr{}".format(args.lr)

	if args.popsize==0:
		fname = "{}{}-{}-{}".format(lrtag,args.optimizer,args.opt,args.diversity_base)
	else:
		fname = "{}BP-{}-{}-{}".format(lrtag,args.optimizer,args.opt,args.diversity_base)
	folder ="{}/{}/".format(datasetName,model.name())

	testRuns(training_log)
	if not os.path.exists("./"+folder):
		os.makedirs(folder)
	atTag=""
	if args.autotuning:
		atTag="mat"        
	pickle_write(np.array(training_log),folder+fname,atTag)
                  
