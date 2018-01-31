import numpy as np
import math
import multiprocessing as mp
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


from collections import namedtuple
from PIL import Image
import time 
import sys




def testRuns(training_log, trainLog=True):
	best_valid_acc = 0
	start=time.time() # just for timing
	for epoch in range(1, args.epochs + 1):
	  # train loop
	  model.eval()
	  resultLogs=np.zeros(5)
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

	    # reward = compute_centered_ranks(reward)
	    # l2_decay = compute_weight_decay(weight_decay_coef, solutions)

	    # reward += l2_decay
	    es.tell(reward)   #want to maximize the value 
	    result = es.result()
	    
	    if (batch_idx % 50 == 0):
	    	print(epoch, batch_idx, best_raw_reward,result[1],result[-1])	 
	    	# print(epoch, batch_idx, result[-1])
	    	temp=[abs(result[1]),abs(reward.mean()),calEntropy(result[3]),abs(reward.std()),result[-1]] 
	    	temp=np.array(temp)
	    	resultLogs+=temp 
	    	# resultLogs=[abs(result[1]),abs(reward.mean()),calEntropy(result[3]),abs(reward.std()),result[-1]] 
	   # resultlog= 
	  curr_solution = es.current_param()
	  update_model(curr_solution, model, model_shapes)
	  resultLogs/=batch_idx/50

	  valid_acc,valid_loss = evaluate(model, valid_loader, print_mode=True)
	  test_acc, test_loss = evaluate(model,test_loader,print_mode=True)
	  #epoch 

	  if trainLog: 
	  	training_log.append([valid_acc,valid_loss,test_acc,test_loss]+list(resultLogs)) 

	  print('valid_acc', valid_acc * 100.)
	  print('diversity', resultLogs[-1])
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
	
	weight_decay_coef = 0.1

	Args = namedtuple('Args', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'cuda', 'seed', 'log_interval'])
	args = Args(batch_size=100, test_batch_size=1000, epochs=100, lr=0.001, cuda=False, seed=0, log_interval=10)



def cifar10Feed():

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

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	valid_loader = train_loader


	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	# done 

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
  			batch_size=args.batch_size, shuffle=True, **kwargs)


if __name__=="__main__":

	configRun()
	global model
	global es
	global diversity_base
	global opt
	diversity_base= float(sys.argv[1])
	if len(sys.argv)>2:
		opt=int(sys.argv[2])


	


	training_log=[]
	dataFeed()
	model =MLPNet() 
	NPARAMS,model_shapes=cal_nparams(model)
	
	# NPOPULATION=int(4+3*np.ceil(np.log(NPARAMS)))
	# NPOPULATION= int(NPOPULATION/2)*2+1
	NPOPULATION = 101 
	print(model.name())
	print("popsize:{}".format(NPOPULATION))

	
	esFunc=dict()
	esFunc['Var']=createXNESVar()
	esFunc['SA']=createXNESSA()
	esFunc['PEPG']=createPEPG()

	# if sys.argv[3]=='Var':
	# 	es = createXNESVar()
	# else:
	# 	es = createXNESSA()

	print("Debug {} function".format(es.name()))
	training_log=[] 
	testRuns(training_log,trainLog=True)
	fname = "BP-{}-{}".format(es.name(),diversity_base)
	folder ="{}/{}/".format("MNIST","MLP") 
	pickle_write(np.array(training_log),folder+fname,"-NN")

	






