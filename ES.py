from OpenES import * 
from PEPG import * 
from XNESVar import * 
from XNESSA import  * 
from XNES import * 
from SimpleGA import * 
from NoveltyGA import * 
from collections import namedtuple
from PEPGVar import * 
from PEPGTorch import * 

ESArgs = namedtuple('esArgs', ['NPARAMS', 'NPOPULATION','diversity_base','opt','lr'])



def createNGA(ea):
	es= NoveltyGA(ea.NPARAMS, 
		popsize=ea.NPOPULATION,
		forget_best=False,
		forget_history=False
		)

	return es 

def createGA(ea):
	es = SimpleGA(ea.NPARAMS,
              popsize=ea.NPOPULATION,
              forget_best=False,
              sigma_init=0.01,
              sigma_decay=0.9999,
              sigma_limit=0.01
             )
	return es


def createPEPG(ea):
	es = PEPGTorch(ea.NPARAMS,
              popsize=ea.NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=ea.lr,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False
             )
	return es 


def createXNES(ea):
	es = XNES(ea.NPARAMS,
              popsize=ea.NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=ea.lr,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False,
             )
	return es 


def createXNESVar(ea):
  es = XNESVar(ea.NPARAMS,
              popsize=ea.NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=ea.lr,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False,
              diversity_base= ea.diversity_base, 
              option=ea.opt
             )
  return es 


def createXNESSA(ea):
  es = XNESSA(ea.NPARAMS,
            popsize=ea.NPOPULATION,
            sigma_init=0.01,
            sigma_decay=0.999,
            sigma_alpha=0.2,
            sigma_limit=0.01,
            learning_rate=ea.lr,            # learning rate for standard deviation
            learning_rate_decay = 0.9999, # annealing the learning rate
            learning_rate_limit = 0.01,   # stop annealing learning rate
            average_baseline=False,
            diversity_base= ea.diversity_base, 
            option=ea.opt
           )
  return es 


def createPEPGVar(ea):
  es = PEPGVar(ea.NPARAMS,
              popsize=ea.NPOPULATION,
              sigma_init=0.01,
              sigma_decay=0.999,
              sigma_alpha=0.2,
              sigma_limit=0.01,
              learning_rate=ea.lr,            # learning rate for standard deviation
              learning_rate_decay = 0.9999, # annealing the learning rate
              learning_rate_limit = 0.01,   # stop annealing learning rate
              average_baseline=False,
              diversity_base =ea.diversity_base,
              option = ea.opt
             )
  return es 