import numpy as np 
from sklearn.neighbors import NearestNeighbors






# class RewardEvaluator:

# 	def __init__(self,popsize, 
# 				  novelty_based=False, 
# 				  nearestN =4
# 				):

# 		self.popsize=popsize
# 		self.raw_reward= np.zeros(self.popsize) # popsize is fixed 
# 		self.popObs = np.zeros((self.popsize,))  # done 
# 		self.novelty_based = novelty_based
# 		self.reward = np.zeros(self.popsize)  # popsize is fixed 
# 		self.nearestN = nearestN



# 	def tell(self, reward_table):

# 		assert(len(reward_table)==self.popsize),"Inconsistent issue "
# 		self.raw_reward=reward_table



# 	def ask(self):
	
# 		if self.novelty_based:
# 			return self.reward

# 		return self.raw_reward 

# 	def tellMore(self, popObs):
# 		#novelty-based only happens over the population 
# 		#let's cal its novelty 
# 		assert(len(popObs)==self.popsize),"Inconsistent issue of population issue"

# 		

# 		self.reward=spareness

# 		# done 





