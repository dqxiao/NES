import numpy as np 
from sklearn.neighbors import NearestNeighbors
import torch 

def torch_compute_ranks(x):

  y,idx=torch.sort(x,0)
  idx =idx.type(torch.LongTensor)
  size = x.size()[0]
  ranks = torch.zeros(x.size())
  ranks[idx] = torch.range(0,size-1)
  return ranks 


def torch_compute_centered_ranks(x,cudaFlag):
  y=torch_compute_ranks(x).float()
  if cudaFlag:
    y=y.cuda()
  size = x.size()[0]
  y /=size-1
  y -=0.5 

  return y 

def reverse(tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]

    idx = torch.LongTensor(idx)
    if "cuda" in tensor.type():
      idx = idx.cuda()
    inverted_tensor = tensor.index_select(0, idx)
    return inverted_tensor
    

def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)



def compute_novelty_obs(obs,K=10):
  """
  Compute KNN distance function 
  """
  nbrs=NearestNeighbors(n_neighbors=K, algorithm='auto',metric='cosine').fit(obs) #
  distances,_=nbrs.kneighbors(obs)
  spareness = np.sum(distances,axis=1)

  #print(spareness)

  return spareness 

class SlideWindow(object):
    """store the recent information for need """
    def __init__(self,capacity):
        self.vals = [] 
        self.capacity = capacity
        self.index = 0 

    def update(self,val):
        if self.index==len(self.vals) and self.index<self.capacity:
            self.vals.append(val)  
        else:
            # more than capacity 
            self.index = self.index % self.capacity
            self.vals[self.index]=val 
        self.index +=1

    def mean(self):
        return np.array(self.vals).mean()
    def std(self):
        return np.array(self.vals).std()


    def lastDiff(self):
        
        c=self.index-1
        if c==0:
            l=self.capacity-1 
        else:
            l=c-1 

        return self.vals[c]-self.vals[l]

    def evident(self):
        l=len(self.vals)
        return l>3 

def calEntropy(x):
    _x=np.log(x)
    return _x.sum()
    