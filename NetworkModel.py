import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable


# using the simplest neural network to perform our experiments agian 
# class SimpleNet(nn.Module): 

#   def __init__(self):
#     super(SimpleNet,self)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.num_filter1 = 8 
    self.num_filter2 = 16
    self.num_padding = 2
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, self.num_filter1, 5, padding=self.num_padding)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(self.num_filter1, self.num_filter2, 5, padding=self.num_padding)
    # feature map size is 7*7 by pooling
    self.fc = nn.Linear(self.num_filter2*7*7, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_filter2*7*7)   # reshape Variable
    x = self.fc(x)
    return F.log_softmax(x)
    # return x 


def cal_nparams(model):
    orig_params=[] 
    model_shapes=[]

    for param in model.parameters():
        p = param.data.cpu().numpy()
        model_shapes.append(p.shape)
        orig_params.append(p.flatten())
    orig_params_flat = np.concatenate(orig_params)
    NPARAMS = len(orig_params_flat)

    return NPARAMS, model_shapes



def update_model(flat_param, model, model_shapes):
  idx = 0
  i = 0
  for param in model.parameters():
    delta = np.product(model_shapes[i])
    block = flat_param[idx:idx+delta]
    block = np.reshape(block, model_shapes[i])
    i += 1
    idx += delta
    block_data = torch.from_numpy(block).float()
    param.data = block_data


def evaluate_anchor(model, test_loader,return_log=True):

  model.eval()

  ob=[]
  for data, target in test_loader:
    output = model(Variable(data))      # get its output 
    _,pred= torch.max(output.data,1)    # get its prediction 
    ob=np.concatenate(Variable(pred==target).data.numpy())
    break 

  ob=np.array(ob)
  #print(len(ob))
  return ob 
    




def evaluate(model, test_loader, print_mode=True, return_loss=False):
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:

    output = model(Variable(data))
    # output = F.log_softmax(output)
    test_loss += F.nll_loss(output, Variable(target), size_average=False).data[0] # sum up batch loss
    _,pred=torch.max(output.data,1)
    correct += (pred==target).sum()


  test_loss /= len(test_loader.dataset)
  acc = correct / len(test_loader.dataset)
  
  if print_mode:
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * acc))

  # if return_loss:
  #   return test_loss
  return acc,test_loss