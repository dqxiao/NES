import pickle 

def pickle_write(data,method, fname):
    pickle_out=open(method+fname+".pickle","wb")
    pickle.dump(data,pickle_out)
    pickle_out.close()