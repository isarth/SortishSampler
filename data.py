import torch
from torch.utils.data import TensorDataset,Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.sampler import Sampler
import numpy as np

class text_dataset(Dataset):
  def __init__(self,x,y):
    super().__init__()
    self.x=x
    self.y=y
  def __len__(self):
    return len(self.x)
  def __getitem__(self,i):
    return (self.x[i],self.y[i])

class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))

class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.
    """
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

class DataLoader(object):
    def __init__(self,dataset, batch_size=1, shuffle=False, batch_sampler=None,
    	sampler=None,pad_idx=0, drop_last=False):
        self.dataset = dataset
        self.pad_idx, self.batch_size, self.shuffle = pad_idx, batch_size, shuffle
        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.dataset)
    
    def jag_stack(self, b): 
        if len(b)==1: return np.array(b[0][0]),np.array([b[0][1]])
        b = np.stack(b)
        y = [o[1] for o in b]
        ml = max(len(o[0]) for o in b)
        if min(len(o[0]) for o in b)==ml: return np.stack(b)
        res = np.ones((len(b), ml)) * self.pad_idx
        for i,o in enumerate(b):
            res[i,  :len(o[0])] = o[0]
        return res,y

    def get_batch(self, indices):
        xb,yb = self.jag_stack([self.dataset[i] for i in indices])

        return xb,yb
    
    def __iter__(self):
        for xb,yb in map(self.get_batch, iter(self.batch_sampler)):
            yield torch.Tensor(xb),torch.Tensor(yb)



if __name__=='__main__':
	X = [ [1,2,3,4],[5,6,7,8,1,9,6],[54,45,5,4,4,3,3,4,5,6,5],[5,4]] # variable examples
	y = [1,3,4,6] # labels
	X_lengths = [len(sentence) for sentence in X]
	bs=2
	train_ds = text_dataset(X,y)
	sampler = SortishSampler(X, key=lambda x: len(X[x]), bs=bs)
	train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler,pad_idx=0)

	for xb,yb in train_dl:
		print (xb,yb)
