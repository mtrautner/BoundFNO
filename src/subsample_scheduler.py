import torch

class SubsampleScheduler:
    '''
    Training with adaptive subsampling.
    '''
    def __init__(self, ss_max, ndim=2, threshold=0.9, patience=10):
        self.ss_max = ss_max # maximum subsampling
        self.ndim = ndim
        self.threshold = threshold
        self.patience = patience
        self.num_bad_epochs = None
        self.last_epoch = 0
        self.best = inf
        
    def __call__(self,x,y):
        ss = self.ss
        if self.ndim==1:
            return x[...,::ss], y[...,::ss]
        elif self.ndim==2:
            return x[...,::ss,::ss],y[...,::ss,::ss]
        else:
            raise NotImplementedError('Subsampling scheduler for ndim={self.ndim} not implemented.')

    def step(self,loss):
        current = loss
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            # reduce the subsampling rate
            self.ss = max(1, self.ss//2)
            self.num_bad_epochs = 0

    def is_better(self, val, best):
        return val < self.threshold * best
