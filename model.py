#simple SNN
#V[t]=f(V[t-1],X[t])
#H[t]=delta(V[t]-V_threshold)

import torch 
import torch.nn as nn
import torch.autograd as autograd

class DeltaUnitSubGraident(autograd.Function):
    @staticmethod
    def forward(ctx, mem, k=2.0):
        ctx.save_for_backward(mem)
        ctx._k=k
        return (mem>0).float()
    
    @staticmethod
    def backward(ctx, grad):
        #atan k/(2*(1+(pi/2 k x)^2)))))
        _mem, = ctx.saved_tensors
        _k = ctx._k
        _grad=grad*(_k/(2*(1+(_k*_mem/3.1415/2).pow(2))))
        _grad=_grad.clamp(min=-0.1,max=0.1)
        return _grad,None
    
class LIFNeuronActivation(nn.Module):
    def __init__(self,threshold=1,
                    v_reset=0,
                    tau=2,
                    beta=1,
                    funct=DeltaUnitSubGraident(),
                    soft_reset=True):
        super(LIFNeuronActivation,self).__init__()
        self.threshold=threshold
        self.v_reset=v_reset
        self.tau=tau
        self.beta=beta
        self.funct=funct
        self.soft_reset=soft_reset
        self.Existed_v=None
        
    def _init_v(self,inp):
        self.Existed_v=inp.detach().clone()*0
        
    def reset(self):
        self.Existed_v=None
        
    def forward(self,inp):
        #inp [b,x1,x2...]
        if self.Existed_v is None:
            self._init_v(inp)
        else:
            assert inp.shape==self.Existed_v.shape
            
        h=self.Existed_v*self.beta+1/self.tau*(inp+self.Existed_v)
        spk=self.funct.apply(h-self.threshold)
        if self.soft_reset:
            v_init=h-self.threshold*spk
        else:
            v_init=h*(1-spk)+self.v_reset*spk
        self.Existed_v=v_init
        return spk
    
    @property
    def v(self):
        return self.Existed_v
    

class MNISTClassifierSNN(nn.Module):
    def __init__(self,example,batch_first=False):
        super(MNISTClassifierSNN,self).__init__()
        self.batch_first=batch_first
        self.layers1=[
            nn.Conv2d(1,32,3),
            nn.BatchNorm2d(32),
            LIFNeuronActivation(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,3),
            nn.BatchNorm2d(16),
            LIFNeuronActivation(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #LIFNeuronActivation(),
        ]
        self.layers1=nn.Sequential(*self.layers1)
        with torch.no_grad():
            def _auto_infer(input_example,layers):
                return layers.forward(input_example).shape[1]
        if not self.batch_first:
            example=example.transpose(0,1).contiguous()
        shape=_auto_infer(example[0],self.layers1)
        self.layers2=[
            nn.Linear(shape,128),
            LIFNeuronActivation(),
            nn.Linear(128,10),
            LIFNeuronActivation(),
        ]
        self.layers2=nn.Sequential(*self.layers2)
    def forward(self,x):
        for layers in self.layers1:
            if isinstance(layers,LIFNeuronActivation):
                layers.reset()
        for layers in self.layers2:
            if isinstance(layers,LIFNeuronActivation):
                layers.reset()
        if self.batch_first:
            x=x.transpose(0,1).contiguous()
        temp=[]
        for t in range(x.size(0)):
            temp.append(
                self.layers2(
                    self.layers1(x[t])
                )
            )
        x=torch.stack(temp,dim=0)
        if self.batch_first:
            x=x.transpose(0,1).contiguous()
        #x=nn.functional.normalize(x,dim=1,p=1)
        return x
    
class MNISTClassifierANN(nn.Module):
    def __init__(self,example,batch_first=False):
        super(MNISTClassifierANN,self).__init__()
        self.batch_first=batch_first
        self.layers1=[
            nn.Conv2d(1,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,3),
            nn.ReLU(),
            LIFNeuronActivation(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            #LIFNeuronActivation(),
        ]
        self.layers1=nn.Sequential(*self.layers1)
        with torch.no_grad():
            def _auto_infer(input_example,layers):
                return layers.forward(input_example).shape[1]
        if not self.batch_first:
            example=example.transpose(0,1).contiguous()
        shape=_auto_infer(example[0],self.layers1)
        self.layers2=[
            nn.Linear(shape,128),
            nn.ReLU(),
            nn.Linear(128,10),
            nn.Softmax(dim=1),
        ]
        self.layers2=nn.Sequential(*self.layers2)
    def forward(self,x):
        for layers in self.layers1:
            if isinstance(layers,LIFNeuronActivation):
                layers.reset()
        for layers in self.layers2:
            if isinstance(layers,LIFNeuronActivation):
                layers.reset()
        if self.batch_first:
            x=x.transpose(0,1).contiguous()
        temp=[]
        for t in range(x.size(0)):
            temp.append(
                self.layers2(
                    self.layers1(x[t])
                )
            )
        x=torch.stack(temp,dim=0)
        if self.batch_first:
            x=x.transpose(0,1).contiguous()
        #x=nn.functional.normalize(x,dim=1,p=1)
        return x