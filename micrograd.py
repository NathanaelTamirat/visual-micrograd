import math 
from utils import RNG,gen_data_yinyang,draw_dot,vis_color
random=RNG(50)

class Value: #store a single scalar value and the gradient (similar to torch tensor)
    def __init__(self,data,_prev=(),_op=''):
        self.data=data
        self.grad=0
        #used for autograd graph construction
        self._backward=lambda: None
        self._prev=_prev
        self._op=_op
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out= Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    def __mul__(self,other):
        other= other if isinstance(other,Value) else Value(other)
        out=Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data * out.grad
            other.grad+=self.data * out.grad
        out._backward=_backward
        return out  
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only support int/float powers"
        out=Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad+=(other*self.data**(other-1))*out.grad
        out._backward=_backward
        return out   
    def relu(self):
        out=Value(0 if self.data <0 else self.data,(self,),"ReLU")
        def _backward():
            self.grad+=(1-out.data**2) * out.grad
        out._backward=_backward
        return out   
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out=Value(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    def log(self):
        out=Value(math.log(self.data),(self,),'log')
        def _backward():
            self.grad+=(1/self.data) * out.grad
        out._backward=_backward
        return out
    def backward(self): #topological order of allthe childern in the graph
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1  #go one variable at a time and apply chain rule 
        for v in reversed(topo):
            v._backward()
    def __neg__(self): # -self
        return self * -1.0
    def __radd__(self,other): # other +self
        return self + other
    def __sub__(self,other):# self - other
        return self+(-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self, other): # other * self
        return self * other
    def __truediv__(self, other): # self / other
        return self * other**-1
    def __rtruediv__(self, other): # other / self
        return other * self**-1
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

##### MLP #####
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0
    def parameters(self):
        return []
class Neuron(Module):
    def __init__(self,nin,nonlin=True):
        self.w=[Value(random.uniform(-1,1)* nin**-0.5) for _ in range(nin)]
        self.b=Value(0)
        self.nonlin=nonlin
        # color the neuron params light green (only used in graphviz visualization)
        vis_color([self.b]+self.w,"lightgreen")
    def __call__(self,x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return act.tanh() if self.nonlin else act
    def parameters(self):
        return self.w+[self.b]
    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron{len(self.w)}"
class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
class MLP(Module):
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))] 
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters() ]
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

#### loss funtion ####
def cross_entropy(logits,target):
    ex=[x.exp() for x in logits]
    denom=sum(ex)
    probs=[x/denom for x in ex] # normalize
    logp=(probs[target]).log() #log prob
    nll=-logp #negative log likelyhood
    return nll

#### optimizer ####
class AdamW:
    def __init__(self,parameters,lr=1e-1,betas=(0.9,0.95),eps=1e-8, weight_decay=0.0):
        self.parameters=parameters
        self.lr=lr
        self.beta1,self.beta2=betas
        self.eps=eps
        self.weight_decay=weight_decay

        #state of optimizer
        self.t=0 #step counter
        for p in self.parameters:
            p.m=0 #first moment
            p.v=0#second moment
    def step(self):
        self.t+=1
        for p in self.parameters:
            if p.grad is None:
                continue
            p.m = self.beta1 * p.m + (1 - self.beta1) * p.grad
            p.v = self.beta2 * p.v + (1 - self.beta2) * (p.grad ** 2)
            m_hat = p.m / (1 - self.beta1 ** self.t)
            v_hat = p.v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (v_hat ** 0.5 + 1e-8) + self.weight_decay * p.data)
    def zero_grad(self):
        for p in self.parameters:
            p.grad=0

#### training ####
train_split,val_split,test_split=gen_data_yinyang(random,n=100) # generate a dataset with 100 2-d datapoints in 3 classes
model=MLP(2,[8,3]) # 2d inpit 8 neuron 3 output
optimizer=AdamW(model.parameters(),lr=1e-1,weight_decay=1e-4)
def loss_func(model,split): #evaluate loss func
    total_loss=Value(0.0)
    for x,y in split:
        logits=model(x)
        loss=cross_entropy(logits,y)
        total_loss=total_loss+loss
    mean_loss=total_loss*(1.0/len(split))
    return mean_loss
#### train ####
num_steps=100
for step in range(num_steps):
    if step%10==0:
        val_loss=loss_func(model,val_split)
        print(f"step {step+1}/{num_steps}, val loss {val_loss.data:.6f}")
    loss=loss_func(model,train_split)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"step {step+1}/{num_steps}, train loss {loss.data}")

# for visualization take origin (0,0) and draw the computational graph
x, y = (Value(0.0), Value(0.0)), 0
loss = loss_func(model, [(x, y)])
loss.backward()
try:
    vis_color(x, "lightblue") 
    draw_dot(loss)
except Exception as e:
    print("graphviz not installed? skipped visualization")





