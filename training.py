#Training algorithms for feed forward neural networks
import theano
from theano import tensor as T

def sgd(cost,params,lr=0.01):
	gradients=T.grad(cost=cost,wrt=params)
	updates=[]
	for g,p in zip(gradients,params):
		updates.append([p,p-(g*lr)])
	return updates
	
def sgdm(cost,params,lr=0.01,alpha=0.9):
	assert alpha>=0 and alpha<1
	gradients=T.grad(cost=cost,wrt=params)
	updates=[]
	for p,g in zip(params,gradients):
		param_updates=theano.shared(p.get_value()*0.,broadcastable=p.broadcastable)
		updates.append([p,p-(lr*param_updates)])
		updates.append([param_updates,alpha*param_updates+(1-alpha)*g])
	return updates

def adam(cost,params,lr = 0.01,beta1 = 0.9, beta2=0.999, epsilon = 0.00000001):
	gradients = T.grad(cost,params)
	updates = []
	t = 0
	for p,g in zip(params,gradients):
		mt = theano.shared(0.,broadcastable = p.broadcastable)
		vt = theano.shared(0.,broadcastable = p.broadcastable)
		updates.append([mt,(beta1*mt + (1-beta1)*g)/(1-(beta1**t))])
		updates.append([vt,(beta2*vt + (1-beta2)*(g**2))/(1-(beta2**t))])
		updates.append([p, p - alpha*mt/(epsilon+T.sqrt(vt))])
		t=t+1
	return updates
		
