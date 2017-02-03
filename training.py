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

def adagrad(cost,params,lr=0.01):
	gradients = T.grad(cost,params)
	updates = []
	for p,g in zip(params,gradients):
		param_updates = theano.shared(0.,broadcastable=p.broadcastable)
		updates.append([p,p-(lr*g/T.sqrt(param_updates))])
		updates.append([param_updates,param_updates+g*g])
	return updates
