import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import training as Training
import indices as INDEX
from theano.compile.debugmode import DebugMode
import sklearn.metrics

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape)*0.01))

	
def relu(X):
	return T.maximum(0.5,X)

def mse(X,Y):
	return T.mean((X-Y)**2)
	
def logloss(X,Y):
	return -T.mean(Y*T.log(X)+(1-Y)*T.log(1-X))

def dropout_from_layer(layer, p):    
	rng = np.random.RandomState(1000)
	srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
	
	mask = srng.binomial(n=1, p=1-p, size=layer.shape) 
	
	output = layer * T.cast(mask, theano.config.floatX)
	return output
	
def model(X,w1,w2,w3,w4):
	layer1 = T.tanh(T.dot(X,w1))
	layer2 = T.tanh(T.dot(layer1, w2))
	layer3 = T.tanh(T.dot(layer2, w3))
	py = T.dot(layer3,w4)
	return layer1,layer2,layer3,py

X = T.dmatrix()
Y = T.dvector()
B = T.dvector()

w1 = init_weights((36,40))
w2 = init_weights((40,20))
w3 = init_weights((20,10))
w4 = init_weights((10,1))

params = [w1,w2,w3,w4]

reg_param = 0

l1,l2,l3,py_x = model(X,w1,w2,w3,w4)

cost = mse(py_x.T,Y) + reg_param * T.mean(T.dot(Y,B))

updates = Training.sgdm(cost,params,lr=3,alpha=0.4)
train = theano.function(inputs = [X,Y,B], updates = updates, outputs = cost, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = py_x, allow_input_downcast = True)

def fit(trX, trY, B, epoch = 10, batch_size = 1000, early_stop=False, epsilon=0.00001):
	trX = np.array(trX)
	trY = np.array(trY)
	val_size = int(0.1*trX.shape[0])	
	pred_acc = 0
	print 'Training started'
	for i in range(epoch):		
		for j in range(0,trX.shape[0],batch_size):
			batch_xs = trX[j:min(j+batch_size,trX.shape[0])]
			batch_ys = trY[j:min(j+batch_size,trX.shape[0])]
			batch_B = B[j:min(j+batch_size,trX.shape[0])]
			if(np.random.randint(low = 0, high=100)%13==0):
				train(batch_xs, batch_ys, batch_B)
		prX = predict(trX)
		eloss = sklearn.metrics.mean_squared_error(prX,trY)	
		print 'Epoch Loss: ', eloss
		if(early_stop and eloss<epsilon):
			break		
	print 'Training ended'
	return
