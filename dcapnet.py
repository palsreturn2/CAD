import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import training as Training
import indices as INDEX
from theano.compile.debugmode import DebugMode
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit

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
	
	
def model(X,w1,w2,w3):
	layer1 = T.tanh(conv2d(X,w1))
	layer2 = T.tanh(conv2d(layer1,w2))
	output = T.nnet.softmax(T.dot(T.flatten(layer2,outdim=2),w3))
	return layer1,layer2,output

X = T.dtensor4()
B = T.dvector()
Y = T.dmatrix()

reg_param=100

w1 = init_weights((3,4,2,2))
w2 = init_weights((3,3,2,2))
w3 = init_weights((3,3))

l1,l2,py_x = model(X,w1,w2,w3)
py = T.argmax(py_x,axis=1)
params = [w1,w2,w3]
cost = mse(py_x.T,Y.T) + reg_param*T.mean(T.argmax(Y)*B)
updates = Training.sgdm(cost,params,lr=0.1,alpha=0.9)

train = theano.function(inputs = [X,Y,B], updates = updates, outputs = cost, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = py, allow_input_downcast = True)
layer1 = theano.function(inputs = [X], outputs = l1, allow_input_downcast = True)

def fit(trX, trY, B, epoch = 5000, batch_size = 1000, early_stop=False, epsilon=0.00001):
	trX = np.array(trX)
	trY = np.array(trY)
	Y = []
	for y in trY:
		if(y==0):
			Y.append([1,0,0])
		elif(y==1):
			Y.append([0,1,0])
		else:
			Y.append([0,0,1])
	Y = np.array(Y)
	print Y.shape
	val_size = int(0.1*trX.shape[0])	
	pred_acc = 0
	print 'Training started'
	for i in range(epoch):		
		for j in range(0,trX.shape[0],batch_size):
			batch_xs = trX[j:min(j+batch_size,trX.shape[0])]
			batch_ys = Y[j:min(j+batch_size,trX.shape[0])]
			batch_B = B[j:min(j+batch_size,trX.shape[0])]
			#if(np.random.randint(low = 0, high=100)%13==0):
			train(batch_xs, batch_ys, batch_B)
		prX = predict(trX)
		print 'Epoch Loss: ', sklearn.metrics.mean_squared_error(prX,trY)
		
	print 'Training ended'
	return
	
def cross_validate(trX, trY, B):
	rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	for train_index,test_index in rs.split(trX):
		strX = trX[train_index]
		strY = trY[train_index]
		strB = B[train_index]
		steX = trX[test_index]
		steY = trY[test_index]
		steB = B [test_index]
		for i in range(250):		
		for j in range(0,strX.shape[0],1000):
			batch_xs = strX[j:min(j+batch_size,trX.shape[0])]
			batch_ys = strY[j:min(j+batch_size,trX.shape[0])]
			batch_B = strB[j:min(j+batch_size,trX.shape[0])]			
			train(batch_xs, batch_ys, batch_B)
		prX = predict(trX)
		print 'Epoch Loss: ', sklearn.metrics.mean_squared_error(prX,trY)
	


