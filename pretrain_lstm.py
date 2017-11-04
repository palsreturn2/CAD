import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

class MyLSTMCell(BasicLSTMCell):
	def __init__(self, num_units, resolution = 500, val_range = [-1, 1]):
		BasicLSTMCell.__init__(self,num_units)
		learning_rate = 0.1

		x = tf.placeholder("float", [None,1,1])	
		y = tf.placeholder("float", [None,1])
		#self.weights = tf.Variable(tf.random_normal([num_units,1]))
		#self.bias = tf.Variable(tf.random_normal([1]))
		
		self.weights(num_units)
		self.bias()
		
		x_list = tf.unstack(x, 1, 1)

		outputs, states = tf.nn.dynamic_rnn(self, x, dtype = tf.float32)
		outputs = tf.reshape(outputs, [-1,1])
		#outputs = tf.stack(outputs)
		#outputs = tf.transpose(outputs, [1, 0, 2])
		#batch_size = tf.shape(outputs)[0]
		#index = tf.range(0, batch_size) * 1
		#outputs = tf.gather(tf.reshape(outputs, [-1, num_units]), index)
		
		#self.y_pred = tf.matmul(outputs, self.weights) + self.bias
		
		self.y_pred(outputs)

		mse = tf.losses.mean_squared_error(labels = y, predictions = self.y_pred)
		loss = tf.reduce_mean(mse)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
	
		sess = tf.Session()
		saver = tf.train.Saver()

		init = tf.global_variables_initializer()
		sess.run(init)
		
		X,Y = self.cons_func_data(val_range, 0.92, resolution = resolution)
		
		for i in range(0,1000):
			sess.run(optimizer, feed_dict = {x: X, y: Y})
			#print sess.run(loss, feed_dict = {x: X, y: Y})

		'''X, y = self.cons_func_data(val_range, 0.92, resolution = resolution)
		Y_pred = sess.run(y_pred, feed_dict = {x:X})
		print Y_pred
		plt.ylim(0.5, 1.5)
		plt.plot(Y.reshape([resolution]), 'ro')
		plt.plot(Y_pred.reshape([resolution]), 'bo')
		plt.show()'''
	
	def weights(self, num_units):
		self.weights = tf.Variable(tf.random_normal([num_units,1]))
	
	def bias(self):
		self.bias = tf.Variable(tf.random_normal([1]))
	
	def y_pred(self, outputs):
		self.y_pred = tf.matmul(outputs, self.weights) + self.bias
	
	def cons_func_data(self, val_range, constant, resolution = 100):
		X = np.random.uniform(val_range[0], val_range[1], resolution)
		X = X.reshape([resolution,1,1])
	
		Y = np.array([constant for i in range(0, resolution)])
		Y = Y.reshape([resolution,1])
	
		return X,Y
	
	
if __name__=='__main__':
	X = tf.placeholder("float", [None, 3, 1])
	x_list = tf.unstack(X, 3, 1)
	model = MyLSTMCell(1)
	outputs, states = tf.contrib.rnn.static_rnn(model.y_pred, x_list, dtype=tf.float32)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	y = sess.run(outputs, feed_dict={X:np.random.uniform(-1,1,100*3).reshape([100,3,1])})
	print y[0]




