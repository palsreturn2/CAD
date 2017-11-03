import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

class MyLSTMCell(BasicLSTMCell):
	def __init__(self, num_units, resolution = 100, val_range = [0.005, 0.009]):
		BasicLSTMCell.__init__(self,num_units)
		learning_rate = 0.1
		resolution = 100

		x = tf.placeholder("float", [None,1,1])	
		y = tf.placeholder("float", [None,1,1])
		
		x_list = tf.unstack(x, 1, 1)
		y_list = tf.unstack(y, 1, 1)

		#outputs, states = BasicLSTMCell.apply(self, x, state)
		outputs, states = tf.nn.dynamic_rnn(self, x, dtype = tf.float32)

		outputs = tf.stack(outputs)
		outputs = tf.transpose(outputs, [1, 0, 2])

		mse = tf.losses.mean_squared_error(labels = y, predictions = outputs)
		loss = tf.reduce_mean(mse)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
	
		sess = tf.Session()
		saver = tf.train.Saver()

		init = tf.global_variables_initializer()
		sess.run(init)
		
		X,Y = self.cons_func_data([-1,1], 0.901, resolution = resolution)
		for i in range(0,1000):
			sess.run(optimizer, feed_dict = {x: X, y: Y})
			#print sess.run(loss, feed_dict = {x: X, y: Y})

		Y_pred = sess.run(outputs, feed_dict = {x:X})
		'''plt.ylim(0.005, 0.009)
		plt.plot(Y.reshape([resolution]), 'ro')
		plt.plot(Y_pred.reshape([resolution]), 'bo')
		plt.show()'''

	def cons_func_data(self, val_range, constant, resolution = 100):
		X = np.random.uniform(val_range[0], val_range[1], resolution)
		X = X.reshape([resolution,1,1])
	
		Y = np.array([constant for i in range(0, resolution)])
		Y = Y.reshape([resolution,1,1])
	
		return X,Y
	
	
if __name__=='__main__':
	model = MyLSTMCell(1)







