import tensorflow as tf
import numpy as np

class DynamicRNNAE:
	def __init__(self):
		# Parameters
		self.learning_rate = 0.1
		self.training_steps = 30
		self.batch_size = 128
		self.display_step = 200

		# Network Parameters
		self.seq_max_len = 36 # Sequence max length
		self.n_hidden = 1 # hidden layer num of features
		
		# tf Graph input
		self.x = tf.placeholder("float", [None,self.seq_max_len,1])	
		# A placeholder for indicating each sequence length
		self.seqlen = tf.placeholder("int32", [None])
		
		self.weights = {'out': tf.Variable(tf.random_normal([self.n_hidden,1]))}
		self.biases = {'out': tf.Variable(tf.random_normal([1]))}
		
		x_list = tf.unstack(self.x, self.seq_max_len, 1)
		
		with tf.variable_scope('lstm1'):
			encoder = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
			self.outputs_enc, states = tf.contrib.rnn.static_rnn(encoder, x_list, dtype=tf.float32, sequence_length = self.seqlen)
		
		self.outputs_enc = tf.stack(self.outputs_enc)
		self.outputs_enc = tf.transpose(self.outputs_enc, [1, 0, 2])
		
		batch_size = tf.shape(self.outputs_enc)[0]
		index = tf.range(0, batch_size) * self.seq_max_len + (self.seqlen - 1)
		
		self.outputs_enc = tf.gather(tf.reshape(self.outputs_enc, [-1, self.n_hidden]), index)
			
		enc_rep = tf.matmul(self.outputs_enc, self.weights['out']) + self.biases['out']
		enc_rep = tf.tile(tf.reshape(enc_rep, [-1,1,1]), multiples = (1,self.seq_max_len,1))
		enc_rep = tf.unstack(enc_rep, self.seq_max_len, 1)
		
		with tf.variable_scope('lstm2'):
			decoder = tf.contrib.rnn.BasicLSTMCell(1)
			self.outputs_dec, states = tf.contrib.rnn.static_rnn(decoder, enc_rep, dtype=tf.float32, sequence_length=self.seqlen)
		
		self.outputs_dec = tf.stack(self.outputs_dec)
		self.outputs_dec = tf.transpose(self.outputs_dec, [1, 0, 2])

		mse = tf.losses.mean_squared_error(labels = self.x, predictions = self.outputs_dec) 
		self.cost = tf.reduce_mean(mse)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
		
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
	
	def run_dynamic_rnn(self, X, S):
		#dec_output = self.rnn_decoder()
		#enc_output = self.rnn_encoder()
		for i in range(0,self.training_steps):
			for j in range(0,X.shape[0], self.batch_size):
				batch_x = X[j:min(j+self.batch_size, X.shape[0]),:,:]
				batch_s = S[j:min(j+self.batch_size, S.shape[0])]				
				self.sess.run(self.optimizer, feed_dict = {self.x: batch_x, self.seqlen: batch_s})
			print self.sess.run(self.cost, feed_dict = {self.x: X, self.seqlen: S})
		return 
	
	def get_encoded_features(self, X, S):
		return self.sess.run(self.outputs_enc, feed_dict={self.x: X, self.seqlen: S})
		
		
if __name__ == '__main__':
	rnn = DynamicRNNAE()
	X = np.array([[i for i in range(0,36)]])
	S = np.random.randint(10, size = (1,))
	X = X.reshape([1,-1,1])/np.max(np.ndarray.flatten(X))
	rnn.run_dynamic_rnn(X, S)
	
	
	
	
	
	
