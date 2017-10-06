import tensorflow as tf
import numpy as np

class DynamicRNNAE:
	def __init__(self):
		# Parameters
		self.learning_rate = 0.1
		self.training_steps = 1000
		self.batch_size = 128
		self.display_step = 200

		# Network Parameters
		self.seq_max_len = 36 # Sequence max length
		self.n_hidden = 64 # hidden layer num of features
		
		# tf Graph input
		self.x = tf.placeholder("float", [None,self.seq_max_len,1])	
		# A placeholder for indicating each sequence length
		self.seqlen = tf.placeholder("int32", [None])
		
		self.weights = {'out': tf.Variable(tf.random_normal([self.n_hidden,1]))}
		self.biases = {'out': tf.Variable(tf.random_normal([1]))}
	
	def rnn_encoder(self):
		x_list = tf.unstack(self.x, self.seq_max_len, 1)
		
		with tf.variable_scope('lstm1'):
			encoder = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
			outputs_enc, states = tf.contrib.rnn.static_rnn(encoder, x_list, dtype=tf.float32, sequence_length=self.seqlen)
		
		outputs_enc = tf.stack(outputs_enc)
		outputs_enc = tf.transpose(outputs_enc, [1, 0, 2])
		
		batch_size = tf.shape(outputs_enc)[0]
		index = tf.range(0, batch_size) * self.seq_max_len + (self.seqlen - 1)
		
		outputs_enc = tf.gather(tf.reshape(outputs_enc, [-1, self.n_hidden]), index)
		
		return outputs_enc
	
	def rnn_decoder(self):
		outputs_enc = self.rnn_encoder()
		enc_rep = tf.matmul(outputs_enc, self.weights['out']) + self.biases['out']
		enc_rep = tf.tile(tf.reshape(enc_rep, [-1,1,1]), multiples = (1,self.seq_max_len,1))
		enc_rep = tf.unstack(enc_rep, self.seq_max_len, 1)
		
		with tf.variable_scope('lstm2'):
			decoder = tf.contrib.rnn.BasicLSTMCell(1)
			outputs_dec, states = tf.contrib.rnn.static_rnn(decoder, enc_rep, dtype=tf.float32, sequence_length=self.seqlen)
		
		outputs_dec = tf.stack(outputs_dec)
		outputs_dec = tf.transpose(outputs_dec, [1, 0, 2])
		
		return outputs_dec
	
	def dynamic_train(self):
		dec_output = self.rnn_decoder()
		mse = tf.losses.mean_squared_error(labels = self.x, predictions = dec_output) 
		cost = tf.reduce_mean(mse)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(cost)
		
		return cost, optimizer
	
	def run_dynamic_rnn(self, X, S):
		#dec_output = self.rnn_decoder()
		#enc_output = self.rnn_encoder()
		cost, optimizer = self.dynamic_train()
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		
		for i in range(0,self.training_steps):
			for j in range(0,X.shape[0], self.batch_size):
				batch_x = X[j:min(j+self.batch_size, X.shape[0]),:,:]
				batch_s = S[j:min(j+self.batch_size, S.shape[0])]				
				self.sess.run(optimizer, feed_dict = {self.x: batch_x, self.seqlen: batch_s})
			print self.sess.run(cost, feed_dict = {self.x: X, self.seqlen: S})
		return 
			
		
		
if __name__ == '__main__':
	rnn = DynamicRNNAE()
	X = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]])
	S = np.random.randint(10, size = (1,))
	X = X.reshape([1,-1,1])/np.max(np.ndarray.flatten(X))
	rnn.run_dynamic_rnn(X, S)
	
	
	
	
	
	
