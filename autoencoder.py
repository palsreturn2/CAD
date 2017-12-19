from keras.layers import Dense, Input
from keras.models import Model
import numpy as np

class Autoencoder:
	def __init__(self, input_dim, enc_dim = 3):
		self.encode_dim = enc_dim
		I = Input(shape=(input_dim,))
		
		encoded = Dense(self.encode_dim, activation='relu')(I)
		decoded = Dense(input_dim, activation='sigmoid')(encoded)

		self.autoencoder = Model(I, decoded)
		self.encoder = Model(I, encoded)
		
		encoded_input = Input(shape=(self.encode_dim,))
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(encoded_input, decoder_layer(encoded_input))

		self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	
	def train(self, X):
		self.autoencoder.fit(X, X, epochs=50, batch_size=1000, shuffle=True)
	
	def encode(self, X):
		return self.encoder.predict(X)
	
	def decode(self, X):
		return self.decoder.predict(self.encoder.predict(X))

if __name__=='__main__':
	X = np.load('./dataset/DCAP_trX.npy')
	X = X.reshape([X.shape[0], -1])
	Xi = X[:,0:27]
	ae = Autoencoder(input_dim = Xi.shape[1], enc_dim = 5)
	ae.train(Xi)
	Ex = ae.encode(Xi)
	
	E = np.concatenate([X[:,0:9],Ex], axis=1)
	
	np.save('./dataset/CAD3_trX.npy', E)
	
	
