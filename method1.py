import numpy as np
import geninput as INPUT
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from matplotlib import cm
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.externals import joblib
from osgeo import ogr
#import rnn_enc_dec as RED
import spatial_drnnae as SDRNN
import metrics
from skimage import io
import math
from osgeo import gdal

def select_classifier(trX,trY):
	models = []
	models.append(MLPClassifier(hidden_layer_sizes=(10,), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10,5), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10,5,3), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	print "Model evaluation started"
	for model in models:
		scores = cross_val_score(model,trX,trY,cv=5)
		print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
	print "Model evaluation ended"

def method_fit(trX,trY,B, model):
	#pca=PCA(n_components=5)
	#print np.sum(np.logical_and(trY<=0, B>0))
	
	if(model == "sgd"):
		model = SGDClassifier()
	elif(model=="dt"):
		model = DecisionTreeClassifier()
	elif(model=="rf"):
		model = RandomForestClassifier(n_estimators=10)
	elif(model=="mlp"):
		model = MLPClassifier(hidden_layer_sizes=(100,), activation = "tanh", learning_rate='adaptive', max_iter=100)
	#model = MLPClassifier(activation = "tanh", learning_rate='adaptive', max_iter=100)
	elif(model=="ada"):
		model = AdaBoostClassifier()
	elif(model=="knn"):
		model = KNeighborsClassifier()
	elif(model=="svc"):
		model = SVC(kernel = "linear", tol=0.01, cache=7000)
	elif(model=="nb"):
		model = GaussianNB()
	
	scores = cross_val_score(model,trX,trY,cv=2)
	print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
	#print model
	start = time.time()
	model.fit(trX,trY)
	joblib.dump(model, 'model.pkl')
	print time.time()-start
	#exit()
	#sklearn.tree.export_graphviz(model, out_file = 'ca_Decision_tree.dot', max_depth=None)
	return model
	
def method(Rx, res=1):
	Rv = []
	seqlen = []
	index_array = []
	c=0
	
	max_point_count = 0
	cantor_pair_func = lambda k1,k2: (k1+k2)*(k1+k2+1)/2 + k2
	for r in Rx:
		for i in range(0, r.GetGeometryCount()):
			for j in range(0, r.GetGeometryRef(i).GetPointCount()):
				max_point_count = max(r.GetGeometryRef(i).GetPointCount(), max_point_count)
	
	for k in range(0,Rx.shape[0]):
		r = Rx[k]
		if r.GetGeometryCount()>0:
			index_array.append([k,r.GetGeometryCount()])
		for i in range(0,r.GetGeometryCount()):
			x_train = np.zeros(max_point_count)
			for j in range(0,r.GetGeometryRef(i).GetPointCount()):
				x_train[j] = cantor_pair_func(r.GetGeometryRef(i).GetPoint(i)[0],r.GetGeometryRef(i).GetPoint(i)[1])
				
			Rv.append(x_train)
			seqlen.append(r.GetGeometryRef(i).GetPointCount())
	
	print max_point_count
	return np.array(Rv), np.array(seqlen), np.array(index_array)

def method2(Sx, seqlen, trX, trY, index_array, feature_size=3):
	shp = trX.shape
	extra_points = 0
	max_point_count = 0
	
	rnn_model = RED.DynamicRNNAE()
	rnn_model.run_dynamic_rnn(Sx, seqlen)
	
	enc_features = rnn_model.get_encoded_features(Sx, seqlen)
	c = 0
	
	temp = []
	temp_y = []
	for i in range(0,index_array.shape[0]):
		for j in range(0,index_array[i][1]):
			temp.append(trX[i])
			temp_y.append(trY[i])
	
	temp = np.array(temp)
	temp_y = np.array(temp_y)
	temp = np.concatenate([temp, enc_features], axis=1)

	temp2 = trX[[i for i in range(0, trX.shape[0]) if i not in index_array[:,0]]]
	temp2_y = trY[[i for i in range(0, trY.shape[0]) if i not in index_array[:,0]]]
	temp2 = np.concatenate([temp2, np.zeros([temp2.shape[0], feature_size])], axis=1)

	trX_dash = np.concatenate([temp2, temp], axis=0)
	trY_dash = np.concatenate([temp2_y, temp_y])
	
	return trX_dash, trY_dash

def method3(R, Sx, Sy, seqlen, trX, trY, index_array, normc_x, normc_y, feature_size = 1, nsteps = 10):
	shp =trX.shape
	
	rnn_model = SDRNN.DynamicRNNAE(nfeatures = feature_size, tsteps = nsteps, sml = 18)
	
	Sx = Sx[seqlen>=15]
	Sy = Sy[seqlen>=15]
	seqlen = seqlen[seqlen>=15]
	
	start = time.time()
	#mse = rnn_model.run_dynamic_rnn(Sx, Sy, seqlen)
	end = time.time()
	
	dec_features_x, dec_features_y = rnn_model.get_decoded_features(Sx, Sy, seqlen)
	
	#print zip(Sx[0], dec_features_x[0])
	print dec_features_x[0]
	print dec_features_x[1]
	print dec_features_x[2]
	
	method4(R, Sx, Sy, dec_features_x, dec_features_y, seqlen, normc_x, normc_y)
	#exit()
	enc_features = rnn_model.get_encoded_features(Sx, Sy, seqlen)
	temp = []
	k = 0
	c = 0
	for i in range(0,shp[0]):
		if k<index_array.shape[0] and i==index_array[k,0]:
			#k = np.where(index_array[c:,0]==i)
			temp.append(np.mean(enc_features[c:c+index_array[k,1]], axis=0))
			c = c+index_array[k,1]
			k = k+1
		else:
			temp.append(np.zeros(feature_size*2))
	temp = np.array(temp)
	
			
	trX = np.concatenate([trX, temp], axis=1)
	return trX, trY, end-start, mse

def method4(R, Sx, Sy, X, Y, S, normc_x, normc_y):
	X = X*(normc_x[1]-normc_x[0])+normc_x[0]
	Y = Y*(normc_y[1]-normc_y[0])+normc_y[0]
	raster_ds = gdal.Open('/home/ubuntu/workplace/saptarshi/Data/raw/mumbai/2001.tif')
	#f_w = lambda z: math.floor((math.sqrt(8*z+1)-1)/2)
	#f_t = lambda z: (f_w(z)*f_w(z) + f_w(z))/2
	#f_y = lambda z: z- f_t(z)
	#f_x = lambda z: f_w(z)-f_y(z)
	H = np.zeros([R.shape[1], R.shape[2]])
	for i in range(0,X.shape[0]):
		for j in range(0,1):
			#z = X[i][j]
			#print math.fabs(z) - math.fabs(Sx[i][j])
			#if z>=0:
			#[x,y] = [f_x(z), f_y(z)]
			[row,col] = INPUT.coord2pixel(raster_ds, X[i][j], Y[i][j])
			#if(row>=0 and row<R.shape[1] and col>=0 and col<R.shape[2]):
			H[row%R.shape[1]][col%R.shape[2]] = 1
				
	io.imsave("./dataset/debug2.png",np.transpose(H))
	exit()

def method5(Rx, res=1):
	Rv_x = []
	Rv_y = []
	seqlen = []
	index_array = []
	c=0
			
	max_point_count = 0
	for r in Rx:
		for i in range(0, r.GetGeometryCount()):
			for j in range(0, r.GetGeometryRef(i).GetPointCount()):
				max_point_count = max(r.GetGeometryRef(i).GetPointCount(), max_point_count)
	
	for k in range(0,Rx.shape[0]):
		r = Rx[k]
		if r.GetGeometryCount()>0:
			index_array.append([k,r.GetGeometryCount()])
			for i in range(0,r.GetGeometryCount()):
				x_train = np.zeros(max_point_count)
				y_train = np.zeros(max_point_count)
				for j in range(0,r.GetGeometryRef(i).GetPointCount()):
					x_train[j] = r.GetGeometryRef(i).GetPoint(j)[0]
					y_train[j] = r.GetGeometryRef(i).GetPoint(j)[1]
				
				Rv_x.append(x_train)
				Rv_y.append(y_train)
				seqlen.append(r.GetGeometryRef(i).GetPointCount())
	
	print max_point_count, len(Rv_x)
	return np.array(Rv_x), np.array(Rv_y), np.array(seqlen), np.array(index_array)	

def compute_metrics(R, Bt, Btnxt, P):
	shp = R.shape
	C=np.zeros((shp[1],shp[2]))
	M = np.zeros((shp[1],shp[2]))
	k=0
	for i in range(0,shp[1]):
		for j in range(0,shp[2]):
			if(R[0][i][j]>0):
				C[i][j] = P[k]				
				if(P[k]>0 and Btnxt[i][j]>0):
					M[i][j]=1
				elif(P[k]>0 and Btnxt[i][j]<=0):
					M[i][j]=2
				elif(P[k]<=0 and Btnxt[i][j]>0):
					M[i][j]=3
				k=k+1
	
	C = np.asarray(C>0,dtype = np.int32)

	Btd = np.asarray(Bt>0,dtype = np.int32)
	Btnxtd = np.asarray(Btnxt>0,dtype = np.int32)
	return metrics.change_metric(R,Btd,Btnxtd,C)

if __name__=="__main__":
	raw_loc='/home/ubuntu/workplace/saptarshi/Data/raw/mumbai/'
	label_loc='/home/ubuntu/workplace/saptarshi/Data/labelled/mumbai/'
	
	R = INPUT.give_raster(raw_loc + '2001.tif')
	Bt = INPUT.give_raster(label_loc + 'cimg2001.tif')[0]
	Btnxt = INPUT.give_raster(label_loc + 'cimg2011.tif')[0]
	Rx = np.load('./dataset/Road_trX.npy')
	
	#X, seqlen, index_array = method(Rx)
	
	#X, Y, seqlen, index_array = method5(Rx)
	
	#np.save('./dataset/Sequences_X.npy',X)
	#np.save('./dataset/Sequences_Y.npy',Y)
	
	#np.save('./dataset/Sequences_trX.npy',X)
	#np.save('./dataset/SequenceLenth_trX.npy', seqlen)
	#np.save('./dataset/SequenceIndexArray.npy', index_array)
	
	trX = np.load('./dataset/DCAP_trX.npy')
	trX = trX.reshape([trX.shape[0],-1])
	trY = np.load('./dataset/DCAP_trY.npy')
	B = np.load('./dataset/DCAP_B.npy')
	
	trY[np.logical_and(trY>=0, B<0)] = 1
	trY[np.logical_and(trY<=0, B>0)] = 2
	trY[np.logical_and(trY>=0, B>=0)] = 2
	trY[np.logical_and(trY<=0, B<0)] = 0
	
	#X = np.load('./dataset/Sequences_trX.npy')
	X = np.load('./dataset/Sequences_X.npy')
	Y = np.load('./dataset/Sequences_Y.npy')
	seqlen = np.load('./dataset/SequenceLenth_trX.npy')
	index_array = np.load('./dataset/SequenceIndexArray.npy')
	X = X.reshape([X.shape[0],X.shape[1],1])
	Y = Y.reshape([Y.shape[0],Y.shape[1],1])
	
	normc_x = [np.min(np.ndarray.flatten(X)), np.max(np.ndarray.flatten(X))] 
	normc_y = [np.min(np.ndarray.flatten(Y)), np.max(np.ndarray.flatten(Y))] 
	
	X = (X - np.min(np.ndarray.flatten(X)))/(np.max(np.ndarray.flatten(X)) + np.min(np.ndarray.flatten(X)))
	Y = (Y - np.min(np.ndarray.flatten(Y)))/(np.max(np.ndarray.flatten(Y)) + np.min(np.ndarray.flatten(Y)))
	
	nfeature_arr = [1,2,3,4,5]
	n_steps_arr = [10,20,30,40]
	
	c = 0
	
	for nfeatures in nfeature_arr:
		fp = open('./log.txt', 'a')
		for nstep in n_steps_arr:
			c = c+1
			print 'Run '+ str(c)
			fp.write('************************************\n')
			fp.write('Run: '+str(c)+'\n')
			fp.write('Number of features: '+str(nfeatures)+'\n')
			fp.write('Number of steps: '+str(nstep)+'\n')
			
			trX, trY, time_taken, mse = method3(R, X, Y, seqlen, trX, trY, index_array, normc_x, normc_y, nfeatures, nstep)
			fp.write('Time taken: '+str(time_taken)+'\n')
			fp.write('Errors'+'\n')
			fp.write(str(mse)+'\n')
			
			#np.save('./dataset/CAD_trX.npy', trX)
			#np.save('./dataset/CAD_trY.npy', trY)
	
			#trX = np.load('./dataset/CAD_trX.npy')
			#trY = np.load('./dataset/CAD_trY.npy')
	
			print trX.shape
			print "Started training"
			model = RandomForestClassifier(n_estimators=10)
			#scores = cross_val_score(model,trX,trY,cv=5)
			model.fit(trX,trY)
			print "Ended Training"
			P = model.predict(trX)
	
			#print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
			accuracies = compute_metrics(R,Bt,Btnxt,P)
			
			fp.write('Figure of merit: '+str(accuracies[0]) + '\n')
			fp.write('Producer accuracy '+str(accuracies[1]) + '\n')
			fp.write('User accuracy: '+str(accuracies[2]) + '\n')
			fp.write('Overall accuracy: '+str(accuracies[3]) + '\n')
		fp.close()
	print "Program terminated.."
	
	
	
	
	
	
	
	
