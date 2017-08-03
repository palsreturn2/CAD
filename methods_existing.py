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
from RBFN import RBF
import time
import run
from sklearn.model_selection import cross_val_score

def method_fit(trX,trY, model):	
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
		model = SVC(kernel = "linear", tol=0.01)
	elif(model=="nb"):
		model = GaussianNB()
	#model = RBF(trX.shape[1],10,1)
	
	#scores = cross_val_score(model,trX,trY,cv=10)
	#print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))

	print model
	start = time.time()
	model.fit(trX,trY)
	print time.time()-start
	#sklearn.tree.export_graphviz(model, out_file = 'ca_Decision_tree.dot', max_depth=5)
	return model
	
if __name__ =="__main__":
	data_folder = './dataset/'
	raw_loc='/home/ubuntu/workplace/saptarshi/Data/raw/mumbai/'
	label_loc='/home/ubuntu/workplace/saptarshi/Data/labelled/mumbai/'
	
	R = INPUT.give_raster(raw_loc + '1991.tif')
	Bt = INPUT.give_raster(label_loc + 'cimg1991.tif')[0]
	Btnxt = INPUT.give_raster(label_loc + 'cimg2001.tif')[0]
	Btnxtnxt = INPUT.give_raster(label_loc + 'cimg2011.tif')[0]
	
	Bt = Bt/255
	Btnxt = Btnxt/255
	Btnxtnxt = Btnxtnxt/255
	
	trX = np.load(data_folder+'nmumbaiX.npy')[:]
	trY = np.load(data_folder+'DCAP_trY.npy')
	B = np.load(data_folder +'DCAP_B.npy')
	teX = np.load(data_folder+'testX.npy')[:]
	
	trY[np.logical_and(trY>=0, B<0)] = 1
	trY[np.logical_and(trY>=0, B>=0)] = 2
	trY[np.logical_and(trY<=0, B<0)] = 0
	
	models = ['sgd','dt','mlp']
	
	for m in models:
		model = method_fit(trX,trY,m)
		run.classify(R,trX,Bt,Btnxt,model)
	#run.classify(R,teX,Btnxt,Btnxtnxt,model)
	
