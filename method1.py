import numpy as np
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
import time

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

def method_fit(trX,trY,B):
	#pca=PCA(n_components=5)
	#print np.sum(np.logical_and(trY<=0, B>0))
	trY[np.logical_and(trY>=0, B<0)] = 1
	trY[np.logical_and(trY<=0, B>0)] = 2
	trY[np.logical_and(trY>=0, B>=0)] = 2
	trY[np.logical_and(trY<=0, B<0)] = 0
	
	#model = SGDClassifier()
	#model = DecisionTreeClassifier()
	#model = RandomForestClassifier(max_depth=100,n_estimators=10)
	#model = MLPClassifier(hidden_layer_sizes=(10,5,3), activation = "tanh", learning_rate='adaptive', max_iter=100)
	#model = MLPClassifier(activation = "tanh", learning_rate='adaptive', max_iter=100)
	model = AdaBoostClassifier()
	#model = SVC(kernel = "linear", tol=0.01)
	#model = GaussianNB()
	
	#scores = cross_val_score(model,trX,trY,cv=2)
	#print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
	start = time.time()
	model.fit(trX,trY)
	print time.time()-start
	#exit()
	#sklearn.tree.export_graphviz(model, out_file = 'ca_Decision_tree.dot', max_depth=3)
	return model

	
	
