import numpy as np
import matplotlib.pyplot as plt
import sys

def genplots_ugmetrics(X, fom, pa, ua, oa, filename="ugmetrics.png"):
	plt.figure(1)
	plt.subplot(221)
	plt.plot(X, fom, 'bo-')
	plt.xlabel('Length of fixed length representation')
	plt.ylabel('Figure of merit')
	plt.grid(True)


	plt.subplot(222)
	plt.plot(X, pa, 'bo-')
	plt.xlabel('Length of fixed length representation')
	plt.ylabel('Producers accuracy')
	plt.grid(True)

	plt.subplot(223)
	plt.plot(X, ua, 'bo-')
	plt.xlabel('Length of fixed length representation')
	plt.ylabel('Users accuracy')
	plt.grid(True)

	plt.subplot(224)
	plt.plot(X, oa, 'bo-')
	plt.xlabel('Length of fixed length representation')
	plt.ylabel('Producers accuracy')
	plt.grid(True)
	plt.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.95, hspace=0.45,
                    wspace=0.45)

	plt.savefig(filename)

def ugmetrics_compare(previous, baseline, current, filename="performance_comparison"):
	objects = ['End-to-End', 'baseline method', 'Fixed length method']
	n_groups = 3
	bar_width = 0.15
	opacity = 0.8
	
	fom = [previous[0], baseline[0], current[0]]
	pa = [previous[1], baseline[1], current[1]]
	ua = [previous[2], baseline[2],current[2]]
	oa = [previous[3], baseline[3],current[3]]
	
	index = np.arange(n_groups)
	plt.figure(figsize=(12, 6), dpi=80)

	plt.bar(index, fom, bar_width, alpha = opacity, color='b', label='FoM')
	plt.bar(index+bar_width, pa, bar_width, alpha = opacity, color='g', label='PA')
	plt.bar(index+2*bar_width, ua, bar_width, alpha = opacity, color='r', label='UA')
	plt.bar(index+3*bar_width, oa, bar_width, alpha = opacity, color='y', label='OA')
	
	plt.xlabel('Technique')
	plt.ylabel('Performance')
	plt.xticks(index+2*bar_width, objects)
	plt.legend()
	
	
	plt.tight_layout()
	plt.savefig(filename)

if __name__=='__main__':
	X = np.array([0,2,4,6,8,10])
	fom = np.array([0.5873729089,0.595830837,0.5988396704,0.5995806181,0.5998172867,0.6001656987])
	pa = np.array([0.588618351,0.5970422166,0.6000142946,0.6007210813,0.6009652801,0.6012630835])
	ua = np.array([0.9964106632,0.9966065903,0.9967416051,0.9968441335,0.9968254679,0.9969680774])
	oa = np.array([0.9464464969,0.9475488028,0.9479429794,0.9480422306,0.9480723146,0.9481219402])
	
	#genplots_ugmetrics(X, fom, pa, ua, oa)
	previous = [0.5873729089, 0.588618351, 0.9964106632, 0.9464464969]
	baseline = [0.2288742842, 0.2450326393, 0.7763240659, 0.8930791542]
	current = [0.60016, 0.60126, 0.996968, 0.948121]
	
	ugmetrics_compare(previous, baseline, current)
