import math, random
import os, csv

def dotsum():
	pass

def dotproduct():
	pass

def backPropogation():
	pass

def activation():
	pass


def main():
	# Imput from CSV. Store Number of input Neurons NumI . Ask for num of hidden Neurons NumH, NumO=4
	filename = 'Colon_Cancer_CNN_Features.csv'
	featureMatrix = [[]]
	class_vector = []

	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		featureMatrix = [[1] for i in range(6523)]
		print(featureMatrix)
		for row in csvreader:
			i=0
			print(row[:-1])
			featureMatrix[i].append(row[:-1])
			class_vector.append(row[-1])
			# print(class_vector) 
			i+=1
		print(class_vector)

	# Randomize 2 Weight Matrices Weights_ih, Weights_ho

 	## Set epochs to 2
	### loop over records

	# Calculate Weighted Sum hidden layer

	# Activation Function hidden layer

	# Calculate Weighted Sum output layer

	# Activation Function output layer

	# Back propogation --> update 2 weight matrix 


	## output 
if __name__ == '__main__':
	main()