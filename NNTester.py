import math
import random
import string
import sys

#########################################Network class################################################
class NeuralNetwork(object):
	def __init__(self, numInput, numHidden, numOutput):
		self.numInput = numInput + 1	#nums of nodes in each layer
		self.numHidden = numHidden
		self.numOutput = numOutput

		self.inputLayer = [1] * self.numInput	#creating layers
		self.hiddenLayer = [1] * self.numHidden
		self.outputLayer = [1] * self.numOutput

		#node weight matrixes
		self.connInput = makeMatrix (self.numInput, self.numHidden)
		self.connHidden = makeMatrix (self.numHidden, self.numOutput)
		#initializeVals to random vals
		randomizeMatrix(self.connInput, -0.2, 0.2)
		randomizeMatrix(self.connHidden, -0.2, 0.2)
		#last change matrices for momentum
		self.changeInput = makeMatrix(self.numInput, self.numHidden)
		self.changeOutput = makeMatrix(self.numHidden, self.numOutput)

	def tick(self, inputs):
		for x in range(0, len(inputs)):	#puts values into input nodes
			self.inputLayer[x] = inputs[x]
		for x in range(0, self.numHidden): #fires to hidden layer
			sume = 0.0
			for y in range(0, self.numInput):
				sume += self.inputLayer[y] * self.connInput[y][x]
			self.hiddenLayer[x] = sigmoid(sume)
			print x #tester
		for x in range(0, self.numOutput):
			sume = 0.0
			for y in range(0, self.numHidden):
				sume += self.hiddenLayer[y] * self.connHidden[y][x]
			self.outputLayer[x] = sigmoid(sume)
		return self.outputLayer

	def backProp(self, targets, learningRate, momentum):
		outputErrors = [0] * self.numOutput
		for x in range(0, self.numOutput):	#initial errors
			error = targets[x] - self.outputLayer[x]
			outputErrors[x] = error * dsigmoid(self.outputLayer[x])

		for x in range(0, self.numHidden):	#update hidden to output conns
			for y in range(0, self.numOutput):
				change = outputErrors[y] * self.hiddenLayer[x]
				self.connHidden[x][y] += learningRate * change + momentum * self.changeOutput[x][y]
				self.changeOutput[x][y] = change

		hiddenErrors = [0] * self.numHidden
		for x in range(0, self.numHidden):	#hidden errors
			error = 0.0
			for y in range(0, self.numOutput):
				error += outputErrors[y] * self.connHidden[x][y]
			hiddenErrors[x] = error * dsigmoid(self.hiddenLayer[x])

		for x in range(0, self.numInput):	#update input to hidden conns
			for y in range(0, self.numHidden):
				change = hiddenErrors[y] * self.inputLayer[x]
				self.connInput[x][y] += learningRate * change + momentum * self.changeInput[x][y]
				self.changeInput[x][y] = change

		error = 0.0	#total complete error
		for x in range(0, len(targets)):
			error += 0.5 * (targets[x] - self.outputLayer[x])**2
		return error

	def test(self, patterns):
		for p in patterns:
			inputs = p[0]
			print 'Inputs:', p[0], '-->', self.tick(inputs), '\tTarget', p[1]

	def train (self, patterns, max_iterations = 1000, N=0.5, M=0.1):
		for i in range(max_iterations):
			for p in patterns:
				inputs = p[0]
				targets = p[1]
			self.tick(inputs)
			error = self.backProp(targets, N, M)
		if i % 50 == 0:
			print 'Combined error', error
		self.test(patterns)

######Peripheral functions
def sigmoid (x):
  return math.tanh(x)
def dsigmoid (y):
  return 1 - y**2

def makeMatrix ( I, J, fill=0.0):
  m = []
  for i in range(I):
    m.append([fill]*J)
  return m
  
def randomizeMatrix ( matrix, a, b):
  for i in range ( len (matrix) ):
    for j in range ( len (matrix[0]) ):
      matrix[i][j] = random.uniform(a,b)


#####MAIN
def main ():
  pat = [
      [[0,0], [1]],
      [[0,1], [1]],
      [[1,0], [1]],
      [[1,1], [0]]
  ]
  myNN = NeuralNetwork( 2, 2, 1)
  myNN.train(pat)

if __name__ == "__main__":
    main()
