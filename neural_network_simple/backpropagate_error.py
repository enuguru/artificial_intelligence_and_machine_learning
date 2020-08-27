# Example of backpropagating error

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	print(7,output * (1.0 - output))
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		print(7,errors)
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					print(5,neuron['weights'][j])
					print(4,neuron['delta'])
					error += (neuron['weights'][j] * neuron['delta'])
					print(1,error)
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				print(6,neuron)
				errors.append(expected[j] - neuron['output'])
				print(2,errors)
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
			print(3,neuron['delta'])

# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)
