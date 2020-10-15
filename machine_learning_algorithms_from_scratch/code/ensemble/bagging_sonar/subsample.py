# Example of subsampling a dataste
from random import seed
from random import random
from random import randrange

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers) / float(len(numbers))

# Test subsampling a dataset
seed(1)
# True mean
dataset = [[randrange(10)] for i in range(20)]
print('True Mean: %.3f' % mean([row[0] for row in dataset]))
# Estimated means
ratio = 0.10
for size in [1, 10, 100]:
	sample_means = list()
	for i in range(size):
		sample = subsample(dataset, ratio)
		sample_mean = mean([row[0] for row in sample])
		sample_means.append(sample_mean)
	print('Samples=%d, Estimated Mean: %.3f' % (size, mean(sample_means)))