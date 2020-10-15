import random
import preprocess
import nltk

def get_classifier():

	data = preprocess.get_data()
	random.shuffle(data)

	split = int(0.8 * len(data))

	train_set = data[:split]
	test_set =  data[split:]

	classifier = nltk.NaiveBayesClassifier.train(train_set)

	accuracy = nltk.classify.util.accuracy(classifier, test_set)
	print("Generated Classifier")
	print('-'*70)
	print("Accuracy: ", accuracy)
	return classifier