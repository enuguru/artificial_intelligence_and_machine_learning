# Example of making a prediction with coefficients

# Make a prediction
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i+1] * row[i]
	return yhat

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
#dataset = [[10.26,7400,40,12413], [10.26,7500,50,12413], [4.99,7500,50,4193], [20.57,7500,50,24800], [40.01,7500,50,49500]]
coef = [500.4, 0.3,1]
for row in dataset:
	yhat = predict(row, coef)
	print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))
