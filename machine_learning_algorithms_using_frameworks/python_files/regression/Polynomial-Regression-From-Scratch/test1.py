from generatePolyPoints import generatePolyPoints
from polynomial_regression import PolynomialRegression

x_pts, y_pts = generatePolyPoints(0, 50, 100, [5, 1, 1], 
                                          noiseLevel = 2, plot = 1)
PR = PolynomialRegression(x_pts, y_pts)
theta = PR.fit(method = 'normal_equation', order = 2)
PR.plot_predictedPolyLine()
