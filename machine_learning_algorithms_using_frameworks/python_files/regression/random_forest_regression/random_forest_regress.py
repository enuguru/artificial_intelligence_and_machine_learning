
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                               random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,
                                     n_estimators=100)
regr.fit(X, y)
print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
