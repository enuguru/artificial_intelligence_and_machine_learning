# Machine Learning Engineer Nanodegree
## Project 2: Predicting Boston Housing Prices


### Project Description

This is the 2nd project for the Machine Learning Engineer Nanodegree. In this project, we use the Boston Housing dataset to train an optimal decision tree algorithm to predict the best selling price of a home in Boston based on certain features of the homes and statistical analysis. 

The dataset for this project contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. This dataset was preprocessed through log transformations (to reduce impact of highly skewed features), feature scaling (to ensure the features were treated equally) and one-hot-encoding (to treat non-numeric features).

Additionally, the learning and complexity curves of the decision tree model were studied to determine optimal number of training points needed as well as the maximum depth of the decision tree that minimized bias and variance in the price predictions 

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

### Code

The main code for this project is located in the `boston_housing.ipynb` notebook file. Additional supporting code for visualizing the necessary graphs can be found in `visuals.py`. Additionally, the "Report.html" file contains a snapshot of the main code in the jupyter notebook with all code cells executed.

### Run

In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:

```bash
ipython notebook boston_housing.ipynb
```  
or
```bash
jupyter notebook boston_housing.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

The modified Boston housing dataset consists of 490 data points, with each datapoint having 3 features. This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

**Features**
1.  `RM`: average number of rooms per dwelling
2. `LSTAT`: percentage of population considered lower status
3. `PTRATIO`: pupil-student ratio by town

**Target Variable**
4. `MEDV`: median value of owner-occupied homes
