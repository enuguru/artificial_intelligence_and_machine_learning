{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# python implementation of simple Linear Regression for of Boston housing price prediction\n",
    "# import the libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Libraries \n",
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Data \n",
    "dataset = pd.read_csv('../../datasets/housing_regression_train.csv')\n",
    "array = dataset.values\n",
    "inputx = array[:,0:13]\n",
    "outputy = array[:,13]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[3.93200e-02 0.00000e+00 3.41000e+00 ... 1.78000e+01 3.93550e+02\n",
      "  8.20000e+00]\n",
      " [8.18700e-02 0.00000e+00 2.89000e+00 ... 1.80000e+01 3.93530e+02\n",
      "  3.57000e+00]\n",
      " [1.02330e+01 0.00000e+00 1.81000e+01 ... 2.02000e+01 3.79700e+02\n",
      "  1.80300e+01]\n",
      " ...\n",
      " [2.39120e-01 0.00000e+00 9.69000e+00 ... 1.92000e+01 3.96900e+02\n",
      "  1.29200e+01]\n",
      " [7.99248e+00 0.00000e+00 1.81000e+01 ... 2.02000e+01 3.96900e+02\n",
      "  2.45600e+01]\n",
      " [4.74100e-02 0.00000e+00 1.19300e+01 ... 2.10000e+01 3.96900e+02\n",
      "  7.88000e+00]]\n"
     ]
    }
   ],
   "source": [
    "input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/3, random_state = 0)\n",
    "print(input_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# using simple Linear Regression model to train\n",
    "model = LinearRegression()\n",
    "model.fit(input_train, output_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[27.48203031 34.81495203 19.14830213 31.15517689 18.90640139 20.13439671\n",
      " 21.13911575 11.27952397 34.08071952 23.04583852  8.9894001  25.23324588\n",
      " 22.60901045 33.25963219 22.9145782  32.64213819 16.50336377 41.28536879\n",
      " 21.09514107 22.18243744 16.88700239  5.56147599 20.12813442 19.58991875\n",
      " 19.85172434  6.85074055 15.50460066 17.14383131 36.01187876 32.96779711\n",
      "  9.34066617 20.81196989 22.41767179 22.25693054 23.50711656 20.2276073\n",
      " 25.8351597  16.194724   25.42943971 33.72783492 24.76509571 18.0596646\n",
      " 13.21282185 39.88859115 23.49117352 30.95805024 10.23352218 17.9504915\n",
      " 13.37035994 20.23700475 11.84276197 25.40058702 23.26606458 12.1874096\n",
      " 16.16509383 19.56649749 20.21766357 26.46379915 15.74915456 45.19930008\n",
      " 22.57593753 35.13429413 16.89445221 19.06178117 32.5845866  20.30196841\n",
      " 22.3935304  20.01213954 15.05259381 23.42287455 12.00148017 26.07144038\n",
      " 29.03182243 19.00910651 25.31877714 33.53018041  7.88108708 25.57876677\n",
      " 18.91033213 16.66819925 32.86845138 28.77615049 32.11707431 16.44948534\n",
      " 19.51350771 21.96104062 14.8657819  30.38848254 18.04001994 31.52948641\n",
      "  2.65785453 20.35914517 32.39859926 19.31832391 34.46651256  9.5676025\n",
      " 34.19802878 17.61601911 18.95067142 13.43641296 22.70807329 41.37953676\n",
      " 24.31058728 23.56462775 20.10795386  6.81953762 25.54619967 24.58467126\n",
      " 28.7936725  14.75935918 31.97901199 11.58114267 13.71829583 19.61461684\n",
      " 17.61950365  4.23812052 26.34569474 30.07853105  9.17888435 27.5471907\n",
      " 15.33537593 21.34841304 17.45701125 37.93417069 12.85484907 19.90953565\n",
      " 18.03065814 18.91991992 20.13631711 13.3217804  14.9282682  25.15379964\n",
      " 34.88429912 25.97455911 28.02989454 22.69191268 20.78750219 15.82127169\n",
      " 30.23768689  4.97755321 18.18408843 24.04999156 24.23309246 14.71014442\n",
      " 15.28958699 25.50215912 24.80587623 24.3518394  21.2799728  23.49849004\n",
      " 14.4622459  13.15824114 16.05998895 33.78426406 15.80119373 13.59604923\n",
      " 20.9982898  17.64837495 35.02215913 19.19678955 21.06881645 23.72145873\n",
      " 30.98880808 14.44828842 17.86728161 25.94756069 20.98346546 12.50066984\n",
      " 22.4130159 ]\n"
     ]
    }
   ],
   "source": [
    "# model predicting the Test set results\n",
    "predicted_output = model.predict(input_test)\n",
    "print(predicted_output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
