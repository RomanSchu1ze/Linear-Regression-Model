# Linear Regression Model implementation using NumPy
# Optimization method: Ordinary Least Squares
# Author: Roman Schulze

# import libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

# ----------------------------------------------------------------------------
# 1. Implementation ingredients
# ----------------------------------------------------------------------------

# Parameters:
# X: array of independent variables
# y: dependent variable
# n: number of observations
# p: number of independent variables

# Variables:
# y_hat: predicted values
# residuals: difference between y and y_hat
# Measures of fit to evaluate model performance: R_2, adj_R_2, RMSE
# beta = vector of coefficients
# the formula to derive beta is given by: beta = ((X'X)^-1)X'y

# ----------------------------------------------------------------------------
# 1.1 Define Functions
# ----------------------------------------------------------------------------

# derive coefficients vector beta
def get_beta(X, y, n):
    """
    Derive vector of coefficients:
 
    Parameters
    ----------    
    X: array of independent variables
    y: dependent variable
    n: number of observations
    
    Returns
    -------
    beta: vector of coefficients
    """
    # define a vector of ones having the same number of rows as array X
    ones = np.ones((n, 1))
    # concatenate ones and X horizontally which is needed to estimate
    # the Intercept
    X = np.hstack((ones, X))
    # 1. Derive beta in the following three steps:
    X_t_X_inv = np.linalg.inv(np.dot(np.transpose(X), X))
    # 2. X'X^-1X'
    X_t_X_inv_X_t = np.dot(X_t_X_inv, np.transpose(X))
    # 3. finally muliply y to derive beta
    beta = np.dot(X_t_X_inv_X_t, y)
    # return beta
    return beta

# derive predicted values
def get_predictions(X_test, beta, n):
    """
    Derive predicted values of dependent variable y:
   
    Parameters
    ----------    
    X_test: array of independent variables
    y_test: dependent variable
    n: number of observations
    
    Returns
    -------
    y_hat: vector of predicted values
    """
    # define a vector of ones having the same number of rows as X
    ones = np.ones((n, 1))
    # combine ones and X horizontally
    X = np.hstack((ones, X_test))
    # predict y
    y_hat = np.dot(X, beta)
    # return predictions
    return y_hat

# derive residuals
def get_residuals(y, y_hat):
    """
    Derive vector of residuals:
   
    Parameters
    ----------    
    y: dependent variable
    y_hat: predicted values of y 
    
    Returns
    -------
    residuals: vector containing difference of observed and predicted values
    """
    # Derive difference between predictions and acutal realizations of y
    residuals = np.subtract(y_hat, y)
    # return residuals
    return residuals

# derive R squared
def get_r_2(y, y_hat):
    """
    Derive R sqaured, a performance measure:
   
    Parameters
    ----------    
    y: dependent variable
    y_hat: predicted values of y 
    
    Returns
    -------
    r_2: scalar, to measure performance of model
    """
    # nominator
    nom = np.sum(np.subtract(y, y_hat)**2)
    # denominator
    denom = np.sum(np.subtract(y, np.mean(y))**2)
    # get R squared
    r_2 = 1 - (nom / denom)
    # return r_2
    return r_2

# derive adjusted R squared
def get_adj_r_2(y, y_hat, beta, n, p):
    """
    Derive adjusted R sqaured, a performance measure:
   
    Parameters
    ----------    
    y: dependent variable
    y_hat: predicted values of y 
    beta: vector of coefficients
    n: number of observations
    p: number of independent variables
        
    Returns
    -------
    adj_r_2: scalar, to measure performance of model
    """
    # derive nominator
    nom = (1 - get_r_2(y, y_hat)) * (n - 1)
    # derive denominator
    denom = (n - p - 1)
    # get adjusted R squared
    adj_r_2 = 1 - (nom / denom)
    # return adj_r_2
    return adj_r_2

# derive RMSE
def get_rmse(y, y_hat, n):
    """
    Derive RMSE, a performance measure:
   
    Parameters
    ----------    
    y: dependent variable
    y_hat: predicted values of y 
    n: number of observations
    
    Returns
    -------
    rmse: scalar, to measure performance of model
    """
    # derive nominator
    nom = np.sum((y_hat - y)**2)
    # derive denominator
    denom = n
    # get rmse
    rmse = np.sqrt(nom / denom)
    # return rmse
    return rmse

# plot function to visualize the distribution of the residuals
def plot_residuals(residuals, n, color="blue", 
                   edgecolors="black", alpha=0.6):
    """
    A plot shwong the distribution of the residuals:
   
    Parameters
    ----------    
    residuals: vector containing difference of observed and predicted values
    n: number of observations
    color: color of plot compopnents
    edgecolor: color of components edges
    alpha:
    
    Returns
    -------
    plot: A plot visualizing the distribution of the residuals
    """
    # create a plot containing the distribution of the innovations
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    # create an index for plot
    index = np.arange(0, len(residuals))
    # 1. scatterplot
    axs[0].scatter(index, residuals, color=color, 
                   edgecolors=edgecolors, alpha=alpha)
    # add xlabel
    axs[0].set_xlabel("Index")
    # add ylabel
    axs[0].set_ylabel("Value")
    # 2.histogram
    axs[1].hist(residuals, color=color, edgecolor=edgecolors,
                linewidth=1.2, alpha=alpha,
                # number of bins equals the square root of total observations
                bins=int(np.sqrt(n)))
    # add xlabel
    axs[1].set_xlabel("Value")
    # add ylabel
    axs[1].set_ylabel("Frequency")
    # set a title to the figure
    fig.suptitle("Distribution of Residuals")
    # show plot
    plt.show()

    
# ----------------------------------------------------------------------------
# 1.2 Implementation using OOP
# ----------------------------------------------------------------------------


# Define class
class LinearRegressionModel:
    
    # constructor method 
    def __init__(self):
       pass
    # object representation method
    def __repr__(self):
        return "Instance of class 'LinearRegressionModel'."
    
    # fit/train mehtod to train the model 
    def fit(self, X, y):
        # Assign variables X and y
        self.X = X
        self.y = y
        self.n = X.shape[0]
        # Check X and y for NaN
        X_has_nan = np.isnan(X).any()
        y_has_nan = np.isnan(y).any()
        # stop the execution if arrays contain missing values
        if X_has_nan is True or y_has_nan is True:
            sys.exit("Make sure X and y do not contain nan's!")
        # if there are no missings continue running script
        else:
            # apply derive_beta function to get beta
            self.beta = get_beta(self.X, self.y, self.n)
            # number of estimated coefficients
            self.p = self.beta.shape[0]
            # create an empty dictionary
            coef = {}
            # loop over index and elements of beta
            for i, val in enumerate(self.beta):
                # Assign first element to Intercept...
                if i == 0:
                    coef["Intercept"] = float(np.round(val, 4))
                else:
                    # ...and the others to betas
                    coef[str(f"beta{i}")] = float(np.round(val, 4))
            # return vector of coefficients as a dictionary
            return coef
    
    # predict method 
    def predict(self, X_test=None):
        # Check X and y for NaN
        X_test_has_nan = np.isnan(X_test).any()
        # stop the execution if arrays contain missing values
        if X_test_has_nan is True:
            sys.exit("Make sure X does not contain nan's!")
        # if there are no missings continue running the script
        else:
            # predict y
            self.y_hat = get_predictions(X_test, self.beta, self.n)
            # derive residuals
            self.residuals = get_residuals(self.y, self.y_hat)
            # return predicted values of y
            return self.y_hat
        
    # score method 
    def score(self):
        # get R squared
        r_2 = get_r_2(self.y, self.y_hat)
        # get adjusted R squared
        adj_r_2 = get_adj_r_2(self.y, self.y_hat, self.beta, self.n, self.p)
        # Root mean squared error
        rmse = get_rmse(self.y_hat, self.y, self.n)
        # store measures of fit in a dictionary
        dic = {"R_2": np.round(r_2, 4), "adj_R_2": np.round(adj_r_2, 4),
               "RMSE": np.round(rmse, 4)}
        # return the dictionary
        return dic
    
    # plot method
    def plot(self, **kwargs):
        # plot distribution of residuals
        plot_residuals(self.residuals, self.n, **kwargs)
