
import numpy as np
import pandas as pd 
import random

def split_train_test(data: list, test_size: float=0.4):
 
    '''
        DESCRIPTION
        -----------
        divides data into a training set and a testing set
        test_size is used to define the % of data which should go into
        the testing set
 
        PARAMETERS
        ----------
        a list
 
        RETURNS
        ----------
        a tuple where the first element is the training set
        ans the second element is the testing set
    '''
    # % must be between 0.1 and 0.99
    if not 0.99 > test_size > 0.1:
        raise ValueError('test_size must be between 0.1 and 0.99')
    
    test_size_int = int(test_size * 100)
 
    train_data = []
    test_data = []
 
    for row in data:
        if random.randint(1, 100) <= test_size_int:
            test_data.append(row)
        else:
            train_data.append(row)
    
    return train_data, test_data

class LinearRegression(object):

    '''
        DESCRIPTION
        --------------

        Ordinary least squares Linear Regression class. 
        Uses gradient descend algorithm to find the best thetas. 
        The predictors are standardized before starting to find the best thetas (X - mu) / sigma

        If standardize is set to False then the thetas are de-standardized after the algorithm is done

        When using the 'predict' method, the matrix passed must contain 1's as the first column
    '''


    def __init__(self, standardize=False):
        self._standardize = standardize

    def predict(self, predictors: np.ndarray, coefficients: np.ndarray):
        '''
            PARAMETERS
            -------------------
            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors. 

            DESCRIPTION
            ----------------------
            Dot product of each row of predictors with coefficients
            equivalent to a matrix multiplication (N x M) * (M x 1) = (N x 1)

            RETURNS
            ----------------------
            a N x 1 matrix where each element is a dot product of each
            observation with the respective coefficient
        '''
        # return np.sum(coefficients  * predictors, axis=1)
        return np.matmul(predictors, coefficients)

    def sum_of_squared_errors(self, ys: np.ndarray, coefficients: np.ndarray, predictors: np.ndarray):
        '''
            PARAMETERS
            -------------------
            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors.

            ys: a N x 1 matrix of the dependent variables where N
            is the number of observations

            RETURNS
            --------------------
            the sum of squared errors
        '''
        return np.sum(self._errors(ys, coefficients, predictors)**2)

    def _sum_of_errors(self, ys: np.ndarray, coefficients: np.ndarray, predictors: np.ndarray):
        '''
            PARAMETERS
            -------------------
            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors.

            ys: a N x 1 matrix of the dependent variables where N
            is the number of observations

            RETURNS
            --------------------
            the sum of the errors
        '''
        return np.sum(self._errors(ys, coefficients, predictors))

    def _errors(self, ys: np.ndarray, coefficients: np.ndarray, predictors: np.ndarray):
        '''
            PARAMETERS
            -------------------
            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors.

            ys: a N x 1 matrix of the dependent variables where N
            is the number of observations

            RETURNS
            --------------------
            N x 1 matrix of each observation error
        '''
        predictions = self.predict(predictors, coefficients)
        return ys - predictions

    def _derivatives(self, ys: np.ndarray, coefficients: np.ndarray, predictors: np.ndarray, jth_column: int):
        '''
            PARAMETERS
            ------------
            ys: a N x 1 matrix of the dependent variables where N is the number of observations

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors.

            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            jth_column: the index of the jth column of predictors

            RETURNS
            ----------
            a N x 1 matrix of the partial derivatives with respect to the jth coefficient
        '''

        # calculate the errors
        errors = self._errors(ys, coefficients, predictors)
        # extract jth column from predictors, reshape to N x 1
        predictors_column = predictors[::, jth_column].reshape(-1, 1)
        # calculate the derivatives with respect to jth column:
        # 2 * -xj * (y - (b0*1 + b1*x1 + .... + bn*xn))
        #return 2 * -predictors_column * errors
        return -predictors_column * errors

    def _new_coefficients(self, ys: np.ndarray, coefficients: np.ndarray, predictors: np.ndarray, learning_rate: float):
        '''
            PARAMETERS
            ------------
            ys: a N x 1 matrix of the dependent variables where N is the number of observations

            coefficients: a M x 1 matrix where M is the number of
            attributes of predictors.

            predictors: a N x M matrix where N is the number of observations
            and M is the number of attributes. The values in the first column
            are all 1's which are multiplied against the intercept

            learning_rate: the learning rate

            RETURNS
            ----------
            a N x 1 matrix of the new calculate coefficients
        '''
        new_coefficients = []
        # iterate through the coefficients
        for index, coefficient in enumerate(coefficients, 0):
            # calculate all the derivatives with respect to the cofficient at position Index
            derivatives = self._derivatives(ys, coefficients, predictors, index)
            # calculate new coefficient
            new_coeffient = coefficient[0] - learning_rate * np.mean(derivatives)
            new_coefficients.append([new_coeffient])
            
        return np.array(new_coefficients)
    
    def _standardize_matrix(self, matrix:np.ndarray, mus:float, sigmas:float):
        '''
            DESCRIPTION
            ------------
            Standardize a matrix: (X - mu) / sigma 

            PARAMETERS
            -----------
            matrix: A np.ndarray of size N x M
            mus:    a 1 x M np.ndarray containing the means
            sigmas: a 1 x M np.ndarray containing the standard deviations

            RETURNS
            -----------
            The modified matrix
        '''

        return (matrix - mus) / sigmas

    def _destandardize_matrix(self, matrix:np.ndarray, mu:float, sigma: float):
        '''
            DESCRIPTION
            ------------
            de-standardize a matrix: X * sigma + mu

            PARAMETERS
            -----------
            matrix: A np.ndarray of size N x M
            mus:    a 1 x M np.ndarray containing the original means
            sigmas: a 1 x M np.ndarray containing the original standard deviations

            RETURNS
            -----------
            The modified matrix
        '''
        return matrix * sigma + mu

    def _standardize_predictors(self, predictors:np.ndarray):
        '''
            standardizes the predictors, stores the original means and standard deviations
        '''
        self._predictors_mus = predictors.mean(axis=0)
        self._predictors_sigmas = predictors.std(axis=0)
        return self._standardize_matrix(predictors, self._predictors_mus, self._predictors_sigmas)

    def _standardize_ys(self, ys:np.ndarray):
        '''
            standardizes the ys and stores the original means and standard deviations
        '''
        self._ys_mus = ys.mean(axis=0)
        self._ys_sigmas = ys.std(axis=0)
        return self._standardize_matrix(ys, self._ys_mus, self._ys_sigmas)

    def _destandardize_thetas(self):

        # de standardize the intercept
        self.thetas[0] = self.thetas[0] * self._ys_sigmas + self._ys_mus - \
                            np.sum((self.thetas[1:]  * self._predictors_mus.reshape(-1,1) * self._ys_sigmas) / self._predictors_sigmas.reshape(-1,1))

        # de standardize the coefficients
        self.thetas[1:] = (self.thetas[1:] * self._ys_sigmas)  / self._predictors_sigmas.reshape(-1,1)




    @property
    def intercept(self):
        return self.thetas[0]
    
    @property
    def coefficients(self):
        return self.thetas[1:]
    
    def R_squared(self, ys:np.ndarray, predictors:np.ndarray):
        return 1 -  self.sum_of_squared_errors(ys, self.thetas, predictors) / np.sum((ys - np.mean(ys, axis=0)) ** 2)
 

    def fit(self,
              predictors: np.ndarray,
              ys: np.ndarray,
              learning_rate=0.1,
              max_iters=10000):

        '''
            DESCRIPTION
            -----------
            finds the coefficients of a linear regression

            PARAMETERS
            ----------
            predictors: a N x M array whose first column is a column 1's 

            ys: a N x 1 matrix of the observed values

            learning_rate: the learning rate applied to the gradient descent algorithm, try smaller values if function does not converge

            max_iters: the maximum number of iterations used to find the coefficients
        '''

        # store xs shape
        xs_rows, xs_columns = predictors.shape

        # add column of 1's to predictors
        self._predictors = predictors

        self._ys = ys

        # start with random coefficients
        self.thetas = np.random.randn(xs_columns, 1)

        # standardize predictors (except for 1st column) and ys
        self._predictors[::, 1:] = self._standardize_predictors(self._predictors[::, 1:])
        self._ys = self._standardize_ys(self._ys)

        # get value of cost function
        # ((b0 + b1 * X1 + .... + bn * Xn) - y) / (2N)
        squared_errors = self.sum_of_squared_errors(self._ys, self.thetas, self._predictors) / (2 * xs_rows)

        for i in range(max_iters):

            # calculate new coefficients
            self.thetas = self._new_coefficients(self._ys, self.thetas, self._predictors, learning_rate)

            # new value of cost function
            new_squared_errors = self.sum_of_squared_errors(self._ys, self.thetas, self._predictors) / (2 * xs_rows)

            # if precision reached, return coefficients
            if new_squared_errors == squared_errors:
                break

            squared_errors = new_squared_errors
        
        # de standardize thetas if normalize is set to False
        if self._standardize == False:
            self._destandardize_thetas()
            self._ys = self._destandardize_matrix(self._ys, self._ys_mus, self._ys_sigmas)
            self._predictors[::, 1:] = self._destandardize_matrix(self._predictors[::, 1:], self._predictors_mus, self._predictors_sigmas)