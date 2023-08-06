import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

class LinearRegression():
    def __init__(self):
        self.design_matrix = None
        self.actual = None
        self.coefficient_names = None
        self.coefficients = None
        self.fitted = None
        self.residuals = None
        self.n = None
        self.p = None
        self.dof = None
        self.mse = None
        self.se = None
        
    def fit(self, X, y):
        self.coefficient_names = list(pd.DataFrame(X).columns)
        self.coefficient_names.insert(0, 'Intercept')
        if len(X.shape) == 1:
            X = np.array(X).reshape(len(X),1)
        self.design_matrix = np.hstack((np.ones((X.shape[0], 1)), X))
        self.actual = y
        self.coefficients = np.linalg.inv(self.design_matrix.T.dot(self.design_matrix)).dot(self.design_matrix.T).dot(y)
        self.fitted = self.design_matrix.dot(self.coefficients)
        self.residuals = y - self.fitted
        # t-test for significance
        self.n = self.design_matrix.shape[0]
        self.p = self.design_matrix.shape[1] - 1
        self.dof = self.n - self.p - 1

        self.mse = np.sum(self.residuals**2) / self.dof  # Mean squared error (MSE)
        self.se = np.sqrt(np.diagonal(self.mse * np.linalg.inv(np.dot(self.design_matrix.T, self.design_matrix))))

        self.t_values = self.coefficients / self.se
        self.p_values = 2 * (1 - t.cdf(np.abs(self.t_values), self.dof))

        
    def predict(self, X):
        X_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = X_intercept.dot(self.coefficients)
        return y_pred
    
    def summary(self, significant_threshold):
        summary_df = pd.DataFrame({'Coefficient Name': self.coefficient_names,
                                   'Coefficient Value': self.coefficients,
                                   'Standard Error' : self.se,
                                    'T-Value' : self.t_values,
                                    'P-Value' : self.p_values
                                    })
        summary_df['is_significant'] = (summary_df['P-Value'] < significant_threshold)
        print(summary_df)
    
    def plot_regression_summary(self, figsize=(12,8)):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
        n = range(len(self.actual))
        axs[0].plot(n, self.actual, color='red')
        axs[0].plot(n, self.fitted, color='blue')
        axs[0].set_title('Actual vs Fitted')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Value')
        axs[0].legend(['Actual', 'Fitted'])

        axs[1].scatter(self.fitted, self.residuals)
        axs[1].plot(self.fitted, [0 for i in n], color="red")
        axs[1].set_title('Fitted vs Residuals')
        axs[1].set_xlabel('Actual')
        axs[1].set_ylabel('Residuals')

        axs[2].plot(n, self.residuals, color = 'blue')
        axs[2].plot(n, [0 for i in n], color="red")
        axs[2].set_title('Residual Plot')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Residual')
        
        plt.suptitle('Regression Summary Plots')
        plt.subplots_adjust(hspace=0.4)
        plt.show()


    



    