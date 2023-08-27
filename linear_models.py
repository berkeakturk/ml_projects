import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss = 0

    def fit(self, X, y):
        n_observations, n_features = self.__init_params(X)
        prev_loss = 1.e10
        y = y.reshape(y.shape[0],1)
        for epoch in range(self.epochs):
            y_hat = np.dot(X, self.weights) + self.bias            
            dW = (-2/n_observations) * np.dot(X.T, (y - y_hat))
            db = (-2/n_observations) * np.sum(y - y_hat)
            self.__update_params(dW, db)
            if np.abs(prev_loss - self.loss) <= 1.e-4:
                print("The Algorithm Has Converged")
                break
            prev_loss = self.loss
            self.loss = self.compute_loss(y, y_hat)            
            if epoch % 50 == 0:
                print(f"Epoch = {epoch} , MSE = {self.loss}")

    def __init_params(self, X):
        n_observations, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        return n_observations, n_features
    
    def __update_params(self, dW, db):
        self.weights = self.weights - self.lr * dW
        self.bias = self.bias - self.lr * db
    
    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        return predictions
    
    def compute_loss(self, y, y_hat):
        return np.mean((y - y_hat)**2)
    

            
class LogisticRegression():
    def __init__(self, lr=0.001, epochs = 1000, threshold=0.5):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def __init_params(self, X):
        n_observations, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        return n_observations, n_features
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def fit(self, X, y):
        n_observations, n_features = self.__init_params(X)
        y = y.reshape(y.shape[0],1)
        for epoch in range(self.epochs):
            linear_predictions = np.dot(X, self.weights) + self.bias
            probabilities = self.sigmoid(linear_predictions)
            predictions = np.array([0 if prob <= self.threshold else 1 for prob in probabilities]).reshape(y.shape[0],1)
            dW = - (1/n_observations) * np.dot(X.T, (y - probabilities))
            db = - (1/n_observations) * np.sum(y - probabilities)
            self.__update_params(dW, db)
            if epoch % 50 == 0:
                print(f"Epoch = {epoch} , Accuracy = {self.accuracy(y, predictions)}")

    def __update_params(self, dW, db):
        self.weights = self.weights - dW * self.lr
        self.bias = self.bias - db * self.lr

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_predictions)
        predictions = [0 if prob <= self.threshold else 1 for prob in probabilities]
        return predictions
    
    def accuracy(self, y_true, y_hat):
        return np.sum(y_true == y_hat) / len(y_true)

     








