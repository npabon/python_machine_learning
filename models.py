import numpy as np

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters 
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over training dataset (epochs)
    shuffle : bool (default: True)
      Shuffles training data every epoch if True
      to prevent cycles
    random_state : int
      Random number generator seed for random weight initialization

    
    Attributes 
    ------------
    w_ : 1d-array
      Weights after fitting
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch

    """

    def __init__(self, eta=0.01, n_iter=10, 
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like} shape = [n_samples, n_features]
          Training samples, where n_samples is the number
          of samples and n_features is the number of features
        y : array-like, shape = [n_samples]
          Target values.
          
        Returns
        ---------
        self : object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1: # multiple samples
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else: # single sample
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle the training data"""
        # ** USEFUL ** remember how to do this
        r = self.rgen.permutation(len(y)) # rgen is set when weights initialized
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        # if self.random_state is None, uses a random seed
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True


    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights
           and return the cost"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error) # scalar vector multiplication
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input to neuron"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters 
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over training dataset (epochs)
    random_state : int
      Random number generator seed for random weight initialization

    
    Attributes 
    ------------
    w_ : 1d-array
      Weights after fitting
    cost_ : list
      Sum-of-squares cost function value in each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like} shape = [n_samples, n_features]
          Training samples, where n_samples is the number
          of samples and n_features is the number of features
        y : array-like, shape = [n_samples]
          Target values.
          
        Returns
        ---------
        self : object
        """

        # draw random samples from a normal distribution
        rgen = np.random.RandomState(self.random_state)
        # loc = mean, scale = st.dev., size = number of samples
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output) # a vector
            self.w_[1:] += self.eta * X.T.dot(errors) # make sure this makes sense
            self.w_[0] += self.eta * errors.sum() # b/c all x_0 are 1 by construction
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input to neuron"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class Perceptron(object):
    """Perceptron classifier.
    
    Parameters **(the same for all members of the class)
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over training dataset (epochs)
    random_state : int
      Random number generator seed for random weight initialization
    
    
    Attributes **(specific to each class member)
    ------------
    w_ : 1d-array
      Weights after fitting
    errors_ : list
      Number of misclassifications (updates) in each epoch
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like} shape = [n_samples, n_features]
          Training samples, where n_samples is the number
          of samples and n_features is the number of features
        y : array-like, shape = [n_samples]
          Target values.
          
        Returns
        ---------
        self : object
        """
        
        # draw random samples from a normal distribution
        rgen = np.random.RandomState(self.random_state)
        # loc = mean, scale = st.dev., size = number of samples
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y): # pairs each sample with its label
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update # bias update
                self.w_[1:] += update * xi
                errors += int(update != 0) # number of misclassifications
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input to perceptron"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return predicted class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) # {[-1.],[1.]}
        
        













        
        
        
        