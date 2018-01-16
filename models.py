import numpy as np
import scipy
import itertools
from sklearn.ensemble import RandomForestClassifier



class LincsRandomForestClassifier(object):
    
    "WE ASSUME THE DATA IS GROUPED BY CELL LINE AND HAS 4 FEATURES PER CELL LINE"
   
    def __init__(self, n_cells_per_forest, n_estimators_per_forest=10, max_depth=None, max_features="auto", random_state=1):
        self.n_cells_per_forest = n_cells_per_forest
        self.n_estimators_per_forest = n_estimators_per_forest
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        
    def fit(self, X, y):
        '''
        Train several random forests, each one on a different
        subset of cells. Store forests in a dictionary called
        self.forests.
        '''
        # make sure we have enough data to work with
        min_num_cells = self.get_min_num_cells(X)
        assert min_num_cells >= self.n_cells_per_forest, "Too much missing data for n_cells_per_forest = %s. (Some samples only tested in %d cells)" % \
                                                         (self.n_cells_per_forest, min_num_cells)
        
        # generate cell subsets for training
        # ASSUMES 4 FEATURES PER CELL
        total_num_cells = int(X.shape[1] / 4) # THIS IS HARDCODED IN
        cell_subsets = itertools.combinations(np.arange(total_num_cells), self.n_cells_per_forest)
        
        # initialize dictionary to hold the forests
        self.forests = {}
        
        # train forest on each subset
        for cell_subset in cell_subsets:
            
            # find samples that have complete data from the cell subset
            cell_subset_idx = np.array([ 4*i + np.array([0, 1, 2, 3])for i in cell_subset ]).reshape(1,-1)[0].astype(int)
            cell_subset_data = X[:,cell_subset_idx]
            bad_sample_idx = np.isnan(cell_subset_data).any(axis=1)
            good_samples = cell_subset_data[~bad_sample_idx]
            good_labels = y[~bad_sample_idx]
            
            # train and store a RF classifier on this training subset
            # print('Growing forest for cell subset: %s' % str(cell_subset))
            forest = RandomForestClassifier(criterion='gini',
                                            n_estimators=self.n_estimators_per_forest,
                                            max_depth=self.max_depth,
                                            max_features=self.max_features,
                                            random_state=self.random_state,
                                            n_jobs=-1)
            forest.fit(good_samples, good_labels)
            self.forests[cell_subset] = forest            

        
    def get_min_num_cells(self, X):
        '''
        Calculate the minimum number of cells any sample has data for
        ASSUMES 4 FEATURES PER CELL LINE
        '''
        X_not_missing = ~np.isnan(X)
        num_cells_not_missing = np.count_nonzero(X_not_missing, axis=1) / 4
        min_num_cells = np.min(num_cells_not_missing)
        return min_num_cells
    
    def predict_proba(self, X):
        '''
        Return the class probabilities label OF ONE SINGLE SAMPLE FOR FUCKS SAKE
        '''
        # figure out which cell lines we have data for
        non_nan_idx = np.where(np.isnan(X) == False)[0]
        good_cells = (non_nan_idx[np.where(non_nan_idx/4%1 == 0)[0]] / 4).astype(int)
        
        # select appropriate forests and predict
        cell_subsets = itertools.combinations(good_cells, self.n_cells_per_forest)
        tree_predictions_ = []
        for cell_subset in cell_subsets:
            # extract appropriate data
            cell_subset_idx = np.array([ 4*i + np.array([0, 1, 2, 3])for i in cell_subset ]).reshape(1,-1)[0].astype(int)
            cell_subset_data = X[cell_subset_idx].reshape(1,-1) 
            # extract appropriate forest and make prediction
            forest = self.forests[cell_subset]
            tree_predictions = [ tree.predict(cell_subset_data) for tree in forest.estimators_ ]
            tree_predictions_.append(tree_predictions)
        
        # majority vote of all the trees in all the forests
        results = np.array(tree_predictions_).flatten()
        proba = results.sum() / len(results)
        return np.array([1.-proba, proba])
    
    def predict(self, X):
        '''
        Return the predicted class label OF ONE SINGLE SAMPLE FOR FUCKS SAKE
        '''
        class_probabilities = self.predict_proba(X)
        return np.argmax(class_probabilities)
    
    def predict_proba_(self, X):
        '''
        for a multidimentional X
        '''
        proba_ = np.array([ self.predict_proba(x) for x in X ])
        return proba_
    
    def predict_(self, X):
        '''
        for a multidimentional X
        '''
        predicted_classes = np.array([ self.predict(x) for x in X ])
        return predicted_classes


class LogisticRegressionGD(object):
    """Logistic Regression classifier using gradient
    descent.

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

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number
          of samples and n_features is the number of features
        y : array-like, shape = [n_samples]
          Target values.
          
        Returns
        ---------
        self : object
        """

        # initialize weights
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output) # a vector
            self.w_[1:] += self.eta * X.T.dot(errors) # make sure this makes sense
            self.w_[0] += self.eta * errors.sum() # b/c all x_0 are 1 by construction
            
            # note that we compute the logistic 'cost' now,
            # which is the negative log-likelihood
            cost = (-y.dot(np.log(output)) -
                    ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input to neuron"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z,-250,250)))

    def predict(self, X):
        """Return class label"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X))
        #                 >= 0.5, 1, 0)


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
        
        













        
        
        
        