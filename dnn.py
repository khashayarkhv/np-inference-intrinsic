import numpy as np
import scipy.special
import scipy.stats


class DNN_weights:
    """
    This class computes weights that are used in the DNN algorithm. It has been
    implemented seperately from the original DNN class, so as to allow faster
    monte-carlo simulations (weights only depend on values of n, k, and s and not
    the training set)
    """
    def __init__(self, n, k=1):
        """
        n : number of samples
        k (optional) : number of nearest neighbors in each subsample
        """
        self.n = n
        self.k = k
    
    def find_weights(self):
        """
        Computes an n by n matrix containing weights for different values of s.
        n : number of samples
        k (optional) : number of nearest neighbors in each subsample
        """

        weights = np.zeros((self.n,self.n))
        if self.k == 1:
            for j in range(self.n):
                s = j+1
                weights[0,j] = s/self.n
                for l in np.arange(2, self.n-s+2):
                    weights[l-1,j] = weights[l-2,j] * (self.n-s+2-l) / (self.n+1-l)
                    if weights[l-1,j]/np.amax(weights[:l-1,j]) < 1e-5:
                        break
        else:
            for j in range(self.n):
                a_matrix = np.zeros((self.n,self.k))
                s = j+1
                for l in np.arange(self.k):
                    weights[l, j] = s/(self.n * self.k)
                    a_matrix[self.k-1,l] = (1/self.k) * scipy.special.comb(self.k-1,l)
                    for u in np.arange(l+1):
                        a_matrix[self.k-1,l] *= (s-u)/(self.n-u)
                    for u in np.arange(self.k-l-1):
                        a_matrix[self.k-1,l] *= (self.n-s-u)/(self.n-l-1-u)
                for l in np.arange(self.k+1, self.n-s+2):
                    for u in np.arange(self.k):
                        a_matrix[l-1,u] = a_matrix[l-2,u] * (l-1)/(l-1-u) * (self.n-l+1-s+(1+u))/(self.n-l+1)
                    weights[l-1,j] = np.sum(a_matrix[l-1,:])
                    if weights[l-1,j]/np.amax(weights[:l-1,j]) < 1e-5:
                        break
        return weights

    def find_eps_weight(self):
        """
        Computes an n by n matrix containing weights for different values of s.
        n : number of samples
        k (optional) : number of nearest neighbors in each subsample
        """
        weights = np.zeros((self.n,self.n))
        for j in np.arange(self.k-1,self.n):
            s = j+1
            weights[self.k-1,j] = s/self.n
            for l in np.arange(1, self.k):
                weights[self.k-1,j] *= (s-l)/(self.n-l)
            for l in np.arange(self.k+1, self.n-s+1+self.k):
                weights[l-1,j] = weights[l-2,j] * (l-1)/(l-self.k) * (self.n-s+self.k+1-l)/(self.n+1-l)
                if weights[l-1,j]/np.amax(weights[:l-1,j]) < 1e-5:
                    break
        return weights    


class DNN:
    def __init__(self, w, max_w, k=1, s=None, B=None, sigma_e=None, gamma = 0.1):
        """
        s (optional) : subsampling rate, if None then it is calculated adaptive.
        B (optional) : number of subsamples, if None then complete U-statistic used
        sigma_e (optional) : an external estimate of the residual variance to be
            used as plugin, if None variance is estimated
        k (optional) : number of nearest neighbors in each subsample
        """
        self._s = s
        self._B = B
        self._sigma_e = sigma_e
        self._k = k
        self._gamma = gamma
        self._zeta_k = self.k_coeff()
        self._w = w
        self._maxw = max_w
        return


    def fit(self, X, y):
        """ Stores the training data and if an external estimate of the residual variance is
        not available then it estimates residual variance.
        X : features
        y : outcomes
        """
        y = y.flatten()
        
        # estimate residual variance if not externally given
        if self._sigma_e is None: 
            n = X.shape[0]
            # We predict on the second half of the data using only the first half
            # This is to avoid overfitting and underestimating the variance.
            # In this case, we do not use the second part of data on the estimation
            # on the test points.
            self._X = X[:n//2]
            self._y = y[:n//2]
            y_pred = self._predict_sigma(X[n//2:])
            # We calculate residual variance on the second half
            sigma_e = np.std(y[n//2:] - y_pred, ddof=1)
        else:
            self._X = X
            self._y = y
            sigma_e = self._sigma_e
        self.sigma_e_est = sigma_e

    def comp_asym_std(self, s, n):
        """ 
        Estimates the asymptotic variance of DNN estimator for the k-NN kernel using
        the first-order approximation that is provided in the paper.
        s: sub-sample size
        n: sample size
        """
        return np.sqrt((self._zeta_k * s**2 * self.sigma_e_est**2)/(n * (2*s-1)))

    def _predict_sigma(self, X):
        """ 
        Predicts sigma_e by dividing data into two parts and predicting on the
        second half of data using samples on the first half. The implementation
        is similar to _predict function that is described next.
        """
        y_pred = np.zeros(X.shape[0])
        n = self._X.shape[0]
        for i in range(X.shape[0]):
            x0 = X[i]
            if self._s is None:
                s_estimated = 9*self.predict_s_bin(x0, delta = 1/n)+1
            else:
                s_estimated = self._s
            s_to_use = min(s_estimated, n)
            if self._B is None: # full statistics
                inds = np.argsort(np.linalg.norm(self._X - x0.reshape(1, -1), axis=1))
                y_pred[i] = np.dot(self._y[inds], self._w[:,s_to_use-1])
            else: # incomplete statistics
                w = np.zeros(self._X.shape[0])
                for _ in range(self._B):
                    inds = np.random.choice(np.arange(self._X.shape[0]), s_to_use, replace=False)
                    nn = np.argmin(np.linalg.norm(self._X[inds] - x0.reshape(1, -1), axis=1), axis=0)
                    w[inds[nn]] += 1.0/self._B
                y_pred[i] = np.dot(self._y, w)
        return y_pred

    

    def _predict(self, X, X_train, y_train):
        """ Draws subsamples and finds nearest neighbor on each subsample. Then returns
        weighted average as prediction. If B is None then a closed form weight is used
        that corresponds to the complete U statistic. Also, if s is not provided, it
        estimates s using the adaptive process explained in the paper.
        This is an internal helper function that allows one to choose explicitly the training data.
        X : feature vectors to predict on
        X_train : feature vectors to use for training
        y_train : labels of training vectors
        """
        y_pred = np.zeros(X.shape[0])
        n = X_train.shape[0]
        s_vals = np.zeros(X.shape[0])
        asymp_var = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x0 = X[i]
            if self._s is None:
                s_estimated = 9*self.predict_s_bin(x0, delta = 1/n) + 1
                s_to_use = int(s_estimated * n**(self._gamma))
                s_to_use = min(s_to_use, n)
            else:
                s_estimated = self._s
                s_to_use = min(s_estimated,n)
            s_vals[i] = s_to_use
            asymp_var[i] = self.comp_asym_std(s_to_use,n)
            if self._B is None: # full statistics
                weights = self._w[:,s_to_use-1]
                inds = np.argsort(np.linalg.norm(X_train - x0.reshape(1, -1), axis=1))
                y_pred[i] = np.dot(y_train[inds], weights)
            else: # incomplete statistics
                w = np.zeros(X_train.shape[0])
                for _ in range(self._B):
                    inds = np.random.choice(np.arange(X_train.shape[0]), s_to_use, replace=False)
                    nn = np.argmin(np.linalg.norm(X_train[inds] - x0.reshape(1, -1), axis=1), axis=0)
                    w[inds[nn]] += 1.0/self._B
                y_pred[i] = np.dot(y_train, w)

        return y_pred, s_vals, asymp_var


    
    def predict(self, X):
        """ Predicts on feature points X using all the available labeled data.
        """
        return self._predict(X, self._X, self._y)
    
    def predict_conf(self, X, conf=.98):
        """ Predicts on feature points and also calculates confidence interval
        for each point, using the asymptotic normal approximation.
        """
        y_pred, s_vals, asymp_var = self.predict(X)
        conf_id = np.zeros(len(y_pred))
        # confidence interval based on normal approximation
        for i in range(len(y_pred)):
            conf_id[i] = scipy.stats.norm.ppf(1-(1.0-conf)/2, loc=0, scale=asymp_var[i])
        return y_pred, y_pred - conf_id, y_pred + conf_id, s_vals, asymp_var

    def k_coeff(self):
        """ 
        Computes zeta_k/k^2, the term that appears in the asymptotic variance of
        the sub-sampled k-NN estimator.
        """
        zeta_k = self._k
        for t in range(self._k, 2*self._k-1):
            ins_term = 0
            for i in range(t-self._k+1, self._k):
                ins_term += 2**(-t) * scipy.special.comb(t,i)
            zeta_k += ins_term
        return zeta_k/(self._k**2)

    def predict_s_bin(self, x_test, delta):
        """
        This function finds s_1, the value that is returned by
        the adaptive process in the paper. It uses binary search
        for finding this value and has time complexity of O(log(n)),
        where n is the number of training samples.
        x_test: target point
        delta: error allowed in the estimation process. This is usually
        set to 1/n in the other functions.
        """
        n = self._X.shape[0]
        C = 2*np.log(2*n/delta)
        G_up = np.sqrt(C)
        G_low = np.sqrt(C/n)
        dist = np.sort(np.linalg.norm(self._X-x_test, axis=1))
        H_up = np.dot(dist,self._maxw[:,n-1])/dist[n-1]
        H_low = np.dot(dist,self._maxw[:,self._k-1])/dist[n-1]
        up = n
        low = self._k
        if(H_up >= 2*G_up):
            return n
        else:
            while up-low>=2:
                tmp = int(np.floor((low+up)/2))
                H_tmp = np.dot(dist,self._maxw[:,tmp-1])/dist[n-1]
                G_tmp = np.sqrt(C*tmp/n)
                if H_tmp < 2*G_tmp:
                    up = tmp
                    H_up = H_tmp
                    G_up = G_tmp
                else:
                    low = tmp
                    H_low = H_tmp
                    G_up = G_tmp
            return up