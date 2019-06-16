from dnn import *
import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
import pylab


np.random.seed(12521)
D = 20 # Extrinsic dimension
d = 2 # Intrinsic dimension
X_test_lat = np.random.uniform(-1, 1, size=(1, d)) # Single test point in D dimensions
A = np.random.uniform(-1, 1, size=(d, D)) # Matrix of projection onto d dimensional subspace
fn = lambda x: scipy.special.expit(3*x[:, [0]]) # Regression function
save_results = 1 # Save figures or not

X_test = np.matmul(X_test_lat, A) # Final (projected) test point
mu_test = fn(X_test).flatten() # Target f(x)
n = 20000 # Number of training points
B = None # Number of subsamples. If None, complete setting.
sigma_e = 1 # Noise Std.
k = [1,2,5] # Values of k tried for k-NN
estimate_sigma_e = 0 # To estimate sigma_e or not
w_mat = [] # Keeps matrix of weights for various k's
w_max = [] # Keeps matrix of maximum distance to k-NN, for estimating \epsilon(s)

if estimate_sigma_e == 0:
    for j in range(len(k)):
        W = DNN_weights(n, k=k[j])
        sigma_use = sigma_e
        w_mat.append(W.find_weights())
        w_max.append(W.find_eps_weight())
else:
    for j in range(len(k)):
        W = DNN_weights(n//2, k=k[j])
        sigma_use = None
        w_mat.append(W.find_weights())
        w_max.append(W.find_eps_weight())

mc_samples = 1000 # Number of Monte-Carlo iterations
est_lst = []
cov = np.zeros(len(k))
est_lst = np.zeros((len(k),mc_samples))
asym_std = np.zeros(len(k))
s_pick = np.zeros((len(k),mc_samples))
for i in range(mc_samples):
    print('Round {} started'.format(i))
    X_lat = np.random.uniform(-1, 1, size=(n, d))
    X = np.matmul(X_lat, A)
    mu = fn(X)
    y = mu + np.random.normal(0, sigma_e, size=(n, 1))
    for j in range(len(k)):
        dnn_obj = DNN(sigma_e=sigma_use, w = w_mat[j], max_w = w_max[j], k = k[j])
        dnn_obj.fit(X, y)
        est, est_low, est_up, s_vals, asym_std[j] = dnn_obj.predict_conf(X_test, conf=0.98)
        est_lst[j, i] = est
        s_pick[j,i] = s_vals
        cov[j] += (1.0 * (mu_test >= est_low) * (mu_test <= est_up) / mc_samples)[0]

print("Coverage:", cov)
print("Test point:", X_test[0, 0])
print("True value:", fn(X_test)[0, 0])
print("Std:", np.std(est_lst, axis = 1))
print("Bias:", (fn(X_test) - np.mean(est_lst, axis = 1)))


params = {'legend.fontsize': 'small',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'lines.markersize': 5}

plt.rcParams.update(params)



fig = plt.figure(figsize=(15,15))
for j in range(len(k)):
    plt.rcParams.update({'markers.fillstyle': 'full'})
    plt.subplot(len(k),2,2*j+1)
    plt.title("Distribution of estimates for $k = {}$, Mean: ${:.3f}$, Std: ${:.3f}$".format(k[j], np.mean(est_lst[j,:]), np.std(est_lst[j,:])))
    plt.hist(est_lst[j,:], weights=np.ones(mc_samples)/mc_samples)
    plt.ylabel("Fraction of iterations")
    plt.xlabel("Estimate")
    plt.subplot(len(k),2,2*j+2)
    (osm, osr), (coef, intercept, r) = scipy.stats.probplot(est_lst[j,:], dist="norm", sparams=(fn(X_test)[0, 0], asym_std[j]), plot=pylab)
    plt.rcParams.update({'markers.fillstyle': 'none'})
    plt.plot(osm, osm, label='$f(x)=x$', color = 'red', lw=3.5, ls = "dashed", marker='*', ms = 14, markevery=(0.05,0.1))
    plt.plot(osm, coef*osm+intercept, label='linear fit', color='green', lw=2.5, marker='^', ms=14, markevery=0.1)
    plt.title('Probability plot for $k = {}$'.format(k[j]))
    plt.legend()

plt.subplots_adjust(hspace = 0.5, left = 0.05, right = 0.95)

if save_results ==1:
    h = "coverage_single_point_n_{:d}_sigma_{:.2f}_mcsamples_{:d}".format(n, sigma_e,mc_samples)
    plt.savefig(h+".png",dpi=600, format="png")
    plt.savefig(h+".pdf",dpi=600, format="pdf")

plt.show()

