from dnn import *
import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt


np.random.seed(12521)
D = 20 # Extrinsic dimension
d = 2 # Intrinsic dimension
A = np.random.uniform(-1, 1, size=(d, D))
n_test = 100 # Number of test points
n = 20000 # Number of training points
B = None # Number of subsamples. If None, complete setting.
sigma_e = 1 # Noise Std.
fn = lambda x: scipy.special.expit(3*x[:, [0]]) # Regression function
estimate_sigma_e = 0 # Estimate sigma_e or not
save_results = 1 # Save figures or not

k = [1,2,5] # Values of k tried for k-NN

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

est_th_in = np.zeros((n_test, len(k)))
est_th_ex = np.zeros((n_test, len(k)))
est_prop = np.zeros((n_test, len(k)))

est_th_in_up = np.zeros((n_test, len(k)))
est_th_ex_up = np.zeros((n_test, len(k)))
est_prop_up = np.zeros((n_test, len(k)))

est_th_in_low = np.zeros((n_test, len(k)))
est_th_ex_low = np.zeros((n_test, len(k)))
est_prop_low = np.zeros((n_test, len(k)))


X_test_lat = np.random.uniform(-1, 1, size=(n_test, d))
X_test = np.matmul(X_test_lat, A)
X_test = X_test[np.argsort(X_test[:,0])]
mu_test = fn(X_test).flatten()


mc_samples = 1000 # Number of Monte-Carlo iterations

s_pick = np.zeros((n_test, mc_samples, len(k)))
cov_in = np.zeros((n_test,len(k)))
cov_ex = np.zeros((n_test,len(k)))
cov_prop = np.zeros((n_test,len(k)))

for i in range(mc_samples):
    print('Round {} started'.format(i))
	X_lat = np.random.uniform(-1, 1, size=(n, d))
	X = np.matmul(X_lat, A)
	mu = fn(X)
	y = mu + np.random.normal(0, sigma_e, size=(n, 1))
	for j in range(len(k)):
		s = int(n**(1.05*d/(d+2)))
		dnn_obj = DNN(sigma_e = sigma_use, s=s, w = w_mat[j], max_w = w_max[j], k = k[j])
		dnn_obj.fit(X, y)
		est, est_low, est_up, _, _ = dnn_obj.predict_conf(X_test, conf=0.98)
		est_th_in[:,j] = est
		est_th_in_up[:,j] = est_up
		est_th_in_low[:,j] = est_low
		s = int(n**(1.05*D/(D+2)))
		dnn_obj = DNN(sigma_e = sigma_use, s=s, w = w_mat[j], max_w = w_max[j], k = k[j])
		dnn_obj.fit(X, y)
		est, est_low, est_up, _, _ = dnn_obj.predict_conf(X_test, conf=0.98)
		est_th_ex[:,j] = est
		est_th_ex_up[:,j] = est_up
		est_th_ex_low[:,j] = est_low
		dnn_obj = DNN(sigma_e=sigma_use, w = w_mat[j], max_w = w_max[j], k = k[j])
		dnn_obj.fit(X, y)
		est, est_low, est_up, s_vals, _ = dnn_obj.predict_conf(X_test, conf=0.98)
		est_prop[:,j] = est
		est_prop_up[:,j] = est_up
		est_prop_low[:,j] = est_low
		s_pick[:, i, j] = s_vals
		cov_in[:,j] += (1.0 * (mu_test >= est_th_in_low[:,j]) * (mu_test <= est_th_in_up[:,j]) / mc_samples)
		cov_ex[:,j] += (1.0 * (mu_test >= est_th_ex_low[:,j]) * (mu_test <= est_th_ex_up[:,j]) / mc_samples)
		cov_prop[:,j] += (1.0 * (mu_test >= est_prop_low[:,j]) * (mu_test <= est_prop_up[:,j]) / mc_samples)


fig = plt.figure(figsize=(15,15))

params = {'legend.fontsize': 'small',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

for j in range(len(k)):
	plt.subplot(len(k),5,5*j+1)
	plt.title("$k = {}$, Adaptive with $\zeta=0.1$".format(k[j]))
	plt.fill_between(X_test[:,0], est_prop_low[:,j], est_prop_up[:,j], alpha=0.4, label="Confidence intervals", color='gray')
	plt.scatter(X_test[:,0], fn(X_test).flatten(), label="True")
	plt.legend(loc='upper left')

	plt.subplot(len(k),5,5*j+2)
	plt.title("$k= {}$, Theoretical for d".format(k[j]))
	plt.fill_between(X_test[:,0], est_th_in_low[:,j], est_th_in_up[:,j], alpha=0.4, label="Confidence intervals", color='gray')
	plt.scatter(X_test[:,0], fn(X_test).flatten(), label="True")
	plt.legend(loc='upper left')

	plt.subplot(len(k),5,5*j+3)
	plt.title("$k= {}$, Theoretical for D".format(k[j]))
	plt.fill_between(X_test[:,0], est_th_ex_low[:,j], est_th_ex_up[:,j], alpha=0.4, label="Confidence intervals", color='gray')
	plt.scatter(X_test[:,0], fn(X_test).flatten(), label="True")
	plt.legend(loc='upper left')

	plt.subplot(len(k),5,5*j+4)
	plt.title("$k= {}$, Coverage over ${}$ runs".format(k[j],mc_samples))
	plt.plot(X_test[:, 0], cov_prop[:,j], label='Adaptive with $\zeta = 0.1$', color='green')
	plt.plot(X_test[:, 0], cov_in[:,j], label='Theoretical for $d$', color='red')
	plt.plot(X_test[:, 0], cov_ex[:,j], label='Theoretical for $D$', color='blue')
	plt.legend(loc='lower left')

	s_mat = s_pick[:,:,j]

	plt.subplot(len(k),5,5*j+5)
	plt.title("$k= {}$, $s$ values".format(k[j]))
	plt.plot(X_test[:,0], np.mean(s_mat,axis=1), label="Adaptive")
	plt.plot(X_test[:,0], np.ones(n_test)*int(n**(1.05*d/(d+2))), label='Theoretical for $d$')
	plt.legend(loc='upper right')


plt.subplots_adjust(hspace = 0.3, left = 0.05, right = 0.95)

if save_results ==1:
    h = "coverage_multiple_points_n_{:d}_n_test_{:d}_sigma_{:.2f}_mcsamples_{:d}".format(n, n_test, sigma_e,mc_samples)
    plt.savefig(h+".png",dpi=600, format="png")
    plt.savefig(h+".pdf",dpi=600, format="pdf")

plt.show()