"""
Author: Leo Guelman
Date: 2024-11-17
Description: Implementation of Heckman Robit-Selection model 
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm, chi2, multivariate_normal, wishart
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


class RobitSelection:

    # Initialize the class
    def __init__(self):
        pass

    # Define the log probability function for nu
    def _logp(self, nu, n, alpha0, eta):
        return n * nu / 2 * np.log(nu / 2) - n * gammaln(nu / 2) + (alpha0 - 1) * np.log(nu) - nu * eta

    # Define the first derivative of the log probability function
    def _Dlog(self, nu, n, alpha0, eta):
        return n / 2 * (np.log(nu / 2) + 1 - digamma(nu / 2)) + (alpha0 - 1) / nu - eta

    # Find the maximum of the log probability function
    def _Mlog(self, n, alpha0, eta):
        result = root_scalar(lambda nu: self._Dlog(nu, n, alpha0, eta), bracket=[1e-2, 1e6])
        return result.root

    # Define the second derivative of the log probability function
    def _DDlog(self, nu, n, alpha0, eta):
        return n / 2 * (1 / nu - polygamma(1, nu / 2) / 2) - (alpha0 - 1) / nu**2

    # Metropolized Independent Sampler (MIS) for nu
    def _MISnu(self, nu0, n, alpha0, eta):
        
        #NU0=nu, n=N, alpha0, eta=xi
        
        x = self._Mlog(n, alpha0, eta)
        h = self._DDlog(x, n, alpha0, eta)

        # Gamma approximation parameters
        b = -x * h
        a = -x * x * h + 1

        # Proposal from gamma distribution
        nu = scipy.stats.gamma.rvs(a, scale=1 / b)

        # Acceptance ratio
        logr = (self._logp(nu, n, alpha0, eta) - scipy.stats.gamma.logpdf(nu, a, scale=1 / b) -
                self._logp(nu0, n, alpha0, eta) + scipy.stats.gamma.logpdf(nu0, a, scale=1 / b))

        r = min(1, np.exp(logr))

        if np.random.uniform() < r:
            accept = 1
            return {'nu': nu, 'accept': accept}
        else:
            accept = 0
            return {'nu': nu0, 'accept': accept}

    # Solve a 2x2 matrix inversion
    def _solve2(self, a):
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        b = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]]) / det
        return b

    # Truncated normal random variate generator
    def _rtruncnorm(self, n, mu, sigma, lower=-np.inf, upper=np.inf):
        L = (lower - mu) / sigma
        U = (upper - mu) / sigma
        uniform = np.random.uniform(size=n)
        pL = norm.cdf(L)
        pU = norm.cdf(U)
        X = norm.ppf(pL + uniform * (pU - pL))
        Y = mu + sigma * X
        return Y

    # Main function implementing Robit selection with T-distribution
    def fit_robit(self, Y=None, U=None, X=None, W=None, mu0=None, Sigma0_inv=None,
                  nu0=None, alpha0=None, beta0=None, delta=None, Omega=None,
                  nu=None, Q=None, n_mcmc=10000, verbose=False, out_length=100):
        
        if not hasattr(self, "data"):
            raise ValueError("`simulate_data` must be called prior to fit method")

        # Extract data from simulation
        X = X or self.data['xi'].to_numpy()[:, np.newaxis]
        W = W or np.hstack((self.data['xi'].to_numpy()[:, np.newaxis], self.data['wi'].to_numpy()[:, np.newaxis] ))
        Y = Y or self.data['y_i_binary'].to_numpy()
        U = U or self.data['u_i'].to_numpy()

        # Add a constant term to the covariates
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        W = np.hstack((np.ones((W.shape[0], 1)), W))

        # Default parameters
        N = len(Y)
        K = X.shape[1]
        L = W.shape[1]

        # Prior parameters
        mu0 = mu0 or np.zeros(K + L)
        Sigma0_inv = Sigma0_inv or np.eye(K + L) / 1e2
        nu0 = nu0 or 3
        alpha0 = alpha0 or 1
        beta0 = beta0 or 0.1

        # Initial values
        delta = delta or np.zeros(K + L)
        Omega = Omega or np.eye(2)
        nu = nu or 4
        Q = Q or np.ones(N)

        # Separate delta into beta and gamma
        beta = delta[:K]
        gamma = delta[K:]

        Ulatent = U - 0.5
        Ylatent = np.where(U == 1, Y, 0)

        # Transformed data weighted by Q^0.5
        QY = Ylatent.copy()
        QU = Ulatent.copy()
        QX = X.copy()
        QW = W.copy()

        # MCMC results storage
        DELTA_mcmc = [delta.copy()]
        OMEGA_mcmc = [Omega.flatten()]
        NU_mcmc = [nu]

        # Gibbs sampler
        for mcmc in range(n_mcmc):
            if verbose and mcmc % out_length == 0:
                print(f"{mcmc}/{n_mcmc}")

            # Reparameterization
            rho = Omega[0, 1] 
            Omega_inv = self._solve2(Omega)

            # Imputation step for Selection-Robit Model
            for i in range(N):

                meany_u = np.dot(X[i, :], beta) + rho * (Ulatent[i] - np.dot(W[i, :], gamma))
                sigmay_u = np.sqrt((1 - rho**2) / Q[i])    
                    
                if U[i] == 1:
                    meanu_y = (np.dot(W[i, :], gamma) +
                               rho * (Ylatent[i] - np.dot(X[i, :], beta)))
                    sigmau_y = np.sqrt((1 - rho**2) / Q[i])
                    Ulatent[i] = self._rtruncnorm(1, meanu_y, sigmau_y, 0, np.inf)[0]

                    # For binary Y: adjust Ylatent based on observed Y
                    if Y[i] == 1:
                        Ylatent[i] = self._rtruncnorm(1, meany_u, sigmay_u, 0, np.inf)[0]  # truncated above 0
                    else:
                        Ylatent[i] = self._rtruncnorm(1, meany_u, sigmay_u, -np.inf, 0)[0]  # truncated below 0

                else:
                    meanu = np.dot(W[i, :], gamma)
                    sigmau = np.sqrt(1 / Q[i])
                    Ulatent[i] = self._rtruncnorm(1, meanu, sigmau, -np.inf, 0)[0]
                    Ylatent[i] = np.random.normal(meany_u, sigmay_u)

                # Update Q
                res_Y = Ylatent[i] - np.dot(X[i, :], beta)
                res_U = Ulatent[i] - np.dot(W[i, :], gamma)
                scale_Q = (Omega_inv[0, 0] * res_Y**2 + Omega_inv[1, 1] * res_U**2 +
                           2 * Omega_inv[0, 1] * res_Y * res_U)
                Q[i] = chi2.rvs(nu + 2) / (scale_Q + nu)

                # Transform the data
                sqrtQ = np.sqrt(Q[i])
                QY[i] = sqrtQ * Ylatent[i]
                QU[i] = sqrtQ * Ulatent[i]
                QX[i, :] = sqrtQ * X[i, :]
                QW[i, :] = sqrtQ * W[i, :]

            # Posterior step for delta
            XQX = Omega_inv[0, 0] * np.dot(QX.T, QX)
            XQW = Omega_inv[0, 1] * np.dot(QX.T, QW)
            WQX = XQW.T
            WQW = Omega_inv[1, 1] * np.dot(QW.T, QW)
            XQY = Omega_inv[0, 0] * np.dot(QX.T, QY)
            XQU = Omega_inv[0, 1] * np.dot(QX.T, QU)
            WQY = Omega_inv[1, 0] * np.dot(QW.T, QY)
            WQU = Omega_inv[1, 1] * np.dot(QW.T, QU)

            VOmgV = np.block([[XQX, XQW], [WQX, WQW]])
            VOmgZ = XQY + XQU, WQY + WQU
            VOmgZ = np.concatenate(VOmgZ)

            Sigma_delta = np.linalg.inv(VOmgV + Sigma0_inv)
            Mu_delta = Sigma_delta @ (VOmgZ + Sigma0_inv @ mu0)
            delta = multivariate_normal.rvs(mean=Mu_delta, cov=Sigma_delta)
            DELTA_mcmc.append(delta.copy())

            # Re-separate beta and gamma
            beta = delta[:K]
            gamma = delta[K:]

            # Sample Omega using parameter expansion
            ## draw sigma1 and sigma2 independently from their priors 
            sigma1 = 1 / np.sqrt((1 - rho**2) * chi2.rvs(nu0))
            sigma2 = 1 / np.sqrt((1 - rho**2) * chi2.rvs(nu0))
            ## draw SIGMA matrix
            res_QY = QY - QX @ beta
            res_QU = QU - QW @ gamma
            SS11 = sigma1**2 * np.sum(res_QY**2)
            SS12 = sigma1*sigma2 * np.sum(res_QY * res_QU)
            SS22 = sigma2**2 * np.sum(res_QU**2)
            SS = np.array([[SS11, SS12], [SS12, SS22]])
            Sigma = np.linalg.inv(wishart.rvs(N + nu0, np.linalg.inv(SS + np.eye(2))))
            ## transform back to Omega
            sigma1 = np.sqrt(Sigma[0, 0])
            sigma2 = np.sqrt(Sigma[1, 1])
            Omega[0, 0] = 1
            Omega[0, 1] = Sigma[0, 1] / (sigma1 * sigma2)
            Omega[1, 0] = Omega[0, 1]
            Omega[1, 1] = 1
            OMEGA_mcmc.append(Omega.flatten())

            # Sample degrees of freedom nu
            xi = beta0 + np.sum(Q) / 2 - np.sum(np.log(Q)) / 2
            MHnu = self._MISnu(nu, N, alpha0, xi)
            nu = MHnu['nu']
            NU_mcmc.append(nu)

        # Collect results
        MCMC_robit = {
            'DELTA_mcmc': np.array(DELTA_mcmc),
            'OMEGA_mcmc': np.array(OMEGA_mcmc),
            'NU_mcmc': np.array(NU_mcmc),
        }
        
        self.MCMC_robit = MCMC_robit

        return self
    
# =============================================================================
#     def fit_probit(self, iter=2000, chains=4):
#         
#         
#         #from cmdstanpy import install_cmdstan
#         #install_cmdstan()
#         
#         from cmdstanpy import CmdStanModel
#         
#         if not hasattr(self, "data"):
#             raise ValueError("`simulate_data` must be called prior to fit method")
# 
#         # Extract data from simulation
#         x = self.data['xi'].to_numpy()[:, np.newaxis]
#         z = np.hstack((self.data['xi'].to_numpy()[:, np.newaxis], self.data['wi'].to_numpy()[:, np.newaxis] ))
#         y_i = self.data['y_i_binary'].to_numpy()
#         u_i  = self.data['u_i'].to_numpy()
#         
#         N = len(y_i)
#         K_out = x.shape[1]
#         K_sel = z.shape[1]
#        
#         stan_data = {
#             'N': N,
#             'u_i': u_i,
#             'y_i': y_i,
#             'K_sel': K_sel,
#             'K_out': K_out,
#             'z': z,
#             'x': x
#             }
# 
#         
#         heckman_model = CmdStanModel(stan_file='/Users/newuser/Desktop/RI/code/Heckman-Robit/stan/heckman_probit.stan') 
#         #fit = heckman_model.sampling(data=data_dict, iter=iter, chains=chains)
#        
#         # Extract results
#         #results = fit.extract()
#        # beta_estimates = results['beta']
#        # gamma_estimates = results['gamma']
#        # rho_estimate = results['rho']
#         
#         #MCMC_probit = {
#         #    'beta_mcmc': np.array(beta_estimates),
#         #    'gamma_mcmc': np.array(gamma_estimates),
#         #    'rho_mcmc': np.array(rho_estimate),
#         #}
#         
#         #self.MCMC_probit = MCMC_probit
# 
#         return self
#         
#     
# =============================================================================

    def simulate_data(self, N=2000, error_type='gaussian', rho=0.3, nu=3):
        """
        Simulate data with either Gaussian or t-distributed errors.

        Parameters:
        - N (int): Sample size.
        - error_type (str): Type of error distribution ('gaussian' or 't').
        - rho (float): Correlation between errors.
        - nu (int): Degrees of freedom for t-distribution (only used if error_type is 't').

        Returns:
        - data (DataFrame): A pandas DataFrame containing the simulated data.
        """

        # Set random seed for reproducibility
        np.random.seed(0)

        # Generate xi ∼ N(0, 2^2)
        xi = np.random.normal(0, 2, N)

        # Generate wi ∼ N(0, 2^2)
        wi = np.random.normal(0, 2, N)

        # Mean vector for (εi, ηi)
        mean_vector = [0, 0]

        # Covariance matrix Σ = [[1, 0.3], [0.3, 1]]
        cov_matrix = [[1, rho],
                      [rho, 1]]

        if error_type == 'gaussian':
            # Generate (εi, ηi) ∼ N2((0, 0), Σ)
            epsilon_eta = np.random.multivariate_normal(mean_vector, cov_matrix, N)
        elif error_type == 't':
            # Generate standard normal random variables Z ∼ N(0, Σ)
            Z = np.random.multivariate_normal(mean_vector, cov_matrix, N)

            # Generate chi-squared random variables W ∼ χ²(ν)/ν
            W = np.random.chisquare(nu, N) / nu
            W = W.reshape(-1, 1)  # Reshape for broadcasting

            # Generate (εi, ηi) from bivariate t-distribution with ν degrees of freedom
            epsilon_eta = Z / np.sqrt(W)
        else:
            raise ValueError("error_type must be 'gaussian' or 't'")

        epsilon_i = epsilon_eta[:, 0]
        eta_i = epsilon_eta[:, 1]

        # Compute latent outcome y_i^* = 0.5 + xi + εi
        y_star = 0.5 + xi + epsilon_i

        # Compute selection mechanism u_i^* = 2 + xi + 1.5*wi + ηi
        u_star = 2 + xi + 1.5 * wi + eta_i

        # Compute selection indicator ui = I(u_i^* > 0)
        u_i = (u_star > 0).astype(int)

        # Generate observed outcome yi = y_i^* if ui = 1, else yi = NA
        y_i = np.where(u_i == 1, y_star, np.nan)

        # Replace yi with I(yi > 0)
        # Handle NaN values appropriately
        y_i_binary = np.where(np.isnan(y_i), np.nan, (y_i > 0).astype(int))

        # Create a DataFrame to store the results
        data = pd.DataFrame({
            'xi': xi,
            'wi': wi,
            'epsilon_i': epsilon_i,
            'eta_i': eta_i,
            'y_star': y_star,
            'u_star': u_star,
            'u_i': u_i,
            'y_i': y_i,
            'y_i_binary': y_i_binary
        })

        self.error_type = error_type
        self.data = data

        return self
    


###-------- Fit models and save results --------### 

import pickle
import os
os.chdir('/Users/newuser/Desktop/RI/code/files')

rs =  RobitSelection()
rs.simulate_data(N=1000, error_type='gaussian')
rs.fit_robit(verbose=True, n_mcmc=5*10**4)
Gauss_MCMC_robit = rs.MCMC_robit
with open('Gauss_MCMC_robit.pkl', 'wb') as file:  # 'wb' stands for write binary
    pickle.dump(Gauss_MCMC_robit, file)


rs =  RobitSelection()
rs.simulate_data(N=1000, error_type='t')
rs.fit_robit(verbose=True, n_mcmc=5*10**4)
t_MCMC_robit = rs.MCMC_robit
with open('t_MCMC_robit.pkl', 'wb') as file:  # 'wb' stands for write binary
    pickle.dump(t_MCMC_robit, file)


###-------- Plots results --------### 


def plot_parameters(probit_fit, robit_fit, title):
     
     # First 10^4 draws discarded as burn in period
     
     beta0_group1 = robit_fit.get('DELTA_mcmc')[10000:,0]
     beta0_group2 = probit_fit['beta[1]'].to_numpy() 
     
     beta1_group1 = robit_fit.get('DELTA_mcmc')[10000:,1]
     beta1_group2 = probit_fit['beta[2]'].to_numpy() 
     
     gamma0_group1 = robit_fit.get('DELTA_mcmc')[10000:,2]
     gamma0_group2 = probit_fit['gamma[1]'].to_numpy() 
     
     gamma1_group1 = robit_fit.get('DELTA_mcmc')[10000:,3]
     gamma1_group2 = probit_fit['gamma[2]'].to_numpy() 
     
     gamma2_group1 = robit_fit.get('DELTA_mcmc')[10000:,4]
     gamma2_group2 = probit_fit['gamma[3]'].to_numpy() 
     

     rho_group1 = robit_fit.get('OMEGA_mcmc')[10000:,2]
     rho_group2 = probit_fit['rho'].to_numpy() 
     nu_group1 = robit_fit.get('NU_mcmc')[10000:]


     # Organizing the data into a list
     data = [
         (beta0_group1, beta0_group2, r'$\beta_0$', (0.3, 1.2), 0.5),
         (beta1_group1, beta1_group2, r'$\beta_1$', (0.6, 1.8), 1.0),
         (gamma0_group1, gamma0_group2, r'$\gamma_0$', (1.5, 2.5), 2.0),
         (gamma1_group1, gamma1_group2, r'$\gamma_1$', (0.6, 1.4), 1.0),
         (gamma2_group1, gamma2_group2, r'$\gamma_2$', (0.6, 2.0), 1.5),
        # (rho_group1, rho_group2, r'$\rho$', (0.0, 0.5), 0.32),
        # (nu_group1, nu_group1, r'$\nu$', (0, 60), 0)
     ]
     
     # Setting up the grid layout
     fig, axes = plt.subplots(1, len(data), figsize=(15, 5), sharey=False)
     fig.suptitle(title, fontsize=16, weight='bold')
     
     # Colors for the groups
     colors = ['lightgray', 'gray']
     
     # Creating individual boxplots for each parameter
     for ax, (group1, group2, label, ylim, line_value) in zip(axes, data):
         bplot = ax.boxplot(
             [group1, group2], patch_artist=True, widths=0.3
         )
         
         # Customizing colors for box plots
         for patch, color in zip(bplot['boxes'], colors):
             patch.set_facecolor(color)
         
         # Customizing each subplot
         ax.set_title(label, fontsize=14)
         ax.axhline(y=line_value, color='blue', linestyle='--', linewidth=1.0)  # Blue reference line
         ax.set_xticks([])  # Remove x-axis labels
         ax.set_ylim(ylim)
         ax.grid(axis='y', linestyle='--', alpha=0.6)
     
     # Adding a legend for the groups
     fig.legend(
         handles=[
             plt.Line2D([0], [0], color='lightgray', lw=10, label='Selection-Robit'),
             plt.Line2D([0], [0], color='gray', lw=10, label='Selection-Probit')
         ],
         loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=2, fontsize=12, frameon=False
     )
     
     plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Adjust layout to fit title and legend
     plt.show()

# Get data from Gaussian sim

robit_fit = pd.read_pickle('Gauss_MCMC_robit.pkl')
probit_fit = pd.read_csv('probit_gaussian.csv')

plot_parameters(probit_fit, robit_fit, 
                title="Data Generated with Selection-Probit and analyzed with selection-probit and selection-robit")


# Get data from t-sim

robit_fit = pd.read_pickle('t_MCMC_robit.pkl')
probit_fit = pd.read_csv('probit_t.csv')

plot_parameters(probit_fit, robit_fit, 
                title="Data Generated with Selection-Robit and analyzed with selection-probit and selection-robit")






