import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions.transforms as T

# Define the neural network for F(x; θ)
class FNetwork(PyroModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden_dim, 1)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# Define the neural network for G(z; φ)
class GNetwork(PyroModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden_dim, 1)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        return self.fc2(z).squeeze(-1)

# Define the model
def model(x_data, z_data, d_obs, f_obs):
    pyro.module("F_network", F_network)
    pyro.module("G_network", G_network)
    
    # Priors for ν and ρ
    a_nu = 2.0  # Hyperparameters for InvGamma prior
    b_nu = 2.0
    nu = pyro.sample('nu', dist.InverseGamma(a_nu, b_nu))
    
    rho = pyro.sample('rho_unconstrained', dist.Uniform(-1 + 1e-5, 1 - 1e-5))
    rho = torch.clamp(rho, -1 + 1e-5, 1 - 1e-5)  # Ensure rho is in (-1, 1)
    
    n = x_data.shape[0]
    for i in range(n):
        xi = x_data[i]
        zi = z_data[i]
        
        # Mixing variable λ_i ~ Gamma(ν/2, ν/2)
        lambda_i = pyro.sample(f'lambda_{i}', dist.Gamma(nu / 2, nu / 2))
        
        # Error terms ε_i and η_i given λ_i
        R = torch.tensor([[1.0, rho], [rho, 1.0]])
        scale_tril = torch.cholesky(R / lambda_i)
        eps_eta = pyro.sample(f'eps_eta_{i}', dist.MultivariateNormal(torch.zeros(2), scale_tril=scale_tril))
        eps_i = eps_eta[0]
        eta_i = eps_eta[1]
        
        # Latent variables d_i^* and f_i^*
        d_star = F_network(xi) + eps_i
        f_star = G_network(zi) + eta_i
        
        # Observations
        fi = f_obs[i]
        di = d_obs[i]
        
        # Likelihood for f_i
        with pyro.poutine.mask(mask=~torch.isnan(fi)):
            pyro.sample(f'f_obs_{i}', dist.Bernoulli(logits=f_star), obs=fi)
        
        # Likelihood for d_i (only if f_i == 1)
        if fi == 1 and not torch.isnan(di):
            pyro.sample(f'd_obs_{i}', dist.Bernoulli(logits=d_star), obs=di)

# Define the guide (variational distribution)
def guide(x_data, z_data, d_obs, f_obs):
    pyro.module("F_network", F_network)
    pyro.module("G_network", G_network)
    
    # Variational parameters for ν and ρ
    nu_q_loc = pyro.param('nu_q_loc', torch.tensor(5.0), constraint=dist.constraints.positive)
    nu_q_scale = pyro.param('nu_q_scale', torch.tensor(1.0), constraint=dist.constraints.positive)
    nu_q = pyro.sample('nu', dist.LogNormal(nu_q_loc, nu_q_scale))
    
    rho_q_loc = pyro.param('rho_q_loc', torch.tensor(0.0))
    rho_q_scale = pyro.param('rho_q_scale', torch.tensor(0.5), constraint=dist.constraints.positive)
    rho_q = pyro.sample('rho_unconstrained', dist.Normal(rho_q_loc, rho_q_scale))
    
    n = x_data.shape[0]
    for i in range(n):
        # Variational parameters for λ_i
        lambda_q_alpha = pyro.param(f'lambda_q_alpha_{i}', torch.tensor(5.0), constraint=dist.constraints.positive)
        lambda_q_beta = pyro.param(f'lambda_q_beta_{i}', torch.tensor(1.0), constraint=dist.constraints.positive)
        lambda_i = pyro.sample(f'lambda_{i}', dist.Gamma(lambda_q_alpha, lambda_q_beta))
        
        # Variational parameters for ε_i and η_i
        eps_eta_q_loc = pyro.param(f'eps_eta_q_loc_{i}', torch.zeros(2))
        eps_eta_q_scale = pyro.param(f'eps_eta_q_scale_{i}', torch.ones(2), constraint=dist.constraints.positive)
        pyro.sample(f'eps_eta_{i}', dist.Normal(eps_eta_q_loc, eps_eta_q_scale).to_event(1))

# Instantiate neural networks
input_dim_x = x_data.shape[1]
input_dim_z = z_data.shape[1]
hidden_dim = 10  # You can adjust the hidden dimension

F_network = FNetwork(input_dim=input_dim_x, hidden_dim=hidden_dim)
G_network = GNetwork(input_dim=input_dim_z, hidden_dim=hidden_dim)

# Observed data (replace with your actual data)
# x_data: covariate matrix for default equation (n_samples x p)
# z_data: covariate matrix for funding equation (n_samples x q)
# d_obs: observed defaults (0 or 1, or NaN if not observed)
# f_obs: observed funding indicators (0 or 1)

# Example synthetic data
n_samples = 100
p = 5
q = 5

x_data = torch.randn(n_samples, p)
z_data = torch.randn(n_samples, q)
d_obs = torch.randint(0, 2, (n_samples,)).float()
f_obs = torch.randint(0, 2, (n_samples,)).float()

# Handle missing d_obs when f_obs == 0
d_obs[f_obs == 0] = float('nan')

# Set up the optimizer and inference algorithm
adam_params = {"lr": 0.01}
optimizer = Adam(adam_params)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Training loop
num_iterations = 1000
for step in range(num_iterations):
    loss = svi.step(x_data, z_data, d_obs, f_obs)
    if step % 100 == 0:
        print(f"Step {step} : loss = {loss}")

# After training, you can access the learned parameters
# For example, the variational parameters for ν:
nu_mean = pyro.param('nu_q_loc').item()
nu_std = pyro.param('nu_q_scale').item()
print(f"Estimated ν ~ LogNormal({nu_mean}, {nu_std})")
