# Load necessary libraries
library(rstan)
library(mvtnorm)


simulate_data <- function(N = 1000, error_type = 'gaussian', rho = 0.3, nu = 3) {
  # Load necessary library
  library(mvtnorm)
  
  # Set random seed for reproducibility
  set.seed(0)
  
  # Generate xi ∼ N(0, 2^2)
  xi <- rnorm(N, mean = 0, sd = 2)
  
  # Generate wi ∼ N(0, 2^2)
  wi <- rnorm(N, mean = 0, sd = 2)
  
  # Mean vector for (εi, ηi)
  mean_vector <- c(0, 0)
  
  # Covariance matrix Σ = [[1, rho], [rho, 1]]
  cov_matrix <- matrix(c(1, rho, rho, 1), nrow = 2, byrow = TRUE)
  
  if (error_type == 'gaussian') {
    # Generate (εi, ηi) ∼ N2((0, 0), Σ)
    epsilon_eta <- rmvnorm(N, mean = mean_vector, sigma = cov_matrix)
  } else if (error_type == 't') {
    # Generate standard normal random variables Z ∼ N(0, Σ)
    Z <- rmvnorm(N, mean = mean_vector, sigma = cov_matrix)
    
    # Generate chi-squared random variables W ∼ χ²(ν)/ν
    W <- rchisq(N, df = nu) / nu
    
    # Generate (εi, ηi) from bivariate t-distribution with ν degrees of freedom
    W_sqrt <- sqrt(W)
    epsilon_eta <- Z / W_sqrt
  } else {
    stop("error_type must be 'gaussian' or 't'")
  }
  
  epsilon_i <- epsilon_eta[, 1]
  eta_i <- epsilon_eta[, 2]
  
  # Compute latent outcome y_star = 0.5 + xi + εi
  y_star <- 0.5 + xi + epsilon_i
  
  # Compute selection mechanism u_star = 2 + xi + 1.5*wi + ηi
  u_star <- 2 + xi + 1.5 * wi + eta_i
  
  # Compute selection indicator u_i = I(u_star > 0)
  u_i <- ifelse(u_star > 0, 1, 0)
  
  # Generate observed outcome y_i = y_star if u_i == 1, else NA
  y_i <- ifelse(u_i == 1, y_star, NA)
  
  # Replace y_i with I(y_i > 0), handling NA values appropriately
  y_i_binary <- ifelse(is.na(y_i), NA, ifelse(y_i > 0, 1, 0))
  
  # Create a DataFrame to store the results
  data <- data.frame(
    xi = xi,
    wi = wi,
    epsilon_i = epsilon_i,
    eta_i = eta_i,
    y_star = y_star,
    u_star = u_star,
    u_i = u_i,
    y_i = y_i,
    y_i_binary = y_i_binary
  )
  
  return(data)
}


###---------------Probit-Selection Model fitted to Gaussian data---------------###

# Simulate data with default parameters
data <- simulate_data(error_type = 'gaussian')

# View the first few rows of the simulated data
head(data)

stan_model <- stan_model(file = '/Users/newuser/Desktop/RI/code/Heckman-Robit/stan/heckman_probit.stan')


# Prepare data for Stan
N <- nrow(data)
u_i <- data$u_i
y_i <- ifelse(is.na(data$y_i_binary), 0, data$y_i_binary)
z <- as.matrix(data[, c('xi', 'wi')])
z <- cbind(1, z)
x <- as.matrix(data[, 'xi', drop = FALSE])
x <- cbind(1, x)
K_sel <- dim(z)[2]
K_out <- dim(x)[2]

stan_data <- list(
  N = N,
  u_i = u_i,
  y_i = y_i,
  K_sel = K_sel,
  K_out = K_out,
  z = z,
  x = x
)

# Fit the model
fit <- sampling(
  stan_model,
  data = stan_data,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  seed = 123
)

# Print summary of the results
print(fit, pars = c('beta', 'gamma', 'rho'))


probit_gaussian <- as.data.frame(fit)
head(probit_gaussian)

write.csv(probit_gaussian, file = "/Users/newuser/Desktop/RI/code/files/probit_gaussian.csv", row.names = FALSE)


###---------------Probit-Selection Model fitted to t-data---------------###

# Simulate data with default parameters
data <- simulate_data(error_type = 't')

# View the first few rows of the simulated data
head(data)

stan_model <- stan_model(file = '/Users/newuser/Desktop/RI/code/Heckman-Robit/stan/heckman_probit.stan')

# Prepare data for Stan
N <- nrow(data)
u_i <- data$u_i
y_i <- ifelse(is.na(data$y_i_binary), 0, data$y_i_binary)
z <- as.matrix(data[, c('xi', 'wi')])
z <- cbind(1, z)
x <- as.matrix(data[, 'xi', drop = FALSE])
x <- cbind(1, x)
K_sel <- dim(z)[2]
K_out <- dim(x)[2]

stan_data <- list(
  N = N,
  u_i = u_i,
  y_i = y_i,
  K_sel = K_sel,
  K_out = K_out,
  z = z,
  x = x
)

# Fit the model
fit <- sampling(
  stan_model,
  data = stan_data,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = 4,
  seed = 123
)

# Print summary of the results
print(fit, pars = c('beta', 'gamma', 'rho'))


probit_t <- as.data.frame(fit)
head(probit_t)

write.csv(probit_gaussian, file = "/Users/newuser/Desktop/RI/code/files/probit_t.csv", row.names = FALSE)
