// heckman_binary.stan

functions {
  // Function to compute the bivariate normal CDF
  
  real binormal_cdf(real z1, real z2, real rho) {
  if (z1 != 0 || z2 != 0) {
    real denom = fabs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) : not_a_number();
    real a1 = (z2 / z1 - rho) / denom;
    real a2 = (z1 / z2 - rho) / denom;
    real product = z1 * z2;
    real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - owens_t(z1, a1) - owens_t(z2, a2);
  }
  return 0.25 + asin(rho) / (2 * pi());
}
  
  real Phi2(real x, real y, real rho) {
    return binormal_cdf(x, y, rho);
  }
}

data {
  int<lower=1> N;                 // Total number of observations
  int<lower=0,upper=1> u_i[N];    // Selection indicator (observed for all)
  int<lower=1> K_sel;             // Number of predictors in selection equation
  int<lower=1> K_out;             // Number of predictors in outcome equation
  matrix[N, K_sel] z;             // Predictors for selection equation
  matrix[N, K_out] x;             // Predictors for outcome equation
  int<lower=0,upper=1> y_i[N];    // Outcome variable (0 or 1), observed only if u_i == 1
}

parameters {
  vector[K_sel] gamma;            // Coefficients for selection equation
  vector[K_out] beta;             // Coefficients for outcome equation
  real<lower=-1,upper=1> rho;     // Correlation between errors
}

model {
  // Priors
  beta ~ normal(0, 5);
  gamma ~ normal(0, 5);
  rho ~ uniform(-1, 1);

  for (n in 1:N) {
    // Probit model for selection equation
    real eta_sel = z[n] * gamma;
    // Probit model for outcome equation
    real eta_out = x[n] * beta;

    if (u_i[n] == 1) {
      // For observations where u_i == 1, model joint probability
      if (y_i[n] == 1) {
        // Both u_i and y_i are 1
        target += log(Phi2(eta_sel, eta_out, rho));
      } else {
        // u_i == 1 and y_i == 0
        target += log(Phi(eta_sel) - Phi2(eta_sel, eta_out, rho));
      }
    } else {
      // For observations where u_i == 0
      target += log1m(Phi(eta_sel));
    }
  }
}
