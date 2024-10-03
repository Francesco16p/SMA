# Logistic regression on the Cushing dataset 

This file contains a complete description of how to reproduce some key results concerning the logistic regression estimated on the `Cushing` dataset described in Section 5.2 of the main article and Appendix E.5 of the Supplementary Material (see Tables 3 and E.4). In particular, it provides code to implement both the joint and marginal skew-modal approximation to the model under consideration and to compare the quality of this approximation with that of the Gaussian Laplace, Gaussian variational Bayes and Gaussian expectation propagation approximations. 

Before starting, create a folder called `LogisticCushing` and set it as the working directory for the R environment.

## Upload the Cushing dataset

We start by uploading the Cushing's dataset as described in Section 5.2 of the paper. The aim is to investigate the relationship between four different subtypes of Cushing's syndrome and two steroid metabolites: Tetrahydrocortisone and Pregnanetriol. The data are available in the `Cushings` dataframe of the `MASS` library. This contains information on the response variable (`Cushings$Type`) and the covariates (`Cushings$Tetrahydrocortisone` and `Cushings$Pregnanetriol`). More specifically, `Cushings$Type` summarises information about the underlying type of syndrome of each subject, coded as `a` (adenoma), `b` (bilateral hyperplasia), `c` (carcinoma), or `u` for unknown. To make the response variable suitable for a logistic regression, we create a binary response `y` indicating whether the subject is affected by bilateral hyperplasia or not, and we generate a design matrix `X` for the model with three columns: intercept, `Cushings$Tetrahydrocortisone` and `Cushings$Pregnanetriol`. We save these quantities in the RData file `Cushings.RData` for further use.

```r
# Clear the environment
rm(list = ls())

# Load the MASS library
library(MASS)

# Create the response variable
y <- (MASS::Cushings$Type == "b")

# Determine sample size
n <- length(y)

# Define the design matrix components
x0 <- rep(1,)
x1 <- MASS::Cushings$Tetrahydrocortisone
x2 <- MASS::Cushings$Pregnanetriol

# Combine the components into the design matrix
X <- cbind(x0, x1, x2)

# Save the relevant quantities in the file Cushings.RData
save(y, X, n, file = "Cushings.RData")
```

---

## Logistic regression with STAN

We now estimate a Bayesian logistic regression with `y` as the variable of interest and the design matrix equal to `X` using the `rstan` library. The coefficients are assumed to have independent Gaussian priors with zero mean and standard error `sd = 5`. Since obtaining i.i.d. samples from the posterior is not straightforward, we use the STAN environment to generate 4 Hamiltonian Monte Carlo chains of length 10,000, which allow us to obtain an accurate approximation of the posterior. Let us first define the model in STAN language. In the following, `N` is the sample size, `D` is the number of parameters in the model, `X` is the design matrix containing the intercept, `y` is the response variable, and `sd` is the standard error of the prior.

```
stan_model_file <- 'data{
    int<lower=0> N; // number of observations
    int<lower=0> D; // number of predictors
    matrix[N, D] X; // design matrix
    int<lower=0, upper=1> y[N]; // response variable
    real<lower=0> sd; // standard deviation of the priors
}

parameters {
    vector[D] theta; // coefficients
}

model {
    // Prior
    theta ~ normal(0, sd);
    // Likelihood
    y ~ bernoulli_logit(X * theta);
}'

```

### Data preparation 
We now proceed to estimate the model using Hamiltonian Monte Carlo. First we clear the environment, load the necessary libraries and create a list containing `y`, `X`, `D` (number of predictors), `N` (number of observations) and `sd` (standard deviation of the priors).

```r
rm(list = ls())
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load the dataset
load("Cushings.RData")

# Create the data list for STAN
df <- list(y = y, X = X, D = ncol(X), N = nrow(X), sd = 5)
```


### Compile the STAN model and sample from it

Let us now build the STAN model and sample from its posterior distribution, specifying the number of iterations of the 4 chains, the warm-up period and the seed for reproducibility.

```r
fit <- stan(model_code = stan_model_file, data = df, iter = 10^4, chains = 4, warmup = 5000, seed = 1)
```

To extract the results of the Markov Chain Monte Carlo simulation write.

```r
MCMC_logistic <- extract(fit)$theta

# Save MCMC output for future use
save(MCMC_logistic, file = "MCMC_logistic.RData")
```

### Estimate posterior densities using `mclust`

It is now possible to use Markov Chain Monte Carlo simulation to estimate the joint, bivariate and marginal posterior densities using the `mclust` library. These estimated densities are used as proxies for the exact posterior to evaluate the quality of different approximations. In the following, the numbers 0, 1 and 2 indicate which parameters the density refers to. To estimate such functions, write:

```r
library(mclust)

# Joint posterior density (theta_0,theta_1,theta_2)
d012 <- densityMclust(MCMC_logistic[,1:3])

post_theta_012 <- function(x) {
  x <- matrix(x, ncol = 3)
  predict(d012, x)
}

# Bivariate posterior densities (theta_0,theta_1)
d01 <- densityMclust(MCMC_logistic[,1:2])

post_theta_01 <- function(x) {
  x <- matrix(x, ncol = 2)
  predict(d01, x)
}

# Bivariate posterior densities (theta_0,theta_2)
d02 <- densityMclust(MCMC_logistic[,c(1,3)])

post_theta_02 <- function(x) {
  x <- matrix(x, ncol = 2)
  predict(d02, x)
}

# Bivariate posterior densities (theta_1,theta_2)
d12 <- densityMclust(MCMC_logistic[,c(2,3)])

post_theta_12 <- function(x) {
  x <- matrix(x, ncol = 2)
  predict(d12, x)
}

# Marginal posterior densities theta_0
d0 <- densityMclust(MCMC_logistic[,1])

post_theta_0 <- function(x) {
  predict(d0, x)
}

# Marginal posterior densities theta_1
d1 <- densityMclust(MCMC_logistic[,2])

post_theta_1 <- function(x) {
  predict(d1, x)
}

# Marginal posterior densities theta_2
d2 <- densityMclust(MCMC_logistic[,3])

post_theta_2 <- function(x) {
  predict(d2, x)
}
```

Finally, we save the estimated densities for future use.

```r
save(d0, d1, d2, d01, d02, d12, d012,
     post_theta_0, post_theta_1, post_theta_2,
     post_theta_01, post_theta_02, post_theta_12,
     post_theta_012,
     file = "estimated_posteriors_logistic.RData")
```

## Laplace and skew-modal approximations

Let us now obtain both the Gaussian Laplace and the skew-modal approximations of the posterior distribution. This process involves the computation of the posterior mode (MAP), of the observed Fisher information and of the third log-likelihood derivative.

### Evaluation of the posterior mode

To obtain the MAP estimate, it is sufficient to maximise the posterior distribution using the `R` function `optim()`. We do this by clearing the global environment, loading the `mvtnorm` library and defining the function `log_post()`, which corresponds to the posterior distribution induced by the model under analysis.

```r
# Clear the workspace by removing all objects
rm(list = ls())  

# Loading necessary libraries
library(mvtnorm)  # For multivariate normal distribution functions

# Load dataset
load("Cushings.RData")  

# Determine the number of parameters
d <- dim(X)[2] 

# Define the log-posterior distribution
sd <- 5  # Standard deviation for the Gaussian prior

log_post <- function(theta) {
  eta <- X %*% theta  # Linear predictor
  p <- exp(eta) / (1 + exp(eta))  # Logistic function
  # Log-likelihood of the data and log-prior
  (sum(dbinom(y, size = 1, prob = p, log = TRUE)) + 
     mvtnorm::dmvnorm(theta, sigma = sd^2 * diag(d), log = TRUE))
}
```

At this point, the MAP is obtained by writing 

```
# Compute the maximum a posteriori (MAP) estimate
map <- optim(rep(0, d), function(k) -log_post(k), method = "BFGS")
```

### Log-likelihood derivatives

Let us now define the two functions to compute the observed Fisher information and the third log-likelihood derivative. As described in the main paper, these quantities, evaluated at the MAP are needed to obtain both Gaussian Laplace and skew-modal approximations.

```r

# Define the observed Fisher information (negative second derivative of the log-likelihood)
obs_inf <- function(theta) {
  eta <- X %*% theta  # Linear predictor
  dp_eta <- c(exp(eta) / (1 + exp(eta))^2) * diag(n)  # First derivative of the log-likelihood
  obs <- -t(X) %*% dp_eta %*% X  # Compute observed information matrix
  -obs  # Negate to get the observed information
}

# Define the third derivative of the log-likelihood
trd_derivative <- function(theta) {
  eta <- X %*% theta  # Linear predictor
  dp_eta2 <- exp(eta) * (1 - exp(eta)) / (1 + exp(eta))^3  # Third derivative of the log-likelihood
  trd <- array(0, dim = c(d, d, d))  # Initialize the third derivative array
  for (k in 1:d) {
    for (r in 1:d) {
      for (s in 1:d) {
        trd[k, r, s] <- -sum(dp_eta2 * X[, k] * X[, r] * X[, s])  # Compute third derivative
      }
    }
  }
  trd  # Return the third derivative
}
```

### Gaussian Laplace approximations

At this point it is possible to obtain the joint, bivariates and marginals of the Gaussian Laplace approximation of the posterior density by writing:

```r
# Compute the observed Fisher information matrix at MAP
obsInf <- obs_inf(map$par)

# Define parameters for the Laplace approximation
la <- list()
la$m <- map$par  # MAP estimate of the parameters
la$V <- solve(obsInf)  # Covariance matrix from the observed information

# Joint Gaussian Laplace approximation (theta_0,theta_1,theta_2)
la_theta012 <- function(theta) {
  mvtnorm::dmvnorm(theta, mean = la$m, sigma = la$V)
}

# Bivariate Gaussian Laplace approximation (theta_0,theta1)
la_theta_01 <- function(theta) {
  mvtnorm::dmvnorm(theta, mean = la$m[c(1, 2)], sigma = la$V[c(1, 2), c(1, 2)])
}

# Bivariate Gaussian Laplace approximation (theta_0,theta2)
la_theta_02 <- function(theta) {
  mvtnorm::dmvnorm(theta, mean = la$m[c(1, 3)], sigma = la$V[c(1, 3), c(1, 3)])
}

# Bivariate Gaussian Laplace approximation (theta_1,theta2)
la_theta_12 <- function(theta) {
  mvtnorm::dmvnorm(theta, mean = la$m[c(2, 3)], sigma = la$V[c(2, 3), c(2, 3)])
}

# Marginal Gaussian Laplace approximation theta_0
la_theta_0 <- function(theta) {
  dnorm(theta, mean = la$m[1], sd = sqrt(la$V[1, 1]))
}

# Marginal Gaussian Laplace approximation theta_1
la_theta_1 <- function(theta) {
  dnorm(theta, mean = la$m[2], sd = sqrt(la$V[2, 2]))
}

# Marginal Gaussian Laplace approximation theta_2
la_theta_2 <- function(theta) {
  dnorm(theta, mean = la$m[3], sd = sqrt(la$V[3, 3]))
}
```

We now save these approximations for future use. 

```r
# Save the Laplace approximations
save(la, la_theta012, la_theta_01, la_theta_02, la_theta_12, la_theta_0, la_theta_1, la_theta_2, file = "Logistic_Laplace_approx.RData")
```

### Skew-modal approximations

To obtain joint, bivariates and marginals of the proposed skew-modal approximation of the posterior distribution we follow the expressions reported in Sections 4.1-4.2 of the main paper. As a first step we obtain the $d^3$-dimensional array of  third log-likelihood derivatives evaluated at the MAP.

```r
# Third log-likelihood derivatives evaluated at
nu_ttt <- trd_derivative(map$par)
```
Once the third log-likelihood derivative is obtained, it is possible to compute the skewness inducing coefficients of the bivariate and marginal skew-modal approximation as reported in section 4.2 of the paper. To do this, we define the function `coef_marginal()`, which takes the following parameters as input: `loc`, a numerical vector containing the position of the parameters to which the bivariate/marginal approximation refers; `a2`, which corresponds to the covariance matrix of the Gaussian component and `a3`, the array of the third log-likelihood derivatives.

```r
# Parameters:
# loc = position selected parameters
# a2 = covariance matrix
# a3 = 3rd log-likelohood derivatives 

coef_marginal <- function(loc, a2, a3)
{
  d <- dim(a2)[2]
  d_c <- length(loc)  # dimension sub-set 
  idx <- setdiff(1:d, loc)
  
  # Marginal case 
  if(d_c == 1 )
  {
    # Mean and variance of the approximation
    L <- a2[idx,loc]/a2[loc,loc]
    bomega <- a2[idx,idx] - a2[idx,loc]%*%t(a2[idx,loc])/a2[loc,loc]
    
    # Skewness inducing components
    nu_1 <- 0
    nu_3 <- a3[loc,loc,loc]
    for(s in 1:(d-1))
    {
      nu_3 <- nu_3 + 3*a3[loc,loc,idx[s]]*L[s]
      for(r in 1:(d-1) )
      {
        nu_1 <- nu_1 + 3*a3[loc,idx[s],idx[r]]*bomega[s,r]
        nu_3 <- nu_3 + 3*a3[loc,idx[s],idx[r]]*L[s]*L[r]
        for(k in 1:(d-1))
        {
          nu_1 <- nu_1 + 3*a3[idx[s],idx[r],idx[k]]*bomega[s,r]*L[k]
          nu_3 <- nu_3 + a3[idx[s],idx[r],idx[k]]*L[s]*L[r]*L[k]
        }
      }
    }
  }
  
  #  Multivariate case 
  else{
    # Inversion covariance matrix 
    inv.a2.loc <- solve(a2[loc,loc])
    # Mean and Covariance matrix of the approximation
    L <- a2[idx,loc]%*%inv.a2.loc
    bomega <- a2[idx,idx] - a2[idx,loc]%*%inv.a2.loc%*%a2[loc,idx]
    
    # Skewness inducing components
    nu_1 <- rep(0,d_c)
    nu_3 <- a3[loc,loc,loc]
    
    # Linear component
    for(s in 1:d_c)
    {
      for(r in 1:(d-d_c))
      {
        for(v in 1:(d-d_c) )
        {
          nu_1[s] <- nu_1[s] + 3*a3[loc[s],idx[r],idx[v]]*bomega[r,v]
          for(k in 1:(d-d_c))
          {
            nu_1[s] <- nu_1[s] +3*a3[idx[r],idx[v],idx[k]]*L[k,s]*bomega[r,v]
          }
        }
      }
    }
    
    # Cubic component
    for(s in 1:d_c)
    {
      for(t in 1:d_c)
      {
        for(l in 1:d_c)
        {
          for(r in 1:(d-d_c))
          {
            nu_3[s,t,l] <- nu_3[s,t,l] + 3*a3[loc[s],loc[t],idx[r]]*L[r,l]
            for(v in 1:(d-d_c) )
            {
              nu_3[s,t,l] <- nu_3[s,t,l] + 3*a3[loc[s],idx[r],idx[v]]*L[r,t]*L[v,l]
              for(k in 1:(d-d_c))
              {
                nu_3[s,t,l] <- nu_3[s,t,l] +a3[idx[r],idx[v],idx[k]]*L[r,s]*L[v,t]*L[k,l]
              }
            }
          }
        }
      }
    }
    
  }
  return(list(nu_1 = nu_1,nu_3 = nu_3))
}
```

Let us now obtain the joint, bivariates and marginals of the skew-modal approximation of the posterior density. As with the Gaussian Laplace approximation, the resulting function are saved for future use. 

```r

# Joint skew-modal approximation (theta_0,theta_1,theta_2)
ske_sym_012 <- function(theta)
{
  centered <- (theta-la$m)
  skewness <- 0
  for(s in 1:3)
  {
    for(t in 1:3)
    {
      for(k in 1:3)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*nu_ttt[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*skewness
  2*mvtnorm::dmvnorm(theta, mean = la$m, sigma = la$V)*pnorm(skewness)
}

# Bivariate skew-modal (theta_0,theta_1)
coef_01 <- coef_marginal(c(1,2),la$V, nu_ttt)
ske_sym_01 <- function(theta)
{
  centered <- (theta-la$m[c(1,2)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_01$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_01$nu_1))
  2*mvtnorm::dmvnorm(theta, mean = la$m[c(1,2)], sigma = la$V[c(1,2),c(1,2)])*pnorm(skewness)
}

# Bivariate skew-modal (theta_0,theta_2)

coef_02 <- coef_marginal(c(1,3),la$V, nu_ttt)
ske_sym_02 <- function(theta)
{
  centered <- (theta- la$m[c(1,3)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_02$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_02$nu_1))
  2*mvtnorm::dmvnorm(theta, mean =  la$m[c(1,3)], sigma = la$V[c(1,3),c(1,3)])*pnorm(skewness)
}

# Bivariate skew-modal (theta_1,theta_2)

coef_12 <- coef_marginal(c(2,3),la$V, nu_ttt)

ske_sym_12 <- function(theta)
{
  centered <- (theta-la$m[c(2,3)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_12$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_12$nu_1))
  2*mvtnorm::dmvnorm(theta, mean = la$m[c(2,3)], sigma = la$V[c(2,3),c(2,3)])*pnorm(skewness)
}

# Marginal skew-modal theta_0 

coef_0 <- coef_marginal(1,la$V, nu_ttt)

ske_sym_0 <- function(theta)
{
  centered <- (theta-la$m[1])
  skewness <- sqrt(2*pi)/12*( coef_0$nu_1*centered + coef_0$nu_3*centered^3 )
  2*dnorm(theta, mean = la$m[1], sd = sqrt(la$V[1,1]))*pnorm(skewness)
}

# Marginal skew-modal theta_1

coef_1 <- coef_marginal(2,la$V, nu_ttt)

ske_sym_1 <- function(theta)
{
  centered <- (theta-la$m[2])
  skewness <- sqrt(2*pi)/12*( coef_1$nu_1*centered + coef_1$nu_3*centered^3 )
  2*dnorm(theta, mean = la$m[2], sd = sqrt(la$V[2,2]))*pnorm(skewness)
}

# Marginal skew-modal theta_2

coef_2 <- coef_marginal(3,la$V, nu_ttt)

ske_sym_2 <- function(theta)
{
  centered <- (theta-la$m[3])
  skewness <- sqrt(2*pi)/12*( coef_2$nu_1*centered + coef_2$nu_3*centered^3 )
  2*dnorm(theta, mean = la$m[3], sd = sqrt(la$V[3,3]))*pnorm(skewness)
}

# Save the different skew modal  approximations
save(la,nu_ttt,coef_01,coef_02,coef_12,coef_0, coef_1, coef_2,
     ske_sym_012,ske_sym_01,ske_sym_02,ske_sym_12,ske_sym_0,
     ske_sym_1,ske_sym_2, file="Logistic_SkewM_approx.RData")
```

## Mean-field Gaussian variational Bayes approximation
In the Supplementary Material (Table E.4) we study also the performance of two other common Gaussian approximations, namely, mean-field variational Bayes and Gaussian expectation propagation. This section provides the code to implement the former. To do it, we use the code developed by Durante and Rigon (2019), which can be obtained by downloading the source file `logistic.R` from `https://github.com/tommasorigon/logisticVB`. Once this source file has been saved into the current working directory, the mean and the covariance matrix of the mean field approximation are obtained using the following code.

```r
rm(list = ls())

# Loading the necessary functions from the file logistic.R
source("logistic.R")

# Load data
load("Cushings.RData")

# Specify the parameters for the CAVI algorithm

# Prior hyperparameters
prior <- list(mu = rep(0,3), Sigma = diag(25,3))

# Parameter settings
iter    <- 10^4  # Number of CAVI iterations
tau     <- 1     # Delay parameter
kappa   <- 0.75  # Forgetting rate parameter

# Get mean and covariance optimal parameters from CAVI  
paramsMF <- logit_CAVI(X = X, y = y, prior = prior) # CAVI algorithm
mf <- list()
mf$m <- paramsMF$mu
mf$V <- paramsMF$Sigma
```

To obtain the joint, bivariates and marginals of the mean-field approximation of the posterior density write:

```r
# Joint MF approximation (theta_0,theta_1,theta_2)
mf_theta012 <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf$m, sigma = mf$V)
}

# Bivariate MF approximation (theta_0,theta_1)
mf_theta_01 <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf$m[c(1,2)], sigma = mf$V[c(1,2),c(1,2)])
}

# Bivariate MF approximation (theta_0,theta_2)
mf_theta_02 <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf$m[c(1,3)], sigma = mf$V[c(1,3),c(1,3)])
}

# Bivariate MF approximation (theta_1,theta_2)
mf_theta_12 <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf$m[c(2,3)], sigma = mf$V[c(2,3),c(2,3)])
}

# Marginal MF approximation theta_0
mf_theta_0 <- function(param)
{
  dnorm(param, mean = mf$m[1], sd = sqrt(mf$V[1,1]))
}

# Marginal MF approximation theta_1
mf_theta_1 <- function(param)
{
  dnorm(param, mean = mf$m[2], sd = sqrt(mf$V[2,2]))
}

# Marginal MF approximation theta_2
mf_theta_2 <- function(param)
{
  dnorm(param, mean = mf$m[3], sd = sqrt(mf$V[3,3]))
}
```

Finally, we save these approximations for future use. 
```r
# Save the different MF approximations
save(mf, mf_theta012,mf_theta_01,mf_theta_02,mf_theta_12,
     mf_theta_0,mf_theta_1,mf_theta_2,file="Logistic_MF_approx.RData")
```

## Gaussian expectation propagation

As a last approximation, we consider Gaussian expectation propagation (EP). This approximation can be obtained using the Julia GaussianEP package (Barthelmé, 2024) downloadable at https://github.com/dahtah/GaussianEP.jl. For this reason, the code below must be run in the Julia environment (which can be dowloaded at https://julialang.org/downloads/). In particular, open Julia and then run the following code to load the quantities required for the approximation, obtain an estimate of the mean and the covariance matrix of the Gaussian EP approximation, and finally save these quantities for use in the R environment. 

```julia
# Load needed packages
using Pkg
Pkg.add("RCall")
Pkg.build("RCall")
Pkg.add("CSV")
Pkg.add("DelimitedFiles")
Pkg.build("DelimitedFiles")
pkg"add https://github.com/JuliaInterop/RCall.jl"
pkg"add https://github.com/dahtah/GaussianEP.jl"
using DelimitedFiles
using Statistics
using CSV
using RCall
using GaussianEP
using DelimitedFiles

# Use the R interface to retrive the Cushings dataset using the function file.choose()
# To run the code specify in [...] the path of Cushings.RData
R"Rdata <- load('.../Cushings.RData')"
@rget y;  @rget X;

# Epglm logit model
X = X'
y1 = Bool.(y)
G = ep_glm(X,y1,Logit(),τ=1/5^2) #EP estimation 

# Save mean and covariance matrix
m_ep = mean(G)
cova_ep = cov(G)

# to save in LogitCushing specify in [...] the path of the LogitCushing directory
file_path1 = ".../mean_ep_logistic"
file_path2 = ".../cov_ep_logistic"
writedlm(file_path1, m_ep, ',')
writedlm(file_path2, cova_ep, ',')
```
At this point, it is possible to use the new approximation in the R environment and define the joint, bivariates and marginals of the Gaussian EP approximation as done for the other approximations considered above.

```r
# Load mean 
m_ep <- c(read.csv("mean_ep_logistic", header = F, sep = ",")$V1)
# Load covariance
cov_ep <- as.matrix(read.csv("cov_ep_logistic", header = F, sep = ","))


# Joint, bivariates and marginal EP approximations can be obtained as follows

# Joint EP approximation (theta_0,theta_1,theta_2)
ep_theta012 <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep, sigma = cov_ep)
}

# Bivariate EP approximation (theta_0,theta_1)
ep_theta_01 <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep[c(1,2)], sigma = cov_ep[c(1,2),c(1,2)])
}

# Bivariate EP approximation (theta_0,theta_2)
ep_theta_02 <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep[c(1,3)], sigma = cov_ep[c(1,3),c(1,3)])
}

# Bivariate EP approximation (theta_1,theta_2)
ep_theta_12 <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep[c(2,3)], sigma = cov_ep[c(2,3),c(2,3)])
}

# Marginal EP approximation theta_0
ep_theta_0 <- function(theta)
{
  dnorm(theta, mean = m_ep[1], sd = sqrt(cov_ep[1,1]))
}

# Marginal EP approximation theta_1
ep_theta_1 <- function(theta)
{
  dnorm(theta, mean = m_ep[2], sd = sqrt(cov_ep[2,2]))
}

# Marginal EP approximation theta_2
ep_theta_2 <- function(theta)
{
  dnorm(theta, mean = m_ep[3], sd = sqrt(cov_ep[3,3]))
}

# Save the different EP approximations
save(m_ep,cov_ep, ep_theta012,ep_theta_01,ep_theta_02,ep_theta_12,
     ep_theta_0,ep_theta_1,ep_theta_2,file="Logistic_EP_approx.RData")
```

## Total variation distance between posterior distribution and different approximations

This section completes the work done in the previous part of the notebook by providing importance sampling estimates of the distance in total variation between the joint, bivariates and marginals of the target posterior density and their corresponding approximations. The importance sampling estimate is obtained by using as reference distributions independent Gaussians with mean the MAP and covariance matrix equal to the diagonal component of the inverse observed Fisher information inflated by 3 to guarantee the stability of the results. 

We start by cleaning the working environment, loading the approximations, and sampling from the reference density that will be used to implement importance sampling.

```r
rm(list = ls())
library(mvtnorm)
library(mclust)

# load the estimated posterior densities and the posterior approximations
load("estimated_posteriors_logistic.RData")
load("Logistic_Laplace_approx.RData")
load("Logistic_SkewM_approx.RData")
load("Logistic_EP_approx.RData")
load("Logistic_MF_approx.RData")


# For importance sampling we use a 3-dimensional independent Gaussian random
# variable with mean the MAP and covariance matrix equal to the diagonal component 
# of the inverse observed Fisher information inflated by 3

set.seed(1)
simulated_ex <- rmvnorm(10^4, mean = la$m, sigma = 3*diag(diag(la$V)))
```

Once this procedure is complete, it is possible to proceed by computing the importance sampling estimates.

### TV errors joint posterior $(\theta_0,\theta_1,\theta_2)$

```r
# TV distance joint posterior vs Laplace

diff_la012 <- function(x)
{
  1/2 * abs( post_theta_012(x)-la_theta012(x))/mvtnorm::dmvnorm(x, mean = la$m, sigma = 3*diag(diag(la$V)))
}

tv_la <- apply(simulated_ex,1,diff_la012)

# TV distance joint posterior vs skew-modal
diff_ske012 <- function(x)
{
  1/2 * abs( ske_sym_012(x) - post_theta_012(x))/mvtnorm::dmvnorm(x, mean = la$m, sigma = 3*diag(diag(la$V)))
}

tv_ske <- apply(simulated_ex,1,diff_ske012)


# TV distance joint posterior vs EP

diff_ep012 <- function(x)
{
  1/2 * abs( post_theta_012(x)-ep_theta012(x))/mvtnorm::dmvnorm(x, mean = la$m, sigma = 3*diag(diag(la$V)))
}

tv_ep <- apply(simulated_ex,1,diff_ep012)


# TV distance joint posterior vs mean field

diff_mf012 <- function(x)
{
  1/2 * abs( post_theta_012(x)-mf_theta012(x))/mvtnorm::dmvnorm(x, mean = la$m, sigma = 3*diag(diag(la$V)))
}

tv_mf <- apply(simulated_ex,1,diff_mf012)

```
### TV errors bivariate posterior $(\theta_0,\theta_1)$ 

```r
# TV bivariate posterior vs Laplace 

diff_la01 <- function(x)
{
  1/2 * abs( post_theta_01(x)-la_theta_01(x))/mvtnorm::dmvnorm(x, mean = la$m[1:2], sigma = 3*diag(diag(la$V))[1:2,1:2])
}

tv_la01 <- apply(simulated_ex[ ,1:2],1,diff_la01)

# TV bivariate posterior vs skew-modal 

diff_ske01 <- function(x)
{
  1/2 * abs( ske_sym_01(x) - post_theta_01(x))/mvtnorm::dmvnorm(x, mean = la$m[1:2], sigma = 3*diag(diag(la$V))[1:2,1:2])
}

tv_ske01 <- apply(simulated_ex[,1:2],1,diff_ske01)

# TV bivariate posterior vs EP 

diff_ep01 <- function(x)
{
  1/2 * abs( post_theta_01(x)-ep_theta_01(x))/mvtnorm::dmvnorm(x, mean = la$m[1:2], sigma = 3*diag(diag(la$V))[1:2,1:2])
}

tv_ep01 <- apply(simulated_ex[ ,1:2],1,diff_ep01)


# TV bivariate posterior vs MF

diff_mf01 <- function(x)
{
  1/2 * abs( post_theta_01(x)-mf_theta_01(x))/mvtnorm::dmvnorm(x, mean = la$m[1:2], sigma = 3*diag(diag(la$V))[1:2,1:2])
}

tv_mf01 <- apply(simulated_ex[ ,1:2],1,diff_mf01)
```
### TV errors bivariate posterior $(\theta_0,\theta_2)$

```r
# TV bivariate posterior vs Laplace

diff_la02 <- function(x)
{
  1/2 * abs( post_theta_02(x)-la_theta_02(x))/mvtnorm::dmvnorm(x, mean = la$m[c(1,3)], sigma = 3*diag(diag(la$V))[c(1,3),c(1,3)])
}

tv_la02 <- apply(simulated_ex[ ,c(1,3)],1,diff_la02)

# TV bivariate posterior vs skew-modal 

diff_ske02 <- function(x)
{
  1/2 * abs( ske_sym_02(x) - post_theta_02(x))/mvtnorm::dmvnorm(x, mean = la$m[c(1,3)], sigma = 3*diag(diag(la$V))[c(1,3),c(1,3)])
}

tv_ske02 <- apply(simulated_ex[,c(1,3)],1,diff_ske02)

# TV bivariate posterior vs EP 

diff_ep02 <- function(x)
{
  1/2 * abs( post_theta_02(x)-ep_theta_02(x))/mvtnorm::dmvnorm(x, mean = la$m[c(1,3)], sigma = 3*diag(diag(la$V))[c(1,3),c(1,3)])
}

tv_ep02 <- apply(simulated_ex[ ,c(1,3)],1,diff_ep02)

# TV bivariate posterior vs MF 

diff_mf02 <- function(x)
{
  1/2 * abs( post_theta_02(x)-mf_theta_02(x))/mvtnorm::dmvnorm(x, mean = la$m[c(1,3)], sigma = 3*diag(diag(la$V))[c(1,3),c(1,3)])
}

tv_mf02 <- apply(simulated_ex[ ,c(1,3)],1,diff_mf02)
```

### TV errors bivariate posterior $(\theta_1,\theta_2)$ 

```r
# TV bivariate posterior vs Laplace

diff_la12 <- function(x)
{
  1/2 * abs( post_theta_12(x)-la_theta_12(x))/mvtnorm::dmvnorm(x, mean = la$m[c(2,3)], sigma = 3*diag(diag(la$V))[c(2,3),c(2,3)])
}

tv_la12 <- apply(simulated_ex[ ,c(2,3)],1,diff_la12)

# TV bivariate posterior vs skew-modal

diff_ske12 <- function(x)
{
  1/2 * abs( ske_sym_12(x) - post_theta_12(x))/mvtnorm::dmvnorm(x, mean = la$m[c(2,3)], sigma = 3*diag(diag(la$V))[c(2,3),c(2,3)])
}

tv_ske12 <- apply(simulated_ex[ ,c(2,3)],1,diff_ske12)

# TV bivariate posterior vs EP

diff_ep12 <- function(x)
{
  1/2 * abs( post_theta_12(x)-ep_theta_12(x))/mvtnorm::dmvnorm(x, mean = la$m[c(2,3)], sigma = 3*diag(diag(la$V))[c(2,3),c(2,3)])
}

tv_ep12 <- apply(simulated_ex[ ,c(2,3)],1,diff_ep12)

# TV bivariate posterior vs MF

diff_mf12 <- function(x)
{
  1/2 * abs( post_theta_12(x)-mf_theta_12(x))/mvtnorm::dmvnorm(x, mean = la$m[c(2,3)], sigma = 3*diag(diag(la$V))[c(2,3),c(2,3)])
}

tv_mf12 <- apply(simulated_ex[ ,c(2,3)],1,diff_mf12)

```
### TV errors marginal posterior $\theta_0$

```r
# TV marginal posterior vs Laplace

diff_la0 <- function(x)
{
  1/2 * abs( post_theta_0(x)-la_theta_0(x))/dnorm(x, mean = la$m[1], sd = sqrt(3*diag(diag(la$V))[1,1]))
}

tv_la0 <- diff_la0(simulated_ex[ ,1])

# TV marginal posterior vs skew-modal

diff_ske0 <- function(x)
{
  1/2 * abs( ske_sym_0(x) - post_theta_0(x))/dnorm(x, mean = la$m[1], sd = sqrt(3*diag(diag(la$V))[1,1]))
}

tv_ske0 <- diff_ske0(simulated_ex[ ,1])

# TV marginal posterior vs EP

diff_ep0 <- function(x)
{
  1/2 * abs( post_theta_0(x)-ep_theta_0(x))/dnorm(x, mean = la$m[1], sd = sqrt(3*diag(diag(la$V))[1,1]))
}

tv_ep0 <- diff_ep0(simulated_ex[ ,1])

# TV marginal posterior vs MF

diff_mf0 <- function(x)
{
  1/2 * abs( post_theta_0(x)-mf_theta_0(x))/dnorm(x, mean = la$m[1], sd = sqrt(3*diag(diag(la$V))[1,1]))
}

tv_mf0 <- diff_mf0(simulated_ex[ ,1])

```
### TV errors marginal posterior $\theta_1$

```r
# TV marginal posterior vs Laplace 

diff_la1 <- function(x)
{
  1/2 * abs( post_theta_1(x)-la_theta_1(x))/dnorm(x, mean = la$m[2], sd = sqrt(3*diag(diag(la$V))[2,2]))
}

tv_la1 <- diff_la1(simulated_ex[ ,2])

# TV marginal posterior vs skew-modal

diff_ske1 <- function(x)
{
  1/2 * abs( ske_sym_1(x) - post_theta_1(x))/dnorm(x, mean = la$m[2], sd = sqrt(3*diag(diag(la$V))[2,2]))
}

tv_ske1 <- diff_ske1(simulated_ex[,2])

# TV marginal posterior vs EP

diff_ep1 <- function(x)
{
  1/2 * abs( post_theta_1(x)-ep_theta_1(x))/dnorm(x, mean = la$m[2], sd = sqrt(3*diag(diag(la$V))[2,2]))
}

tv_ep1 <- diff_ep1(simulated_ex[ ,2])

# TV marginal posterior vs MF 

diff_mf1 <- function(x)
{
  1/2 * abs( post_theta_1(x)-mf_theta_1(x))/dnorm(x, mean = la$m[2], sd = sqrt(3*diag(diag(la$V))[2,2]))
}

tv_mf1 <- diff_mf1(simulated_ex[ ,2])

```

### TV errors marginal posterior $\theta_2$ 

```r
# TV marginal posterior vs Laplace 

diff_la2 <- function(x)
{
  1/2 * abs( post_theta_2(x)-la_theta_2(x))/dnorm(x, mean = la$m[3], sd = sqrt(3*diag(diag(la$V))[3,3]))
}

tv_la2 <- diff_la2(simulated_ex[ ,3])

# TV marginal posterior vs skew-modal

diff_ske2 <- function(x)
{
  1/2 * abs( ske_sym_2(x) - post_theta_2(x))/dnorm(x, mean = la$m[3], sd = sqrt(3*diag(diag(la$V))[3,3]))
}

tv_ske2 <- diff_ske2(simulated_ex[ ,3])

# TV marginal posterior vs EP 

diff_ep2 <- function(x)
{
  1/2 * abs( post_theta_2(x)-ep_theta_2(x))/dnorm(x, mean = la$m[3], sd = sqrt(3*diag(diag(la$V))[3,3]))
}

tv_ep2 <- diff_ep2(simulated_ex[ ,3])

# TV marginal posterior vs VB 

diff_mf2 <- function(x)
{
  1/2 * abs( post_theta_2(x)-mf_theta_2(x))/dnorm(x, mean = la$m[3], sd = sqrt(3*diag(diag(la$V))[3,3]))
}

tv_mf2 <- diff_mf2(simulated_ex[ ,3])

```

Finally we create the table with the estimated total variation distance that is reported in Table E.4 of the Supplementary Material (Table 3 in the main article is a subset of it) and we save the results.

```r
# TV estimates
laplace_tv <-c(mean(tv_la),mean(tv_la01),mean(tv_la02),mean(tv_la12),
               mean(tv_la0), mean(tv_la1),mean(tv_la2))
skew_tv <- c(mean(tv_ske),mean(tv_ske01),mean(tv_ske02),
             mean(tv_ske12),mean(tv_ske0),mean(tv_ske1),mean(tv_ske2))
ep_tv <-  c(mean(tv_ep),mean(tv_ep01),mean(tv_ep02),mean(tv_ep12),
            mean(tv_ep0),mean(tv_ep1),mean(tv_ep2))
mf_tv <-  c(mean(tv_mf),mean(tv_mf01),mean(tv_mf02),mean(tv_mf12),
            mean(tv_mf0),mean(tv_mf1),mean(tv_mf2))

df_tv <- cbind(skew_tv,laplace_tv,ep_tv,mf_tv)

# Transform all the quantities in table form
colnames(df_tv) <- c("Skew-Modal", "Laplace", "EP", "MF")
rownames(df_tv) <- c("Joint","theta01","theta02", "theta12","theta0","theta1","theta2")
t(round(df_tv,2))

# Save the results
save(df_tv, file = "Total variation mcmc logistic")
```
## Highest posterior density intervals

We conclude this tutorial by providing the code to compare the quality of the Gaussian Laplace and skew-modal solutions in terms of accuracy in approximating the HPD intervals of the target posterior. First we clean the environment, load the necessary quantities and define a function which generates samples from the skew-modal approximation.
```r
# Load packages
rm(list = ls())
library(mvtnorm)
library(coda)

# Load the MCMC sample obtained with Stan and the parameters of the Laplace and 
# skew-modal approximations

load("MCMC_logistic.RData")
load("Logistic_Laplace_approx.RData")
load("Logistic_SkewM_approx.RData")

# Define the function to simulate from the skew-modal distribution

ske_sim <- function(nsim)
{
  # Function evelauting the skewness inducing factor of the 
  # skew modal approximation
  skewnees_calc <- function(theta)
  {
    skewness <- 0
    for(s in 1:3)
    {
      for(t in 1:3)
      {
        for(k in 1:3)
        {
          skewness <- skewness + theta[s]*theta[t]*theta[k]*nu_ttt[s,t,k]
        } 
      }
    }
    skewness <- sqrt(2*pi)*skewness/12
    skewness
  }
  
  # Simulation symmetric component 
  Z0 <- mvtnorm::rmvnorm(nsim, mean = rep(0,3), sigma = la$V)
  # Evaluate skewness factor
  skn <- pnorm( apply(Z0,1, skewnees_calc) )
  # Flip or not the simulated point
  flip <- 2*rbinom(nsim,1,skn)-1
  out <- Z0*flip
  # Add location parameter
  pst <- matrix(rep(la$m,nsim), ncol = 3, byrow = TRUE)
  out + pst
}
```

To obtain the HPD intervals, we use the `HPDinterval()` function from the `coda` library.
This function requires a sample from the posterior distribution for which the HPD intervals
are to be derived. For the posterior, we use the same sample used to estimate the
densities in the previous part of the notebook, while for the Laplace and skew-modal approximations  two samples of dimension $10^4$ are obtained.

```r
nsim <- 10^4
set.seed(1)
# sample skew-modal
sim_ske <- ske_sim(nsim)
# sample Laplace
sim_la <- mvtnorm::rmvnorm(n = nsim,mean = la$m, sigma = la$V)
```

At this point we can compute the mean absolute difference between the HPD intervals of the logistic posterior and the ones of the two approximation.


### $\alpha = 0.95$
```r
HPD_err_ske095 <- mean( abs(HPDinterval(as.mcmc(sim_ske,0.95)) - HPDinterval(as.mcmc(MCMC_logistic),0.95))) 
HPD_err_la095 <- mean( abs(HPDinterval(as.mcmc(sim_la,0.95)) - HPDinterval(as.mcmc(MCMC_logistic),0.95))) 

HPD_err095 <- c(HPD_err_ske095,HPD_err_la095)
names(HPD_err095) <- c("Skew-modal", "Laplace")
```
A table containing the results for all values of $\alpha$ can be obtained writing
```r
# Final table
round(HPD_err095,2)
```
