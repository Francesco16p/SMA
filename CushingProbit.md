# Probit regression on the Cushing dataset 

This file contains a complete description of how to reproduce some key results concerning the probit regression estimated on the Cushing dataset described in Section 5.2 of the main article and Appendix E.5 of the Supplementary Material (see Tables 3 and E.4). In particular, it provides code to implement both the joint and marginal skew-modal approximation to the model under consideration and to compare the quality of this approximation with that of the Gaussian Laplace, Gaussian variational Bayes, Gaussian expectation propagation, and partially factorized mean-field variational Bayes approximations.

Before starting, create a folder called `ProbitCushing` and set it as the working directory for the R environment. Within such a directory, save also the `Cushings.RData` file (see [`CushingLogistic.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md) for details on such a dataset and how is created).

## Probit regression with STAN

We first define a STAN model that estimates a Bayesian probit regression with `y` as the binary response variable and a design matrix equal to `X` using the `rstan` library. The coefficients are assumed to have independent Gaussian priors with zero mean and standard error `sd = 5`. Since obtaining i.i.d. samples from the posterior is not straightforward, we will use the STAN environment to generate 4 Hamiltonian Monte Carlo chains of length 10,000, which will allow us to obtain an accurate approximation of the posterior. The STAN model depends on five quantities: the sample size `N`, the number of parameters in the model `D`, the design matrix `X`, which also includes the intercept, the response variable `y`, and the standard error of the prior `sd`.

```
rm(list = ls())

// Probit regression with STAN

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
    y ~ bernoulli(Phi(X * theta));  // Using probit link
}

```

To estimate the model, we first load the data in the file `Cushings.RData`, which was previously created in the [`CushingLogistic.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md) tutorial. The code below loads the dataset, estimates the probit regression using 4 chains of length 10,000 of Hamiltonian Monte Carlo and saves the results for future use.

```r
# Load the required library
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load data
load("Cushings.RData")

# Create a list with y,X,N, D = 3 and sd = 5
df <- list( y = y, X = X, D = ncol(X), N = nrow(X), sd = 5)

# Fit the model
fit <- stan(model_code = stan_model_file, data = df, iter = 10^4, chains = 4, warmup = 5000, seed = 1)

# Extract MCMC results
MCMC_probit <- extract(fit)$theta

# Save MCMC output for future use
save(MCMC_probit,file = "MCMC_probit.RData")

```

### Estimate posterior densities using `mclust`

It is now possible to use Markov Chain Monte Carlo simulation to estimate the joint, bivariate and marginal posterior densities using the `mclust` library. These estimated densities are used as proxies for the exact posterior to evaluate the quality of different approximations. In the following, the numbers 0, 1 and 2 indicate which parameters the density refers to. To estimate such functions, write:

```r
library(mclust)

# Estimated joint posterior density (theta_0,theta_1,theta_2)
d012_probit <- densityMclust( MCMC_probit[,1:3])

post_theta_012_probit <- function(x)
{
  x <- matrix(x, ncol = 3)
  predict(d012_probit,x)
}

# Bivariate posterior density (theta_0, theta_1)
d01_probit <- densityMclust( MCMC_probit[,1:2])

post_theta_01_probit <- function(x)
{
  x <- matrix(x, ncol = 2)
  predict(d01_probit, x)
}

# Bivariate posterior density (theta_0, theta_2)
d02_probit <- densityMclust( MCMC_probit[,c(1,3)])

post_theta_02_probit <- function(x)
{
  x <- matrix(x, ncol = 2)
  predict(d02_probit, x)
}

# Bivariate posterior density (theta_1, theta_2)
d12_probit <- densityMclust( MCMC_probit[,c(2,3)])

post_theta_12_probit <- function(x)
{
  x <- matrix(x, ncol = 2)
  predict(d12_probit,x)
}

# Marginal density theta_0
d0_probit <- densityMclust( MCMC_probit[,1])

post_theta_0_probit <- function(x)
{
  predict(d0_probit, x)
}

# Marginal density theta_1
d1_probit <- densityMclust( MCMC_probit[,2])

post_theta_1_probit <- function(x)
{
  predict(d1_probit,x)
}

# Marginal density theta_2
d2_probit <- densityMclust( MCMC_probit[,3])

post_theta_2_probit <- function(x)
{
  predict(d2_probit, x)
}

```

Finally, we save the estimated densities for future use.

```r
# Save the estimated densities for future use
save(d0_probit,d1_probit,d2_probit,d01_probit,
     d02_probit,d12_probit,d012_probit,
     post_theta_0_probit,post_theta_1_probit,
     post_theta_2_probit,post_theta_01_probit,
     post_theta_02_probit,post_theta_12_probit,
     post_theta_012_probit,
     file="estimated_posteriors_probit.RData" )
```

## Laplace and skew-modal approximations

Let us now obtain both the Gaussian Laplace and the skew-modal approximations of the posterior distribution. This process involves the computation of the posterior mode (MAP), of the observed Fisher information and of the third log-likelihood derivative.

### Evaluation of the posterior mode

To obtain the MAP estimate, it is sufficient to maximise the posterior distribution using the `R` function `optim()`. We do this by clearing the global environment, loading the `mvtnorm` library and defining the function `log_post()`, which corresponds to the posterior distribution induced by the model under analysis.

```r
# Clear the workspace by removing all objects
rm(list = ls())

# Loading libraries
library(mvtnorm)

# Load dataset
load("Cushings.RData")
d <- dim(X)[2] # number of parameters

# Log-posterior distribution

sd <- 5 # standard error Gaussian prior
log_post_probit <- function(theta)
{
  eta <- X%*%theta
  p <- pnorm(eta)
  (sum(dbinom(y, size = 1, prob = p, log = T) )+
      mvtnorm::dmvnorm(theta, sigma = sd^2*diag(d), log =T ))
}
```

At this point, the MAP is obtained by writing 

```r
# MAP
map_probit <- optim(rep(0,d), function(k) -log_post_probit(k),method ="BFGS", hessian = T )

```

### Log-likelihood derivatives

The observed Fisher information evaluated at MAP is computed directly by the `optim()` function and stored in `map_probit$par`. Note that, as described in the main paper, this quantity is needed to obtain both Laplace and skew-modal approximations. We also define the function `trd_derivative_probit()` to compute the third derivative of the log-likelihood.

```r

# Observed Fisher information of the model

obsInf <-  map_probit$hessian

# Third log-likelihood derivative
trd_derivative_probit <- function(theta)
{
  eta <- X%*%theta
  dp_eta <- dnorm(drop(eta))*X
  p <- pnorm(eta)
  # three quantities that depends on the data
  q1 <- drop(y/p-(1-y)/(1-p))
  q2 <- drop( -y/p^2 -(1-y)/(1-p)^2)
  q3 <- drop( 2*y/p^3 -2*(1-y)/(1-p)^3  )
  # Here i write 3 instead of p to avoid confusion
  trd <- array(0,dim = c(3,3,3))
  for(k in 1:3)
  {
    for(r in 1:3)
    {
      for(s in 1:3)
      {
        for(i in 1:n)
        {
          trd[k,r,s] <- (trd[k,r,s] + 
                           q3[i]*dnorm(eta[i])^3*X[i,k]*X[i,r]*X[i,s]+
                           q2[i]*( -2*dnorm(eta[i])^2*eta[i]*X[i,k]*X[i,r]*X[i,s] )+   
                           q2[i]*(-dnorm(eta[i])^2*eta[i]*X[i,k]*X[i,r]*X[i,s])+
                           q1[i]*dnorm(eta[i])*(eta[i]^2-1)*X[i,k]*X[i,r]*X[i,s]
          )
        } 
      }
    }
  }
  trd
}

```

### Gaussian Laplace approximations

At this point it is possible to obtain the joint, bivariates and marginals of the Gaussian Laplace approximation of the posterior density by writing:

```r
la_probit <- list()
la_probit$m <- map_probit$par
la_probit$V <- solve(obsInf)


# Joint Gaussian Laplace approximation (theta_0,theta_1,theta_2)
la_theta012_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = la_probit$m, sigma = la_probit$V)
}

# Bivariate Gaussian Laplace approximation (theta_0,theta_1)
la_theta_01_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = la_probit$m[c(1,2)], sigma = la_probit$V[c(1,2),c(1,2)])
}

# Bivariate Gaussian Laplace approximation (theta_0,theta_2)
la_theta_02_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = la_probit$m[c(1,3)], sigma = la_probit$V[c(1,3),c(1,3)])
}

# Bivariate Gaussian Laplace approximation (theta_1,theta_2)
la_theta_12_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = la_probit$m[c(2,3)], sigma = la_probit$V[c(2,3),c(2,3)])
}

# Marginal Gaussian Laplace approximation theta_0
la_theta_0_probit <- function(theta)
{
  dnorm(theta, mean = la_probit$m[1], sd = sqrt(la_probit$V[1,1]))
}

# Marginal Gaussian Laplace approximation theta_1
la_theta_1_probit <- function(theta)
{
  dnorm(theta, mean = la_probit$m[2], sd = sqrt(la_probit$V[2,2]))
}

# Marginal Gaussian Laplace approximation theta_2
la_theta_2_probit <- function(theta)
{
  dnorm(theta, mean = la_probit$m[3], sd = sqrt(la_probit$V[3,3]))
  
}

# Save the different Laplace approximations
save(la_probit, la_theta012_probit,la_theta_01_probit,la_theta_02_probit,la_theta_12_probit,
     la_theta_0_probit,la_theta_1_probit,la_theta_2_probit,file="Probit_Laplace_approx.RData")
```

### Skew-modal approximations

To obtain joint, bivariates and marginals of the proposed skew-modal approximation of the posterior distribution, we follow the expressions reported in Sections 4.1-4.2 of the main paper. As a first step, we obtain the $d^3$-dimensional array of third log-likelihood derivatives evaluated at the MAP.

```r
# Third log-likelihood derivatives evaluated at MAP
nu_ttt_probit <- trd_derivative_probit(map_probit$par)
```
Once the third log-likelihood derivative is obtained, it is possible to compute the skewness-inducing coefficients of the bivariate and marginal skew-modal approximations, as reported in Section 4.2 of the paper. To do this, we define the function `coef_marginal()`, which takes the following parameters as input: `loc`, a numerical vector containing the position of the parameters to which the bivariate/marginal approximation refers; `a2`, which corresponds to the covariance matrix of the Gaussian component and `a3`, the array of the third log-likelihood derivatives.

```r
# Parameters:
# loc = position coef
# a2 = covariance matrix
# a3 = 3rd derivative

coef_marginal <- function(loc, a2, a3)
{
  d <- dim(a2)[2]
  d_c <- length(loc)  # dimension sub-set 
  idx <- setdiff(1:d, loc)
  
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
ske_sym_012_probit <- function(theta)
{
  centered <- (theta-la_probit$m)
  skewness <- 0
  for(s in 1:3)
  {
    for(t in 1:3)
    {
      for(k in 1:3)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*nu_ttt_probit[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*skewness
  2*mvtnorm::dmvnorm(theta, mean = la_probit$m, sigma = la_probit$V)*pnorm(skewness)
}

# Bivariate skew-modal (theta_0,theta_1)
coef_01_probit <- coef_marginal(c(1,2),la_probit$V, nu_ttt_probit)
ske_sym_01_probit <- function(theta)
{
  centered <- (theta-la_probit$m[c(1,2)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_01_probit$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_01_probit$nu_1))
  2*mvtnorm::dmvnorm(theta, mean = la_probit$m[c(1,2)], sigma = la_probit$V[c(1,2),c(1,2)])*pnorm(skewness)
}

# Bivariate skew-modal (theta_0,theta_2)

coef_02_probit <- coef_marginal(c(1,3),la_probit$V, nu_ttt_probit)
ske_sym_02_probit <- function(theta)
{
  centered <- (theta- la_probit$m[c(1,3)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_02_probit$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_02_probit$nu_1))
  2*mvtnorm::dmvnorm(theta, mean =  la_probit$m[c(1,3)], sigma = la_probit$V[c(1,3),c(1,3)])*pnorm(skewness)
}

# Bivariate skew-modal (theta_1,theta_2)

coef_12_probit <- coef_marginal(c(2,3),la_probit$V, nu_ttt_probit)

ske_sym_12_probit <- function(theta)
{
  centered <- (theta-la_probit$m[c(2,3)])
  skewness <- 0
  for(s in 1:2)
  {
    for(t in 1:2)
    {
      for(k in 1:2)
      {
        skewness <- skewness + centered[s]*centered[t]*centered[k]*coef_12_probit$nu_3[s,t,k]
      } 
    }
  }
  skewness <- sqrt(2*pi)/12*( skewness + sum(centered*coef_12_probit$nu_1))
  2*mvtnorm::dmvnorm(theta, mean = la_probit$m[c(2,3)], sigma = la_probit$V[c(2,3),c(2,3)])*pnorm(skewness)
}

# Marginal skew-modal theta_0 

coef_0_probit <- coef_marginal(1,la_probit$V, nu_ttt_probit)

ske_sym_0_probit <- function(theta)
{
  centered <- (theta-la_probit$m[1])
  skewness <- sqrt(2*pi)/12*( coef_0_probit$nu_1*centered + coef_0_probit$nu_3*centered^3 )
  2*dnorm(theta, mean = la_probit$m[1], sd = sqrt(la_probit$V[1,1]))*pnorm(skewness)
}

# Marginal skew-modal theta_1

coef_1_probit <- coef_marginal(2,la_probit$V, nu_ttt_probit)

ske_sym_1_probit <- function(theta)
{
  centered <- (theta-la_probit$m[2])
  skewness <- sqrt(2*pi)/12*( coef_1_probit$nu_1*centered + coef_1_probit$nu_3*centered^3 )
  2*dnorm(theta, mean = la_probit$m[2], sd = sqrt(la_probit$V[2,2]))*pnorm(skewness)
}

# Marginal skew-modal theta_2

coef_2_probit <- coef_marginal(3,la_probit$V, nu_ttt_probit)

ske_sym_2_probit <- function(theta)
{
  centered <- (theta-la_probit$m[3])
  skewness <- sqrt(2*pi)/12*( coef_2_probit$nu_1*centered + coef_2_probit$nu_3*centered^3 )
  2*dnorm(theta, mean = la_probit$m[3], sd = sqrt(la_probit$V[3,3]))*pnorm(skewness)
}

# Save the different skew modal  approximations
save(la_probit,nu_ttt_probit,coef_01_probit,coef_02_probit,
     coef_12_probit,coef_0_probit, coef_1_probit, coef_2_probit,
     ske_sym_012_probit,ske_sym_01_probit,ske_sym_02_probit,
     ske_sym_12_probit,ske_sym_0_probit,
     ske_sym_1_probit,ske_sym_2_probit, file="Probit_SkewM_approx.RData")
```

##  Mean-field Gaussian variational Bayes approximation
In the Supplementary Material (Table E.4) we study also the performance of three other approximations, namely Gaussian mean-field variational Bayes, partially factorized variational Bayes and Gaussian expectation propagation. This section provides the code to obtain the former. To implement the approximation, we use the code developed by Fasano, Durante and Zanella (2022), which can be obtained by downloading the source file `functionsVariational.R` at `https://github.com/augustofasano/Probit-PFMVB`.  Once this source file has been saved into the current working directory, the mean and the covariance matrix of the mean-field approximation are obtained using the following code.

```r
rm(list = ls())

# load the necessary functions from the file functionsVariational.R
source("functionsVariational.R")

# load data
load("Cushings.RData")

# To proceed we need to specify the parameters for the CAVI algorithm

# Prior hyper-parameters
nu2 <- 25
tolerance <- 1e-3 # tolerance to establish ELBO convergence

# get optimal parameters MFVB  
paramsMF = getParamsMF(X,y,nu2,tolerance,maxIter = 1e4)
mf_probit <- list()
mf_probit$m <- paramsMF$meanBeta
mf_probit$V <- diag(paramsMF$diagV)
```

To obtain the joint, bivariates and marginals of the mean-field approximation of the posterior density write:

```r
# Joint MF approximation (theta_0,theta_1,theta_2)
mf_theta012_probit <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf_probit$m, sigma = mf_probit$V)
}

# Bivariate MF approximation (theta_0,theta_1)
mf_theta_01_probit <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf_probit$m[c(1,2)], sigma = mf_probit$V[c(1,2),c(1,2)])
}

# Bivariate MF approximation (theta_0,theta_2)
mf_theta_02_probit <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf_probit$m[c(1,3)], sigma = mf_probit$V[c(1,3),c(1,3)])
}

# Bivariate MF approximation (theta_1,theta_2)
mf_theta_12_probit <- function(param)
{
  mvtnorm::dmvnorm(param, mean = mf_probit$m[c(2,3)], sigma = mf_probit$V[c(2,3),c(2,3)])
}

# Marginal MF approximation theta_0
mf_theta_0_probit <- function(param)
{
  dnorm(param, mean = mf_probit$m[1], sd = sqrt(mf_probit$V[1,1]))
}

# Marginal MF approximation theta_1
mf_theta_1_probit <- function(param)
{
  dnorm(param, mean = mf_probit$m[2], sd = sqrt(mf_probit$V[2,2]))
}

# Marginal MF approximation theta_2
mf_theta_2_probit <- function(param)
{
  dnorm(param, mean = mf_probit$m[3], sd = sqrt(mf_probit$V[3,3]))
}
```

Finally, we save these approximations for future use. 
```r
# Save the different MF approximations
save(mf_probit, mf_theta012_probit,mf_theta_01_probit,
     mf_theta_02_probit,mf_theta_12_probit,
     mf_theta_0_probit,mf_theta_1_probit,mf_theta_2_probit,
     file="Probit_MF_approx.RData")
```

## Partially factorized mean-field variational Bayes
To obtain the partially factorized mean-field approximation, we use as for the mean-field approximation the source code `functionsVariational.R` developed by Fasano, Durante and Zanella (2022).

```r
rm(list = ls())

# load the necessary functions
library(truncnorm)
library(mclust)
source("functionsVariational.R")

# load data
load("Cushings.RData")

set.seed(1)
nu2 <- 25
tolerance <- 1e-3 # tolerance to establish ELBO convergence

# estimate the approximation
paramsSMF = getParamsPFM(X=X,y=y,nu2=nu2,moments =TRUE, 
                         tolerance=tolerance,maxIter=1e4)
```
To sample from the approximation write:
```r
# sample from approximate partially-factorized approximation
set.seed(1) # set seed for reproducibility
simulated_pfvb <- sampleSUN_PFM(paramsPFM=paramsSMF,X=X,y=y,nu2=nu2,
                               nSample=10^4)
simulated_pfvb <- t(simulated_pfvb)
```
We then estimate the densities of the joint, bivariates and marginals of the partially factorised mean-field approximations using the `mclust` library as it was done for the posterior distribution.

```r
# Estimated joint PFVB approximation (theta_0,theta_1,theta_2)

d012pfvb <- densityMclust( simulated_pfvb[,1:3])
pfvb_theta012 <- function(param)
{
  param <- matrix(param, ncol = 3)
  predict(d012pfvb, param)
}

# Estimated bivariate PFVB approximation (theta_0,theta_1)

d01pfvb <- densityMclust( simulated_pfvb[,1:2])

pfvb_theta_01 <- function(param)
{
  param <- matrix(param, ncol = 2)
  predict(d01pfvb, param)
}

# Estimated bivariate PFVB approximation (theta_0,theta_2)

d02pfvb <- densityMclust( simulated_pfvb[,c(1,3)])

pfvb_theta_02 <- function(param)
{
  param <- matrix(param, ncol = 2)
  predict(d02pfvb, param)
}

# Estimated bivariate PFVB approximation (theta_1,theta_2)

d12pfvb <- densityMclust( simulated_pfvb[,c(2,3)])

pfvb_theta_12 <- function(param)
{
  param <- matrix(param, ncol = 2)
  predict(d12pfvb, param)
}

# Estimated marginal PFVB approximation theta_0

d0pfvb <- densityMclust( simulated_pfvb[,1])

pfvb_theta_0 <- function(param)
{
  predict(d0pfvb, param)
}

# Estimated marginal PFVB approximation theta_1

d1pfvb <- densityMclust( simulated_pfvb[,2])

pfvb_theta_1 <- function(param)
{
  predict(d1pfvb,param)
}

# Estimated marginal PFVB approximation theta_2

d2pfvb <- densityMclust( simulated_pfvb[,3])

pfvb_theta_2 <- function(param)
{
  predict(d2pfvb, param)
}

```

Finally, we save these approximations for future use. 
```r
# Save the different PFMF approximations
save(d012pfvb, d01pfvb,d02pfvb,d12pfvb,d0pfvb,d1pfvb,d2pfvb,
     simulated_pfvb, pfvb_theta012,pfvb_theta_01,pfvb_theta_02,
     pfvb_theta_12,pfvb_theta_0,pfvb_theta_1,pfvb_theta_2,
     file="Probit_PFVB_approx.RData")
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
Pkg.add("Distributions")
Pkg.build("Distributions")
pkg"add https://github.com/JuliaInterop/RCall.jl"
pkg"add https://github.com/dahtah/GaussianEP.jl"
using Statistics
using CSV
using RCall
using GaussianEP
using Distributions
using DelimitedFiles

# Use the R interface to retrive the Cushings dataset using the function file.choose()
# To run the code specify in [...] the path of Cushings.RData
R"Rdata <- load('.../Cushings.RData')"
@rget y;  @rget X;

# Epglm logit model
X = X'
y1 = Bool.(y)

# Define probit link
struct Probit <: GaussianEP.Likelihood end
function GaussianEP.log_dens(::Probit, η, y)
    return logcdf(Normal(0, 1), η) * y + logcdf(Normal(0, 1), -η) * (1 - y)
end

G = ep_glm(X,y1,Probit(),τ=1/5^2) #EP estimation 

# to save in ProbitCushing specify in [...] the path of the ProbitCushing directory
m_ep_probit = mean(G)
cova_ep_probit = cov(G)
file_path1 = ".../mean_ep_probit"
file_path2 = ".../cov_ep_probit"
writedlm(file_path1, m_ep_probit, ',')
writedlm(file_path2, cova_ep_probit, ',')
```
At this point, it is possible to use the new approximation in the R environment and define the joint, bivariates and marginals of the Gaussian EP approximation as done for the other approximations considered above.

```r
rm(list = ls())

# load mean ep probit 
m_ep_probit <- c(read.csv("mean_ep_probit", header = F, sep = ",")$V1)

# load covariance ep probit
cov_ep_probit <- as.matrix(read.csv("cov_ep_probit", header = F, sep = ","))


# Joint, bivariates and marginal EP approximations can be obtained as follows

# Joint EP approximation (theta_0,theta_1,theta_2)
ep_theta012_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep_probit, sigma = cov_ep_probit)
}

# Bivariate EP approximation (theta_0,theta_1)
ep_theta_01_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep_probit[c(1,2)], sigma = cov_ep_probit[c(1,2),c(1,2)])
}

# Bivariate EP approximation (theta_0,theta_2)
ep_theta_02_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep_probit[c(1,3)], sigma = cov_ep_probit[c(1,3),c(1,3)])
}

# Bivariate EP approximation (theta_1,theta_2)
ep_theta_12_probit <- function(theta)
{
  mvtnorm::dmvnorm(theta, mean = m_ep_probit[c(2,3)], sigma = cov_ep_probit[c(2,3),c(2,3)])
}

# Marginal EP approximation theta_0
ep_theta_0_probit <- function(theta)
{
  dnorm(theta, mean = m_ep_probit[1], sd = sqrt(cov_ep_probit[1,1]))
}

# Marginal EP approximation theta_1
ep_theta_1_probit <- function(theta)
{
  dnorm(theta, mean = m_ep_probit[2], sd = sqrt(cov_ep_probit[2,2]))
}

# Marginal EP approximation theta_2
ep_theta_2_probit <- function(theta)
{
  dnorm(theta, mean = m_ep_probit[3], sd = sqrt(cov_ep_probit[3,3]))
}

# Save the different EP approximations
save(m_ep_probit,cov_ep_probit, ep_theta012_probit,ep_theta_01_probit,
     ep_theta_02_probit,ep_theta_12_probit,
     ep_theta_0_probit,ep_theta_1_probit,ep_theta_2_probit,file="Probit_EP_approx.RData")

```

## Total variation distance between posterior distribution and different approximations

This section completes the work done in the previous part of the notebook by providing importance sampling estimates of the distance in total variation between the joint, bivariates and marginals of the target posterior density and their corresponding approximations. The importance sampling estimate is obtained by using as reference distributions independent Gaussians with mean the MAP and covariance matrix equal to the diagonal component of the inverse observed Fisher information inflated by 3 to guarantee the stability of the results.

We start by cleaning the working environment, loading the approximations, and sampling from the reference density that will be used to implement importance sampling.

```r
rm(list = ls())
library(mvtnorm)
library(mclust)

# First we load the estimated posterior densities and the posterior approximations

load("estimated_posteriors_probit.RData")
load("Probit_Laplace_approx.RData")
load("Probit_SkewM_approx.RData")
load("Probit_EP_approx.RData")
load("Probit_MF_approx.RData")
load("Probit_PFVB_approx.RData")

# For importance sampling we use a 3-dimensional independent Gaussian random
# variable with mean the MAP and covariance matrix equal to the diagonal component 
# of the inverse observed information infalted by 3

set.seed(1)
simulated_ex <- rmvnorm(10^4, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
```

Once this procedure is complete, it is possible to proceed by computing the importance sampling estimates.

### TV errors joint posterior $(\theta_0,\theta_1,\theta_2)$  

```r
# TV distance joint posterior vs Laplace

diff_la012 <- function(x)
{
  1/2 * abs( post_theta_012_probit(x)-la_theta012_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
}

tv_la <- apply(simulated_ex,1,diff_la012)

# TV distance joint posterior vs skew-modal
diff_ske012 <- function(x)
{
  1/2 * abs( ske_sym_012_probit(x) - post_theta_012_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
}

tv_ske <- apply(simulated_ex,1,diff_ske012)


# TV distance joint posterior vs EP

diff_ep012 <- function(x)
{
  1/2 * abs( post_theta_012_probit(x)-ep_theta012_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
}

tv_ep <- apply(simulated_ex,1,diff_ep012)


# TV distance joint posterior vs mean field

diff_mf012 <- function(x)
{
  1/2 * abs( post_theta_012_probit(x)-mf_theta012_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
}

tv_mf <- apply(simulated_ex,1,diff_mf012)

# TV distance joint posterior vs PFVB

diff_pfvb012 <- function(x)
{
  1/2 * abs( post_theta_012_probit(x)-pfvb_theta012(x))/mvtnorm::dmvnorm(x, mean = la_probit$m, sigma = 3*diag(diag(la_probit$V)))
}

tv_pfvb <- apply(simulated_ex,1,diff_pfvb012)

```
### TV errors bivariate posterior $(\theta_0,\theta_1)$ 

```r
# TV bivariate posterior vs Laplace 

diff_la01 <- function(x)
{
  1/2 * abs( post_theta_01_probit(x)-la_theta_01_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[1:2], sigma = 3*diag(diag(la_probit$V))[1:2,1:2])
}

tv_la01 <- apply(simulated_ex[ ,1:2],1,diff_la01)

# TV bivariate posterior vs skew-modal 

diff_ske01 <- function(x)
{
  1/2 * abs( ske_sym_01_probit(x) - post_theta_01_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[1:2], sigma = 3*diag(diag(la_probit$V))[1:2,1:2])
}

tv_ske01 <- apply(simulated_ex[,1:2],1,diff_ske01)

# TV bivariate posterior vs EP 

diff_ep01 <- function(x)
{
  1/2 * abs( post_theta_01_probit(x)-ep_theta_01_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[1:2], sigma = 3*diag(diag(la_probit$V))[1:2,1:2])
}

tv_ep01 <- apply(simulated_ex[ ,1:2],1,diff_ep01)


# TV bivariate posterior vs MF

diff_mf01 <- function(x)
{
  1/2 * abs( post_theta_01_probit(x)-mf_theta_01_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[1:2], sigma = 3*diag(diag(la_probit$V))[1:2,1:2])
}

tv_mf01 <- apply(simulated_ex[ ,1:2],1,diff_mf01)

# TV bivariate posterior vs PFVB

diff_pfvb01 <- function(x)
{
  1/2 * abs( post_theta_01_probit(x)-pfvb_theta_01(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[1:2], sigma = 3*diag(diag(la_probit$V))[1:2,1:2])
}

tv_pfvb01 <- apply(simulated_ex[ ,1:2],1,diff_pfvb01)
```
### TV errors bivariate posterior $(\theta_0,\theta_2)$

```r
# TV bivariate posterior vs Laplace

diff_la02 <- function(x)
{
  1/2 * abs( post_theta_02_probit(x)-la_theta_02_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(1,3)], sigma = 3*diag(diag(la_probit$V))[c(1,3),c(1,3)])
}

tv_la02 <- apply(simulated_ex[ ,c(1,3)],1,diff_la02)

# TV bivariate posterior vs skew-modal 

diff_ske02 <- function(x)
{
  1/2 * abs( ske_sym_02_probit(x) - post_theta_02_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(1,3)], sigma = 3*diag(diag(la_probit$V))[c(1,3),c(1,3)])
}

tv_ske02 <- apply(simulated_ex[,c(1,3)],1,diff_ske02)

# TV bivariate posterior vs EP 

diff_ep02 <- function(x)
{
  1/2 * abs( post_theta_02_probit(x)-ep_theta_02_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(1,3)], sigma = 3*diag(diag(la_probit$V))[c(1,3),c(1,3)])
}

tv_ep02 <- apply(simulated_ex[ ,c(1,3)],1,diff_ep02)

# TV bivariate posterior vs MF 

diff_mf02 <- function(x)
{
  1/2 * abs( post_theta_02_probit(x)-mf_theta_02_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(1,3)], sigma = 3*diag(diag(la_probit$V))[c(1,3),c(1,3)])
}

tv_mf02 <- apply(simulated_ex[ ,c(1,3)],1,diff_mf02)

# TV bivariate posterior vs PFVB

diff_pfvb02 <- function(x)
{
  1/2 * abs( post_theta_02_probit(x)-pfvb_theta_02(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(1,3)], sigma = 3*diag(diag(la_probit$V))[c(1,3),c(1,3)])
}

tv_pfvb02 <- apply(simulated_ex[ ,c(1,3)],1,diff_pfvb02)
```

### TV errors bivariate posterior $(\theta_1,\theta_2)$ 

```r
# TV bivariate posterior vs Laplace

diff_la12 <- function(x)
{
  1/2 * abs( post_theta_12_probit(x)-la_theta_12_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(2,3)], sigma = 3*diag(diag(la_probit$V))[c(2,3),c(2,3)])
}

tv_la12 <- apply(simulated_ex[ ,c(2,3)],1,diff_la12)

# TV bivariate posterior vs skew-modal

diff_ske12 <- function(x)
{
  1/2 * abs( ske_sym_12_probit(x) - post_theta_12_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(2,3)], sigma = 3*diag(diag(la_probit$V))[c(2,3),c(2,3)])
}

tv_ske12 <- apply(simulated_ex[ ,c(2,3)],1,diff_ske12)

# TV bivariate posterior vs EP

diff_ep12 <- function(x)
{
  1/2 * abs( post_theta_12_probit(x)-ep_theta_12_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(2,3)], sigma = 3*diag(diag(la_probit$V))[c(2,3),c(2,3)])
}

tv_ep12 <- apply(simulated_ex[ ,c(2,3)],1,diff_ep12)

# TV bivariate posterior vs MF

diff_mf12 <- function(x)
{
  1/2 * abs( post_theta_12_probit(x)-mf_theta_12_probit(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(2,3)], sigma = 3*diag(diag(la_probit$V))[c(2,3),c(2,3)])
}

tv_mf12 <- apply(simulated_ex[ ,c(2,3)],1,diff_mf12)

# TV bivariate posterior vs PFVB

diff_pfvb12 <- function(x)
{
  1/2 * abs( post_theta_12_probit(x)-pfvb_theta_12(x))/mvtnorm::dmvnorm(x, mean = la_probit$m[c(2,3)], sigma = 3*diag(diag(la_probit$V))[c(2,3),c(2,3)])
}

tv_pfvb12 <- apply(simulated_ex[ ,c(2,3)],1,diff_pfvb12)
```
### TV errors marginal posterior $\theta_0$

```r
# TV marginal posterior vs Laplace

diff_la0 <- function(x)
{
  1/2 * abs( post_theta_0_probit(x)-la_theta_0_probit(x))/dnorm(x, mean = la_probit$m[1], sd = sqrt(3*diag(diag(la_probit$V))[1,1]))
}

tv_la0 <- diff_la0(simulated_ex[ ,1])

# TV marginal posterior skew-modal

diff_ske0 <- function(x)
{
  1/2 * abs( ske_sym_0_probit(x) - post_theta_0_probit(x))/dnorm(x, mean = la_probit$m[1], sd = sqrt(3*diag(diag(la_probit$V))[1,1]))
}

tv_ske0 <- diff_ske0(simulated_ex[ ,1])

# TV marginal posterior vs EP

diff_ep0 <- function(x)
{
  1/2 * abs( post_theta_0_probit(x)-ep_theta_0_probit(x))/dnorm(x, mean = la_probit$m[1], sd = sqrt(3*diag(diag(la_probit$V))[1,1]))
}

tv_ep0 <- diff_ep0(simulated_ex[ ,1])

# TV marginal posterior vs MF

diff_mf0 <- function(x)
{
  1/2 * abs( post_theta_0_probit(x)-mf_theta_0_probit(x))/dnorm(x, mean = la_probit$m[1], sd = sqrt(3*diag(diag(la_probit$V))[1,1]))
}

tv_mf0 <- diff_mf0(simulated_ex[ ,1])

# TV marginal posterior vs PFVB

diff_pfvb0 <- function(x)
{
  1/2 * abs( post_theta_0_probit(x)-pfvb_theta_0(x))/dnorm(x, mean = la_probit$m[1], sd = sqrt(3*diag(diag(la_probit$V))[1,1]))
}

tv_pfvb0 <- diff_pfvb0(simulated_ex[ ,1])


```
### TV errors marginal posterior $\theta_1$

```r
# TV marginal posterior vs Laplace 

diff_la1 <- function(x)
{
  1/2 * abs( post_theta_1_probit(x)-la_theta_1_probit(x))/dnorm(x, mean = la_probit$m[2], sd = sqrt(3*diag(diag(la_probit$V))[2,2]))
}

tv_la1 <- diff_la1(simulated_ex[ ,2])

# TV marginal posterior vs skew-modal

diff_ske1 <- function(x)
{
  1/2 * abs( ske_sym_1_probit(x) - post_theta_1_probit(x))/dnorm(x, mean = la_probit$m[2], sd = sqrt(3*diag(diag(la_probit$V))[2,2]))
}

tv_ske1 <- diff_ske1(simulated_ex[,2])

# TV marginal posterior vs EP

diff_ep1 <- function(x)
{
  1/2 * abs( post_theta_1_probit(x)-ep_theta_1_probit(x))/dnorm(x, mean = la_probit$m[2], sd = sqrt(3*diag(diag(la_probit$V))[2,2]))
}

tv_ep1 <- diff_ep1(simulated_ex[ ,2])

# TV marginal posterior vs MF 

diff_mf1 <- function(x)
{
  1/2 * abs( post_theta_1_probit(x)-mf_theta_1_probit(x))/dnorm(x, mean = la_probit$m[2], sd = sqrt(3*diag(diag(la_probit$V))[2,2]))
}

tv_mf1 <- diff_mf1(simulated_ex[ ,2])

# TV marginal posterior vs PFVB 

diff_pfvb1 <- function(x)
{
  1/2 * abs( post_theta_1_probit(x)-pfvb_theta_1(x))/dnorm(x, mean = la_probit$m[2], sd = sqrt(3*diag(diag(la_probit$V))[2,2]))
}

tv_pfvb1 <- diff_pfvb1(simulated_ex[ ,2])
```

### TV errors marginal posterior $\theta_2$ 

```r
# TV marginal posterior vs Laplace 

diff_la2 <- function(x)
{
  1/2 * abs( post_theta_2_probit(x)-la_theta_2_probit(x))/dnorm(x, mean = la_probit$m[3], sd = sqrt(3*diag(diag(la_probit$V))[3,3]))
}

tv_la2 <- diff_la2(simulated_ex[ ,3])

# TV marginal posterior vs skew-modal

diff_ske2 <- function(x)
{
  1/2 * abs( ske_sym_2_probit(x) - post_theta_2_probit(x))/dnorm(x, mean = la_probit$m[3], sd = sqrt(3*diag(diag(la_probit$V))[3,3]))
}

tv_ske2 <- diff_ske2(simulated_ex[ ,3])

# TV marginal posterior vs EP 

diff_ep2 <- function(x)
{
  1/2 * abs( post_theta_2_probit(x)-ep_theta_2_probit(x))/dnorm(x, mean = la_probit$m[3], sd = sqrt(3*diag(diag(la_probit$V))[3,3]))
}

tv_ep2 <- diff_ep2(simulated_ex[ ,3])

# TV marginal posterior vs VB 

diff_mf2 <- function(x)
{
  1/2 * abs( post_theta_2_probit(x)-mf_theta_2_probit(x))/dnorm(x, mean = la_probit$m[3], sd = sqrt(3*diag(diag(la_probit$V))[3,3]))
}

tv_mf2 <- diff_mf2(simulated_ex[ ,3])

# TV marginal posterior vs PFVB 

diff_pfvb2 <- function(x)
{
  1/2 * abs( post_theta_2_probit(x)-pfvb_theta_2(x))/dnorm(x, mean = la_probit$m[3], sd = sqrt(3*diag(diag(la_probit$V))[3,3]))
}

tv_pfvb2 <- diff_pfvb2(simulated_ex[ ,3])

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
pfvb_tv <-  c(mean(tv_pfvb),mean(tv_pfvb01),mean(tv_pfvb02),mean(tv_pfvb12),
            mean(tv_pfvb0),mean(tv_pfvb1),mean(tv_pfvb2))

df_tv <- cbind(skew_tv,laplace_tv,ep_tv,mf_tv,pfvb_tv)

# Transform all the quantities in table form

colnames(df_tv) <- colnames(df_tv_sd) <- c("Skew-Laplace", "Laplace", "EP", "MF", "PFBV")
rownames(df_tv) <- rownames(df_tv_sd) <- c("Joint","theta01","theta02", "theta12","theta0","theta1","theta2")
t(round(df_tv,2))

# Save the results

save(df_tv, file = "Total variation mcmc Probit")
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

load("MCMC_probit.RData")
load("Probit_Laplace_approx.RData")
load("Probit_SkewM_approx.RData")

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
          skewness <- skewness + theta[s]*theta[t]*theta[k]*nu_ttt_probit[s,t,k]
        } 
      }
    }
    skewness <- sqrt(2*pi)*skewness/12
    skewness
  }
  
  # Simulation symmetric component 
  Z0 <- mvtnorm::rmvnorm(nsim, mean = rep(0,3), sigma = la_probit$V)
  # Evaluate skewness factor
  skn <- pnorm( apply(Z0,1, skewnees_calc) )
  # Flip or not the simulated point
  flip <- 2*rbinom(nsim,1,skn)-1
  out <- Z0*flip
  # Add location parameter
  pst <- matrix(rep(la_probit$m,nsim), ncol = 3, byrow = TRUE)
  out + pst
}
```

To obtain the HPD intervals, we use the `HPDinterval()` function from the `coda` library.
This function requires a sample from the posterior distribution for which the HPD intervals
are to be derived. For the posterior, we use the same sample used to estimate the
posterior densities in the previous part of the notebook, while for the Laplace
and skew-modal approximations, two samples of dimension $10^4$ are obtained.

```r
nsim <- 10^4
set.seed(1) # set seed for reproducibility
# sample skew_modal
sim_ske <- ske_sim(nsim)
# sample Laplace
sim_la <- mvtnorm::rmvnorm(n = nsim,mean = la_probit$m, sigma = la_probit$V)
```

At this point we can compute the mean absolute difference between the HPD intervals of the probit posterior and the ones of the two approximation.

### $\alpha = 0.95$
```r
HPD_err_ske095 <- mean( abs(HPDinterval(as.mcmc(sim_ske,0.95)) - HPDinterval(as.mcmc(MCMC_probit),0.95))) 
HPD_err_la095 <- mean( abs(HPDinterval(as.mcmc(sim_la,0.95)) - HPDinterval(as.mcmc(MCMC_probit),0.95))) 

HPD_err095 <- c(HPD_err_ske095,HPD_err_la095)
names(HPD_err095) <- c("Skew-modal", "Laplace")
```
A table containing the results can be obtained writing:
```r
# Final table
round(rbind(HPD_err08,HPD_err09,HPD_err095),2)
```
