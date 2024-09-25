# Skewed Bernstein-von Mises theorem and skew modal distirbutions

This repository is associated with the article [Durante, Pozza and Szabo (2024). *Skewed Bernstein-von Mises theorem and skew-modal approximations*](https://arxiv.org/pdf/2301.03038). The **key contribution of this paper is outlined below**.

> We introduce a new and tractable class of asymmetric deterministic approximations of posterior distributions which arise from a novel treatment of a third–order version of the Laplace method. Under general assumptions which also account for misspecified models and non–i.i.d. settings, this novel family of approximations is shown to have a total variation distance from the target posterior whose rate of convergence improves by at least one order of magnitude the one achieved by the Gaussian from the classical Bernstein–von Mises theorem.

This repository provides **tutorials to implement the inference methods associated with such a new result**. Here, the focus is on two illustrative applications regarding logistic and probit regression which are described in Sections 5.2-D.4 of the paper.

- [`CushingLogit.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md). This tutorial reproduces the study on logistic regression discussed in Sections 5.2-D.4 of the paper. The aim is to compare the proposed skew-modal approximation with four common Gaussian approximations, namely Gaussian Laplace (see, e.g., [Gelman et al. (2013)](http://www.stat.columbia.edu/~gelman/book/) p. 318), mean-field variational Bayes ([Durante and Rigon (2019)](https://projecteuclid.org/journals/statistical-science/volume-34/issue-3/Conditionally-Conjugate-Mean-Field-Variational-Bayes-for-Logistic-Models/10.1214/19-STS712.full)), and Gaussian expectation propagation ([Minka,2001](https://arxiv.org/abs/1301.2294)). The comparison is made by estimating the total variation distance between joint, bivariate and marginal posterior distributions and their corresponding approximations. 
  
- [`CushingProbit.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md). This tutorial reproduces the same study as `CushingLogit.md`, but in this case for a probit regression estimated on the same dataset. An additional approximation, the partially factorised variational Bayes method of [Fasano, Durante and Zanella (2022)](https://arxiv.org/abs/1911.06743) is also considered.

All the analyses are performed with a **MacBook Pro (OS Sonoma, version 14.5)**, using a `R` version **4.3.2**.

