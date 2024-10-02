# Skewed Bernstein-von Mises theorem and skew-modal approximations

This repository is associated with the article [Durante, Pozza and Szabo (2024). *Skewed Bernstein-von Mises theorem and skew-modal approximations*](https://arxiv.org/pdf/2301.03038). The **main contribution of the article is outlined below**.

> We introduce a new and tractable class of asymmetric deterministic approximations of posterior distributions which arise from a novel treatment of a third–order version of the Laplace method. Under general assumptions which also account for misspecified models and non–i.i.d. settings, this novel family of approximations is shown to have a total variation distance from the target posterior whose rate of convergence improves by at least one order of magnitude the one achieved by the Gaussian from the classical Bernstein–von Mises theorem.

This repository provides **tutorials to implement the inference methods associated with such a new result**. Here, the focus is on two illustrative applications regarding logistic and probit regression which are described in Section 5.2 of the main article and Appendix E.5 in the Supplementary Material.

- [`CushingLogit.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md). This tutorial reproduces the study on logistic regression. The aim is to compare the proposed skew-modal approximation with four common Gaussian approximations, namely Gaussian Laplace (see, e.g., [Gelman et al. (2013)](http://www.stat.columbia.edu/~gelman/book/) p. 318), mean-field variational Bayes ([Durante and Rigon (2019)](https://projecteuclid.org/journals/statistical-science/volume-34/issue-3/Conditionally-Conjugate-Mean-Field-Variational-Bayes-for-Logistic-Models/10.1214/19-STS712.full)), and Gaussian expectation propagation ([Minka, 2001](https://arxiv.org/abs/1301.2294)). The comparison is made by estimating the total variation distance between joint, bivariate and marginal posterior distributions and their corresponding approximations. 
  
- [`CushingProbit.md`](https://github.com/Francesco16p/SMA/blob/main/CushingLogistic.md). This tutorial reproduces the same study as `CushingLogit.md`, but in this case for a probit regression applied to the same dataset. In this case, we consider also an additional deterministic approximation. I.e., the partially factorized variational Bayes method of ([Fasano, Durante and Zanella (2022)](https://doi.org/10.1093/biomet/asac026)).

All the analyses are performed with a **MacBook Pro (OS Sonoma, version 14.5)**, using a `R` version **4.3.2**.

**IMPORTANT:** Although a seed is set at the beginning of each routine, the final output may be subject to slight variations depending on which version of the `R` packages has been used in the implementation of the code. This is due to possible internal changes of certain functions when the package version has been updated. **However, the magnitude of these minor variations is negligible and does not affect the final conclusions.**
