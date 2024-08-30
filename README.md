# Skewed Bernstein-von Mises theorem and skew modal distirbutions

This repository is associated with the article [Durante, Pozza and Szabo (2024). *SKEWED BERNSTEIN–VON MISES THEOREM
AND SKEW–MODAL APPROXIMATIONS*]([https://arxiv.org/abs/1802.09565](https://arxiv.org/pdf/2301.03038)). The **key contribution of this paper is outlined below**.

> We introduce a new and tractable class of asymmetric deterministic approximations of posterior distributions which arise from a novel treatment of a third–order version of the Laplace method. Under general assumptions which also account for misspecified models and non–i.i.d. settings, this novel family of approximations is shown to have a total variation distance from the target posterior whose rate of convergence improves by at least one order of magnitude the
one achieved by the Gaussian from the classical Bernstein–von Mises theorem

This repository provides **codes and tutorials to implement the inference methods associated with such a new result**. Here, the focus is on two illustrative applications regarding Logistic and Probit regression which are described in Section 5.2 of the paper.

- [`genes_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/genes_tutorial.md). This tutorial is discussed in Section --- of the paper and focuses on . The goal is  . These include the ** ---** by [Albert and Chib (1993)](https://www.jstor.org/stable/2290350) (`R` package `bayesm`) ----

- [`voice_tutorial.md`](https://github.com/danieledurante/ProbitSUN/blob/master/voice_tutorial.md). This tutorial implements the algorithms for posterior inference discussed above on a dataset with lower *p* and larger *n*. Specifically, the illustrative application considered here refers to a voice rehabilitation study available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation). As discussed in the article, when *p* decreases and *n* increases, the MCMC methods in `bayesm`, `rstan` and `LaplacesDemon`  are expected to progressively improve performance, whereas `Algorithm 1` may face more evident issues in computational time. This behavior is partially observed in this tutorial, although `Algorithm 1` is still competitive.


All the analyses are performed with a **MacBook Pro (OS X El Capitan, version 10.11.6)**, using a `R` version **3.4.1**. ADD 

