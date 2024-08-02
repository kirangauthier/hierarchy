# MCMC convergence and diagnostics 

## Diagnosing MCMC inference on simulated data 

`mrgsolve` is a Metrum based package that solves ODEs with covariates through the LSODE which is from the the Fortran ODE library, and uses R through the RCpp library. This is an excellent tool for us to simulate data from PKPD models with covariates with known values and analyze in NONMEM and Stan. 

It also gives us a good sense of prior predictive diagnostics (which are usually not included in discussion of MCMC convergence but are evident when things go wrong). I'll include a future writeup on the difference between *weakly informative* and *diffuse* priors and how they can frustrate even the most sophisticated samplers given a sufficiently complex model / sufficiently weak data. 

Try to have parameters exist on the same order of magnitude as eachother, from [here](https://discourse.mc-stan.org/t/brms-multilevel-nonlinear-model-model-fails-to-converge-when-converted-from-individual-level-to-multilevel/4691/3?u=kiran). And if chains really don't behave, set them at 0 with `inits = 0` even if they have a non-zero lower bound. Stan will respect it. 

### Inclusion of covariates 

Usually, the influence of a continuous covariate, denoted $\text{COVAR}$, on an individual parameter $\theta_i$ is modeled by 
$$
\theta_{i, \text{COVAR}} = \theta_i \left[ \frac{\text{COVAR}_i}{\text{median}(\text{COVAR})} \right]^\rho
$$
where the search for significant covariates on PK parameters can be performed by evaluating repeated regressions (in linear and nonlinear form of Eq. (1)) on HMC-fitted individual PK parameters. Based on the level of statistical significance ($p < 0.01$ or $p < 0.05$), covariate coefficients such as Gender, Body weight, Age, Height, Creatinine serum, Creatinine clearance, Urea, Alanine aminotransferase, ..., can be tested for each fit. If one, or many, exhibit a significant difference for more than half the gits, it is retained, which is consistent with the covariate selection performed in NONMEM ([Beal (2009)](https://www.semanticscholar.org/paper/NONMEM-User%E2%80%99s-Guides.-(1989%E2%80%932009)-Beal-Boeckmann/1964357daa9975ac959840262a810b2e0b39c8f4)) detailed in [Imbert et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/26207768/). 

### Convergence of MCMC sampling and diagnostic tools  

There are different convergence diagnostics that serve different purposes, let's go through them in order of development (generally get more complicated as we get closer to the present). 

For any and all of these diagnostics, I assume you have sampled **at least 2** chains (although I recommend a minimum of 4 chains). Sampling using only 1 chain is **very** dangerous and can lead to biased inference with no warning flags as statistics like the split $\hat{R}$ can only be used in a multi-chain setting). 

#### Pre-data regime 

1. Prior predictive modeling: a useful first step to see if your specified model and prior distribution can generate outcomes that look similar to your observed data, a very cost-effective way to see if there are any issues with the model or priors, $p(\theta)$, for more discussion see [Box, 1980](https://www.jstor.org/stable/2982063), [Gabry et al., 2019](https://arxiv.org/pdf/1709.01449.pdf), and section 7.3 of [Vehtari et al. 2020](https://arxiv.org/pdf/2011.01808.pdf)) and the ["Prior predictive modeling"](#Prior-predictive-modeling) section below 
   
2. Inference based on simulated results: Usually used in conjunction with prior predictive plots or in any situation where we have a good sense of what the "true" parameters would look like, allows us to test our inference engine and model when generating simulated data, see the below section on "[Inference based on simulated results](#Inference-based-on-simulated-results)" 
   
3. Exploring approximate inference algorithms: If at this point we recognize that inference is taking too long, we can immediately move to more approximate inference algorithms, which could help us focus our attention to the relevant areas of the posterior, see "[Exploring approximate inference algorithms](Exploring-approximate-inference-algorithms)" below 

#### Post-data regime 

1. Prior-posterior comparison: Comparing the prior and posterior is a simple and effective way to diagnose sampling and model issues, for the population parameters, $\theta$, the posterior should fall within the prior, $\omega$ and $\sigma$ can be a bit outside depending on the prior specification, see "[Prior-posterior comparison](#Prior-posterior comparison)" below 
   
2. Posterior summaries: Mean, MCSE, 95% quantiles, SD, 95% credible intervals, $\text{ESS}_\text{bulk}$, where we remember that the mean, SD, and 95% CI are only sensible for unimodal, normally distributed posteriors, 95% quantiles are much more robust, see "[Posterior summaries](#Posterior-summaries)" 

   1. **Note:** Interpretation of Bayesian posterior credible intervals is straightforward: a 95% CI is the central portion of the posterior that concentrates 95% of sampled values; given the observed data, the parameter falls into the interval with probability of 95%. 

3. Traceplots: monitors the evolution of the parameter estimates from MCMC draws over the iterations, see "[Traceplots](#Traceplots)" below 
   
4. Pair plots: used to identify collinearity between variables, non-identifiability between parameters (crescent-like shapes), or strong correlation that decreases the sampling efficiency across parameters 
   
5. HMC sampler diagnostics: sanity checking for the Bayesian inference algorithm, HMC provides the largest number of diagnostic statistics to probe, see the below section on "[HMC sampler diagnostics](#HMC-sampler-diagnostics)" 
   
6. Posterior predictive plots: generate predictions of simulated outcomes given the posterior samples, usually a simple and effective tool to diagnose model misspecification, useful anytime posterior samples are generated using observed data, see "[Posterior predictive checks](#Posterior-predictive-checks)" below. May be overly optimistic due to the "double use" of data for fitting and subsequent checking (see Vehtari et al., 2017, Paananen et al., 2020 for efficient implementations of leave-one-out cross-validation, LOO-CV, data fitting using importance sampling) 

   1. Answers questions such as: does the fitted model and its estimated parameters generate data similar to those observed experimentally? Are the individual and population variability such as the influence of covariates consistently modeled? by simulating multiple datasets according to the predictive distribution $y \sim y_\text{rep}$ and comparing them with the observed data visually or using summary statistics (test statistics)
   2. **Note:** generating posterior predictive plots and test statistics in this way there is a "double-use" of the data, first, the data is used to generate the posterior and secondly, these same posterior samples are used to generate the posterior predictive quantities. It's no different than any other MLE estimation procedure, in any case, it would be best to train the model on a set of data independent of the test set. 
   3. **Note:** To this last point, validation prediction evaluated on a separate dataset or unseen data (e.g., using WAIC) can test systematic discrepancies between model predictions and data would indicate an inadequate model predictive power / model failures. 

7. Residuals: Non-normalized residuals are cheap to collect and can also highlight errors of assigning inadequate compartmentalization or ODE-solver inaccuracies, should be zero-centered (non-biased), and there should be no significant difference in the RMSE or across the time domain, see "[Residuals](#Residuals)" below
   
8. Test statistics and goodness-of-fit tests: A computationally cheap way to test our model, say, we have a model where 80% of males at a dose level of 300 mg have a 30% change in a baseline output variable, we can then generate sample replicates from our posterior $\{\theta_s\}^{m=1}_M$ where $M$ is the number of replicates and $s$ is the number of draws in each posterior set. We then see what percentage of males experience a change of 30% or greater relative to the baseline for each of our sample replicates and importantly how many *posterior credible intervals* include 0.8, i.e. [0.71, 0.92] or [0.78, 0.83]. If a disproportionate amount do not include statistics seen in the data, this highlights model misspecification / failure or failure of inference. I suggest **not** to account for multiple hypothesis testing following [Andrew Gelman](https://statmodeling.stat.columbia.edu/2016/08/22/bayesian-inference-completely-solves-the-multiple-comparisons-problem/) as Bayesian hierarchical (multilevel) models already correct for multiple hypothesis testing 

   1. Posterior predictive:  $p(\tilde{y} \mid y) = \int p(\tilde{y} \mid y, \theta) \ p(\theta \mid y)$  computes the probability of unobserved data $\tilde{y}$ which can either be simulated from the posterior or are excluded from inference, better calibrated models+inference will have lower log posterior predictive scores 
   2. Goodness-of-fit tests: Depending on the goals of the analysis and the costs and benefits specific to the circumstances, we may tolerate that the model fails to capture certain aspects of the data or it may be essential to invest in improving the model. In general, we try to find “severe tests” (Mayo, 2018): checks that are likely to fail if the model would give misleading answers to the questions we care most about. 
      

9. Autocorrelation plots: measures the degree of correlation between draws of MCMC samples, useful when used in conjunction with the estimated number of effective samples, $N_\text{eff}$, usually parameters with high autocorrelation will have a lower $N_\text{eff}$. Not typically useful on it's own, and usually used to support earlier diagnostics. 

   

10. MCSE: gives an estimate of the number of significant digits we can confident for various expectations of the posterior samples (mean, variance, quantiles) we have drawn *given the number of samples we've drawn* 

    

11. Split $\hat{R}$: an extension of the original potential scale reduction factor*, $\hat{R}$, which calculates $\hat{R}$ on the split-half chains, an additional precaution Stan uses to detect non-stationary of individual chains**, see "[Split Rhat](#Split-Rhat)" below

    

12. Number of effective samples, $N_\text{eff}$: gives an estimate of the number of uncorrelated samples drawn from a Markov chain for a unit of time***, basically a measure of the efficiency of our sampler and a proxy for the correlation between parameters (stronger correlation = lower $N_\text{eff}$ = higher MCSE), see "[Neff and ESS](#Neff-and-ESS)" below

    1. Bulk-ESS and tail-ESS: The effective sample size (ESS) is proportional to the number of effective samples, $N_\text{eff}$, but doesn't take into account the amount of time used to generate the samples, usually calculated on the rank-normalized posterior draws for the bulk (bulk-ESS) and tail (tail-ESS) of the posterior samples, inference is generally considered unreliable if ESS < 400 for 2000 independent draws ([Vehtari et al. 2021](https://doi.org/10.1214%2F20-ba1221)) but I would generally say that if $N_\text{eff} / N > 0.5$ we have reason to doubt the inference 

       1. Bulk-ESS: assesses the efficiency of the samples in the bulk, estimating the *means* of the model parameters (should increase linearly with $N$ if sampling is well-behaved)

       2. Tail-ESS: assesses the efficiency of the samples in the tail, estimating the *quantiles* of the model parameters (should increase linearly with $N$ if sampling is well-behaved)

          

13. Divergences: divergent transitions usually occur when a set of "bad" parameters are passed into the likelihood, thankfully, the Hamiltonian Monte Carlo (HMC) sampler can alert us to this pathology$^\dagger$, it is useful to diagnose these with the `bayesplot::mcmc_parcoord()` and `bayesplot::mcmc_pairs()` plots, amongst others, which are discussed below in "[Divergences](Divergences)". Divergence-free sampling is always the goal and even a small number of divergences ($\sim 10$) can bias the inference and make the parameter estimates unreliable. Strategies to counteract divergences are discussed below. 

    

14. Rank plots and rank-normalization: Rank plots are an alternative to traceplots for visual sanity checking of convergence in posterior chains, and are histograms of posterior draws, ranked over all chains, and plotted for each chain. If all chains are sampling the same posterior distribution (stationary distribution), the rank histograms should note deviate significantly from a uniform distribution, for an example, see the below section on "[Rank plots](#Rank-plots)" 

    

15. Watanabe-Akaike information criterion (WAIC): Measures the out-of sample prediction penalizing for complexity, and is a useful diagnostic when comparing multiple models, more robust than the AIC, BIC, and Bayes factors, especially when used in conjunction with Pareto smoothed importance sampling (PSIS). See the below section on "[Measuring out of sample prediction and model comparison](#Measuring-out-of-sample-prediction-and-model-comparison)" and [Gelman et al. (2013)](https://link.springer.com/article/10.1007/s11222-013-9416-2)

    1. Pareto smoothed importance sampling (PSIS):  Used with WAIC and is a diagnostic tool that can be used to assess the reliability of the WAIC estimates. The PSIS diagnostic is based on importance sampling, which is a technique for estimating the expected value of a function under a probability distribution. The PSIS diagnostic is used to assess whether the importance sampling weights used to compute the WAIC are reliable or whether they may be biased or have a high variance. 
       

16. **Poststratification**: Adjusts the weight of the sample data to better represent the population of interest by stratifying the patient population by relevant variables such as age, gender, or disease status. Initially used for sample data where they may be too many White respondents relative to Black, old to young, the data is then weighted by the inverse of the proportion of the stratum in the population, so that strata with a smaller representation in the sample receive a larger weight. This is a new and important idea that could really separate BMS from the rest of the pack. 





## Further discussion and example diagnostics 

### Prior predictive modeling 

**Note:** It is possible to validate inference even when the posterior is not tractable, see this section from "On Bayesian Inference" 

> A more comprehensive approach than what we present in Section 4.1 is simulation-based calibration (SBC; Cook et al., 2006, Talts et al., 2020). In this scheme, the model parameters are drawn from the prior; then data are simulated conditional on these parameter values; then the model is fit to data; and finally the obtained posterior is compared to the simulated parameter values that were used to generate the data. By repeating this procedure several times, it is possible to check the coherence of the inference algorithm. The idea is that by performing Bayesian inference across a range of datasets simulated **using parameters drawn from the prior, we should recover the prior**. Simulation-based calibration is useful to evaluate how closely approximate algorithms match the theoretical posterior even in cases when the posterior is not tractable. 
>
> ...
> A serious problem with SBC is that it clashes somewhat with most modelers’ tendency to specify their priors wider than they believe necessary. The slightly conservative nature of weakly informative priors can cause the data sets simulated during SBC to occasionally be extreme. Gabry et al. (2019) give an example in which fake air pollution datasets were simulated where the pollution is denser than a black hole. These extreme data sets can cause an algorithm that works well on realistic data to fail dramatically. But this isn’t really a problem with the computation so much as a problem with the prior. One possible way around this is to ensure that the priors are very tight. However, a pragmatic idea is to keep the priors and compute reasonable parameter values using the real data. This can be done either through rough estimates or by computing the actual posterior. We then suggest widening out the estimates slightly and using these as a prior for the SBC. This will ensure that all of the simulated data will be as realistic as the model allows.

**Note:** It is generally known that as the number of predictors increases, we need stronger priors on model coefficients (or enough data) if we want to push the model away from extreme predictions. A useful approach is to consider priors on outcomes and then derive a corresponding joint prior on parameters (see, e.g., Piironen and Vehtari, 2017, and Zhang et al., 2020). More generally, joint priors allow us to control the overall complexity of larger parameter sets, which helps generate more sensible prior predictions that would be hard or impossible to achieve with independent priors.

As per Andrew Gelman's [advice](https://statmodeling.stat.columbia.edu/2013/11/21/hidden-dangers-noninformative-priors/), I do **not** advise the use of uniform (flat) priors in the case of weak data and hierarchical models, which can lead to unintended consequences as data per parameter becomes more sparse, and priors specified in low dimensional space become strong and highly informative in high dimensions, see [here](http://www.stat.columbia.edu/~gelman/research/published/deep.pdf) 



### Exploring approximate inference algorithms 

In the model building / model exploration phase it may be useful to limit the wasting of compute power as much as possible, to this end, we can use model stacking (Yao, Vehtari, and Gelman, 2020) using cross validation to assign weights to the different chains. It also has the capability of throwing out chains that are slow to converge and should not be assumed to be the same thing as Bayesian inference but does serve many of the same purposes. Usually very helpful to see where we should focus our attention in the posterior. 



### Prior-posterior comparison 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327105044225.png" alt="image-20230327105044225" style="zoom:80%;" />



### Posterior summaries 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327102319012.png" alt="image-20230327102319012" style="zoom:80%;" />



### Traceplots 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327103847464.png" alt="image-20230327103847464" style="zoom:67%;" />

#### Histograms based on posterior samples 

![image-20230327104038737](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327104038737.png)





### Inference based on simulated results 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327093802780.png" alt="image-20230327093802780" style="zoom:80%;" />



### Pair plots 

**Note:** These pair plots are generated based on simulated prior predictive data and are for the population parameters only, the second figure is across all $\theta$, $\omega$, and $\sigma$ parameters 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327093910379.png" alt="image-20230327093910379" style="zoom:67%;" />

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327103948748.png" alt="image-20230327103948748" style="zoom:100%;" />



### HMC sampler diagnostics 

Diagnostics we should be on the lookout for: average / SE integrator steps (lower is better), average / SE step size (a "Goldilocks" type parameter, not too large, not too small, larger generally better than smaller though), average / max ESS (higher is better), average / max $N_\text{eff}$ (higher is better), effective number of parameters $\rho_\text{eff}$ (lower is better), number of parameters for which $\hat{R} > 1.05$ (higher is better),  number of parameters for which $\hat{R} > 1.01$ (higher is better), number of divergences (anything > 0 is cause for concern), number of times reaching max tree depth (not the biggest worry but > 0 reduces computational efficiency), number of parameters (population and individual, for $\theta$, $\omega$, and $\sigma$), sampling time / chain,  maximum population PK score (probably lower is better, anything that's not out of realm of possibility), WAIC (-2 log score, lower is better). 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327092050135.png" alt="image-20230327092050135" style="zoom:67%;" />



### Posterior predictive checks 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327103103761.png" alt="image-20230327103103761" style="zoom:80%;" />

### Goodness-of-fit tests 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327181632453.png" alt="image-20230327181632453" style="zoom:67%;" />





### Residuals 

![image-20230327105118338](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327105118338.png)

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327103349481.png" alt="image-20230327103349481" style="zoom:50%;" />



### Split Rhat

*Sampling from the target distribution is only reliable if the parameters have converged to the stationary distribution, the best way to check if the parameters have converged to the same stationary distribution is to initialize many different chains. Although MCMC is asymptotically exact in the limit of infinite samples ($N \rightarrow \infty$), we have finite time to draw a finite number of samples, the potential scale reduction statistic, $\hat{R}$ ([Gelman and Rubin 1992](https://doi.org/10.1214/ss/1177011136)), assesses the convergence of multiple Markov chains on the same target distribution by comparing the intra-chain variance to the inter-chain variance. The $\hat{R}$ diagnostic provides an estimate of how much the variance of our parameter estimates would be reduced by running our chains longer. $\hat{R}$ values of 1 indicate good mixing of the chains, which occurs when all chains are sampling the same stationary distribution. 

The $\hat{R}$ is calculated from the between- (B / inter-) and within- (W / intra-) chain variances, where the marginal posterior variance is (over-) estimated by, 
$$
\widehat{\text{var}}^+(\theta \mid y) = \frac{N - 1}{N}W + \frac{1}{N}B
$$
where $N$ is the number of draws in each of $M$ chains. The potential scale reduction factor, $\hat{R}$, is then defined as, 
$$
\hat{R} = \sqrt{\frac{\widehat{\text{var}}^+(\theta \mid y)}{W}}
$$
where $W$ is the averaged sum of squares of the within-chain variance of every chain ([Gelman's BDA3 2014](http://www.stat.columbia.edu/~gelman/book/)). If the potential scale reduction is high, then we have reason to believe that proceeding with further simulations may improve our inference about the target distribution of the associated scalar estimand. 



**A simple example: two chains with complementary increasing and decreasing values will yield $\hat{R}$ close to 1, even though they are not mixing, see below for a demo 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230325133234651.png" alt="image-20230325133234651" style="zoom:50%;" />

### Neff and ESS

***Remember that Markov chains *always* have a memory of 1 step, that is, the $t^{th}$ is necessarily correlated to the $(t-1)^{th}$ step. In certain cases (i.e. in cases where the posterior has strong correlation between the parameters, and very commonly in hierarchical models), the memory can extend to be much further as sequential samples are all drawn from the same "neighborhood" of probability mass. In this case, we can simply thin or remove every $n^{th}$ sample from the chain to lessen the correlation between sequential draws. There are other tricks to increasing the value of $N_\text{eff}$ such as non-centered parameterization or broadening / tightening the priors but it is a helpful diagnostic to check that $N_\text{eff}$ is not *too* low (i.e. less than 50% of the number of originally drawn samples $N_\text{eff} / N > 0.5$). 

![image-20230327102406205](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327102406205.png)

 

### Rank plots 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327093149220.png" alt="image-20230327093149220" style="zoom:80%;" />





### Divergences 

$^\dagger$Divergent transitions are usually due to strong and highly varying posterior curvature that frustrates the numerical (leapfrog) sampler (think about trying to drive along a road with a strong, but varying curve, go too fast and you can drive off the edge (divergence)) see an example of plots referenced above that visualize divergent transitions. Only some of the useful functions are included here, the entire list that can be useful to diagnose biased inference would be `bayesplot::mcmc_parcoord()`, `bayesplot::mcmc_pairs()`, `bayesplot::mcmc_scatter()`, `bayesplot::mcmc_trace()`, `bayesplot::mcmc_nuts_divergence()` (specific to samples drawn from a Stan NUTS sampler), `bayesplot::mcmc_nuts_energy()` (specific to samples drawn from a Stan NUTS sampler) 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230325135458408.png" alt="image-20230325135458408" style="zoom:100%;" />

![image-20230325135529885](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230325135529885.png)

![image-20230325135611434](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230325135611434.png)

## Cross-validation of "stacked" models 

> We can use cross validation to compare models fit to the same data (Vehtari et al., 2017). When performing model comparison, if there is non-negligible uncertainty in the comparison (Sivula et al., 2020), **we should not simply choose the single model with the best cross validation results, as this would discard all the uncertainty from the cross validation process**. Instead, we can maintain this information and use stacking to **combine inferences** using a weighting that is set up to **minimize cross validation error** (Yao et al., 2018b). We think that stacking makes more sense than traditional Bayesian model averaging, as the latter can depend strongly on aspects of the model that have minimal effect on predictions. For example, for a model that is well informed by the data and whose parameters are on unit scale, changing the prior on parameters from $\text{normal}(0, 10)$ to $\text{normal}(0, 100)$ will divide the marginal likelihood by roughly $10^k$ (for a model with $k$ parameters) while keeping all predictions essentially the same. In addition, stacking takes into account the joint predictions and works well when there are a large number of similar but weak models in the candidate model list. 
>
> In concept, stacking can sometimes be viewed as pointwise model selection. When there are two models and the first model outperforms the second model 20% of the time, the stacking weights will be close to (0.2, 0.8). In light of this, stacking fills a gap between independent-error oriented machine learning validation and the grouped structure of modern big data. Model stacking is therefore also an indicator of heterogeneity of model fitting, and this suggests we can further improve the aggregated model with a hierarchical model, so that the stacking is a step toward model improvement rather than an end to itself. In the extreme, model averaging can also be done so that different models can apply to different data points (Kamary et al., 2019, Pirš and Štrumbelj, 2019).
>
> ... 
> In our own applied work we have not generally had many occasions to perform this sort of model averaging, as we prefer continuous model expansion, but there will be settings where users will reasonably want to make predictions averaging over competing Bayesian models, as in Montgomery and Nyhan (2010).



### Projection predictive variable selection should be used over minimizing the cross-validation scores for candidate models who can be described by special cases of a single expanded model 

> There are many problems, for example in linear regression with several potentially relevant predictors, where **many candidate models are available**, all of which can be described as **special cases of a single expanded model**. If the number of candidate models is large, we are often **interested in finding a comparably smaller model that has the same predictive performance as our expanded model**. This leads to the problem of **predictor (variable) selection**. If we have many models making similar predictions, selecting one of these models based on minimizing cross validation error would lead to **overfitting** and **suboptimal** model choices (Piironen and Vehtari, 2017). In contrast, projection predictive variable selection has been shown to be stable and reliable in finding smaller models with good predictive performance (Piironen and Vehtari, 2017, Piironen et al., 2020, Pavone et al., 2020). While searching through a big model space is usually associated with the danger of overfitting, the projection predictive approach avoids it by examining only the projected submodels based on the expanded model’s predictions and not fitting each model independently to the data. In addition to variable selection, projection predictive model selection can be used for structure selection in generalized additive multilevel models (Catalina et al., 2020) and for creating simpler explanations for complex nonparametric models (Afrabandpey et al., 2020).

### Measuring out of sample prediction and model comparison 



Has it been explained why the k^k^﻿ values are highest for the **hierarchical model**? **Fewer observations affect each parameter, magnifying the effect of "outliers" on the posterior, resulting in more different posteriors, making IS  more difficult.**

Watanabe-Akaike information criterion (WAIC) ([Gelman 2014](https://doi.org/10.1007/s11222-013-9416-2)) which is a fully Bayesian information criterion that uses the whole posterior distribution, allowing us to measure the consequential effect of our priors, it is defined by 

$$ \text{WAIC} = -2(\text{lppd} - \text{p}_\text{eff}) $$

where $\text{lppd}$ is the log pointwise predictive density (log-likelihood) for a new data point (an accuracy term) and $\text{p}_\text{eff}$ is the effective number of parameters (a penalty term for overfitting).  WAIC is an indicator for the comparison of point-wise out-of-sample predictive accuracy of Bayesian models, based on the whole estimated posterior distribution. 

**Note:** AIC and BIC only use point estimates, and that the relative difference of two WAICs is useful as a measure of the strength of evidence for two competing models, the lower the value of the WAIC, the better the fit. Relative differences of 10 or more indicate that the competing model has almost very unlikely and has virtually no support (see [Burnham and Anderson 2002](https://link.springer.com/book/10.1007/b97636)). 

**Note:** The $\text{lppd}$ is proportional to the mean squared error for normal models with constant variance and the effective number of parameters is calculated by summing over all the posterior variance of the log predictive density for each data point.



We can also use the `loo` package in R to measure the leave-one-out cross-validation metric, and gives a couple of different ways that we can compute the WAIC and in their words, 

> WAIC stands for "Widely Applicable Information Criterion" and is a measure of model fit and complexity in Bayesian data analysis. The WAIC is calculated by comparing the log-likelihood of the data under a model to a penalized estimate of the effective number of parameters in the model. The idea is to balance the goodness of fit of the model to the data with the complexity of the model, so that the WAIC can be used to compare different models and choose the one that best balances these trade-offs. In addition to computing the WAIC, `loo` also provides methods for computing the Pareto smoothed importance sampling (PSIS) diagnostic, which is used to assess the reliability of the WAIC estimates.
>
> The `loo` package is particularly useful for comparing Bayesian models with different sets of predictors or different model structures. By comparing the WAIC values of different models, researchers can identify the models that best balance goodness of fit and model complexity. This can be especially useful in fields like pharmacology, where researchers often use complex models to describe complex pharmacokinetic processes.



#### Assessing the WAIC using Pareto smoothed importance sampling (PSIS) 

PSIS stands for "Pareto Smoothed Importance Sampling" and is a diagnostic tool that can be used to assess the reliability of the WAIC estimates. The PSIS diagnostic is based on importance sampling, which is a technique for estimating the expected value of a function under a probability distribution. The PSIS diagnostic is used to assess whether the importance sampling weights used to compute the WAIC are reliable or whether they may be biased or have a high variance.

The PSIS diagnostic works by comparing the estimated importance sampling weights to a smoothed version of the Pareto distribution, which is a distribution that is often used to model extreme events or outliers. If the estimated importance sampling weights are well-behaved and have a reasonable shape, then they will match the smoothed Pareto distribution closely. However, if the weights are biased or have a high variance, then they may not match the smoothed Pareto distribution and the PSIS diagnostic will flag this as a potential problem.

By using the WAIC and the PSIS diagnostic together, we identify the best-fitting model and assess the reliability of the model selection procedure. It is more reliable than conventional methods such as AIC and BIC, *especially* for complex models and small samples sizes (both of which are common for Phase II data). WAIC with PSIS can also be used with a wide range of models, including hierarchical models, mixed-effects models, and models with non-Gaussian likelihoods.

#### Comparison to Bayes factors 

Bayes factors, on the other hand, are a frequentist method that compare the evidence for two or more competing models based on their marginal likelihoods. Bayes factors can be used to quantify the relative support for different models, and can be interpreted as the odds ratio of the models given the data. However, Bayes factors are *sensitive to the prior distribution*, and the choice of the prior can have a large effect on the Bayes factor results.

Bayes factors can also be misleading, especially when the data are weak or the models being compared are very different. In some cases, the Bayes factor can favor the wrong model or give ambiguous results. I strongly advocate for the use of the WAIC over Bayes factors. 

### Challenges with measuring predictive model fit 

The current state of the art of measurement of predictive model fit remains unsatisfying. Formulas such as AIC, DIC, and WAIC fail in various examples: AIC does not work in settings with strong prior information, DIC gives nonsensical results when the posterior distribution is not well summarized by its mean, and WAIC relies on a data partition that would cause difficulties with structured models such as for spatial or network data. Cross-validation is appealing but can be computationally expensive and also is not always well defined in dependent data settings.

Thus we see the value of the methods described here, for all their flaws. Right now our preferred choice is cross-validation. Bayesian cross-validation is asymptotically equal to WAIC. Pareto-smoothed importance sampling LOO-CV is computationally as efficient as WAIC, but more robust in the finite case with weak priors or influential observations.



## Modeling and transforming pharmacological parameters 

From [On Bayesian Workflow](https://arxiv.org/abs/2011.01808), 

> For example, in a problem in pharmacology (Weber et al., 2018) we had a parameter that we expected would take on values of approximately 50 on the scale of measurement; following the principle of scaling we might set up a model on log(θ/50), so that 0 corresponds to an interpretable value (50 on the original scale) and a difference of 0.1, for example, on the log scale corresponds to increasing or decreasing by approximately 10%. This sort of transformation is not just for ease of interpretation; it also sets up the parameters in a way that readies them for effective hierarchical modeling. As we build larger models, for example by incorporating data from additional groups of patients or additional drugs, it will make sense to allow parameters to vary by group (as we discuss in Section 12.5), and partial pooling can be more effective on scale-free parameters. For example, a model in toxicology required the volume of the liver for each person in the study. Rather than fitting a hierarchical model to these volumes, we expressed each as the volume of the person multiplied by the proportion of volume that the liver; we would expect these scale-free factors to vary less across people and so the fitted model can do more partial pooling compared to the result from modeling absolute volumes. The scaling transformation is a decomposition that facilitates effective hierarchical modeling.
>
> In many cases we can put parameters roughly on unit scale by using logarithmic or logit transformations or by standardizing, subtracting a center and dividing by a scale. If the center and scale are themselves computed from the data, as we do for default priors in regression coefficients in rstanarm (Gabry et al., 2020a), we can consider this as an approximation to a hierarchical model in which the center and scale are hyperparameters that are estimated from the data. More complicated transformations can also serve the purpose of making parameters more interpretable and thus facilitating the use of prior information; Riebler et al. (2016) give an example for a class of spatial correlation models, and Simpson et al. (2017) consider this idea more generally. 



### Estimating the variance when $t$-distributed priors are used instead of normally distributed priors 

> Priors must be specified for each model within a workflow. An expanded model can require additional thought regarding parameterization. For example, when going from $\text{normal}(\mu, \sigma)$ to a $t$-distribution $t_\nu(\mu, \sigma)$ with $\nu$ degrees of freedom, we have to be careful with the prior on $\sigma$. The scale parameter $\sigma$ looks the same for the two models, but the variance of the $t$-distribution is actually $\frac{\nu}{\nu - 2}\sigma^2$ rather than $\sigma^2$ . Accordingly, if $\nu$ is small, $\sigma$ is no longer close to the residual standard deviation. 



## Errors with priors as we go to high dimensions - the importance of prior predictive checking 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327151631466.png" alt="image-20230327151631466" style="zoom:80%;" />



## Useful examples of Bayesian convergence, diagnostics, and applications in PMx and beyond 

1. [Bayesian Pharmacometrics Analysis of Baclofen for Alcohol Use Disorder](https://www.biorxiv.org/content/10.1101/2022.10.25.513675v1.full.pdf+html)
   1. Analysis of a 1-compartment PK model where data is in NONMEM format! 
2. [An Introduction to Bayesian Multilevel Models Using brms: A Case Study of Gender Effects on Vowel Variability in Standard Indonesian](https://psyarxiv.com/guhsa/) 
   1. Paul Bürkner is a Stan developer and the lead author on the `brms` package for hierarchical Bayesian modeling in R 
3. [On Bayesian Workflow](https://arxiv.org/abs/2011.01808)
   1. Aki Vehtari is one of the pre-eminent probabilistic programming statisticians, this paper provides an excellent basis for diagnosing with Bayesian inference 



## Bayesian Workflow 

Bayesian inference is only the formulation and computation of conditional probability of probability densities, i.e. $p(\theta \mid y) \propto p(y \mid \theta) p(\theta)$. Bayesian workflow includes the three steps of model building, inference, and model checking / improvement, along with comparison of different models. 

> Statistics is all about uncertainty. In addition to the usual uncertainties in the data and model parameters, we are often uncertain whether we are fitting our models correctly, uncertain about how best to set up and expand our models, and uncertain in their interpretation.

### Priors as *constraints* in a sequence of models 

> Another way to view a prior distribution, or a statistical model more generally, is as a constraint. For example, if we fit a linear model plus a spline or Gaussian process, $y = b_0 + b_1 x + g(x) + \text{error}$, where the nonlinear function $g$ has bounded variation, then with a strong enough prior on the $g$, we are fitting a curve that is close to linear. The prior distribution in this example could represent prior information, or it could just be considered as part of the model specification, if there is a desire to fit an approximately linear curve. Simpson et al. (2017) provide further discussion on using prior distributions to shrink towards simpler models. This also leads to the more general point that priors are just like any other part of a statistical model in which they can serve different purposes. Any clear distinction between model and prior is largely arbitrary and often depends mostly on the conceptual background of the one making the distinction. 
>
> The amount of prior information needed to get reasonable inference depends strongly on the role of the parameter in the model as well as the depth of the parameter in the hierarchy (Goel and DeGroot, 1981). For instance, parameters that mainly control central quantities (such as the mean or the median) typically tolerate vague priors more than scale parameters, which again are more forgiving of vague priors than parameters that control tail quantities, such as the shape parameter of a generalized extreme value distribution. When a model has a hierarchical structure, parameters that are closer to the data are typically more willing to indulge vague priors than parameters further down the hierarchy.
>
> In Bayesian workflow, **priors are needed for a sequence of models**. Often as the model becomes more complex, all of the priors need to become tighter. The following simple example of multilevel data (see, for example, Raudenbush and Bryk, 2002) shows why this can be necessary.

#### Expansion from simple model (no-group) to group-modeling and the effect of destabilization on parameter estimation 

> Consider a workflow where the data are $y_{ij}$, $i = 1, \dots, n_j$, $j = 1, \dots, J$. Here $i$ indexes the observation and $j$ indexes the group. Imagine that for the first model in the workflow we elect to ignore the group structure and use a simple normal distribution for the deviation from the mean. In this case some of the information budget will be spent on estimating the overall mean and some of it is spent on the observation standard deviation. If we have a moderate amount of data, the mean will be approximately $\bar{y} = \sum_{i=1}^n y_i / n$ regardless of how weak the prior is. Furthermore, the predictive distribution for a new observation will be approximately $\text{normal}(\bar{y}, s)$, where $s$ is the sample standard deviation. Again, this will be true for most sensible priors on the observation standard deviation, regardless of how vague they are. 
>
> If the next step in the workflow is to allow the mean to vary by group using a multilevel model, then the information budget still needs to be divided between the standard deviation and the mean. However, the model now has $J+1$ **extra parameters** (one for each group plus one for the standard deviation across groups) so the budget for the mean needs to be further **divided to model the group means**, whereas the budget for the standard deviation needs to be split **between the within group variation and the between group variation**. But we still have the same amount of data, so we need to be careful to ensure that this model expansion does not destabilize our estimates. This means that as well as putting appropriate priors on our new parameters, we probably need to tighten up the priors on the overall mean and observation standard deviation, lest a lack of information lead to nonsense estimates.

### Funneling in hierarchical (multilevel) models 

> In other cases, models that are well behaved for larger datasets can have computational issues in small data regimes; Figure 13 shows an example. While the funnel-like shape of the posterior in such cases looks similar to the funnel in hierarchical models, this pathology is much harder to avoid, and we can often only acknowledge that the full model is not informed by the data and a simpler model needs to be used. Betancourt (2018) further discusses this issue.

Include the Betancourt as the more substantive discussion here. 

### Calibrating models based on hard to predict features 

> In addition to looking at the calibration of the conditional predictive distributions, we can also look at which observations are hard to predict and see if there is a pattern or explanation for why some are harder to predict than others. This approach can reveal potential problems in the data or data processing, or point to directions for model improvement (Vehtari et al., 2017, Gabry et al., 2019). We illustrate with an analysis of a survey of residents from a small area in Bangladesh that was affected by arsenic in drinking water. Respondents with elevated arsenic levels in their wells were asked if they were interested in switching to water from a neighbor’s well, and a series of models were fit to predict this binary response given household information (Vehtari et al., 2017).



### LOO 

>  Gabry et al. (2019) provide an example where LOO-CV indicated problems that motivated efforts to improve the statistical model.

#### LOO for hierarchical models 

> Cross validation for multilevel (hierarchical) models requires more thought. Leave-one-out is still possible, but it does not always match our inferential goals. For example, when performing multilevel regression for adjusting political surveys, we are often interested in estimating opinion at the state level. A model can show real improvements at the state level with this being undetectable at the level of cross validation of individual observations (Wang and Gelman, 2016). Millar (2018), Merkle, Furr, and Rabe-Hesketh (2019), and Vehtari (2019) demonstrate different cross validation variants and their approximations in hierarchical models, including leave-one-unit-out and leave-one-group-out. In applied problems we have performed a mix, holding out some individual observations and some groups and then evaluating predictions at both levels (Price et al., 1996).
>
> Unfortunately, approximating such cross validation procedures using importance sampling tends to be much harder than in the leave-one-out case. This is because more observations are left out at a time which implies stronger changes in the posterior distributions from the full to the subsetted model. As a result, we may have to rely on more costly model refits to obtain leave-one-unit-out and leave-one-group-out cross validation results. 







### Citations for shrinkage 

> One approach is to compute the shrinkage between prior and posterior, for example, by comparing prior to posterior standard deviations for each parameter or by comparing prior and posterior quantiles. If the data relative to the prior are informative for a particular parameter, shrinkage for that parameter should be stronger. This type of check has been extensively developed in the literature; see, for example, Nott et al. (2020).

#### Using importance sampling to interpolate between posteriors 

> Another approach is to use importance sampling to approximate the posterior of the new model using the posterior of old model, provided the two posteriors are similar enough for importance sampling to bridge (Vehtari et al., 2019, Paananen et al., 2020). And if they are not, this is also valuable information in itself (see Section 6.2).



### Funnels when plotting the 