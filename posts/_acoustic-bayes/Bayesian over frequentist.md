# Bayesian over frequentist 

See also "Motivations for Bayesian..." in Probabilistic Programming for PKPKD 

## Abstract from "Bayesian Workflow" 

> The Bayesian approach to data analysis provides a powerful way to handle uncertainty in all observations, model parameters, and model structure using probability theory. Probabilistic programming languages make it easier to specify and fit Bayesian models, but this still leaves us with many options regarding constructing, evaluating, and using these models, along with many remaining challenges in computation. Using Bayesian inference to solve real-world problems requires not only statistical skills, subject matter knowledge, and programming, but also awareness of the decisions made in the process of data analysis. All of these aspects can be understood as part of a tangled workflow of applied Bayesian statistics. Beyond inference, the workflow also includes iterative model building, model checking, validation and troubleshooting of computational problems, model understanding, and model comparison. We review all these aspects of workflow in the context of several examples, keeping in mind that in practice we will be fitting many models for any given problem, even if only a subset of them will ultimately be relevant for our conclusions.



## Bayesian vs frequentist 

> A primary motivation for Bayesian thinking is that it facilitates a common-sense interpretation of statistical conclusions. For instance, a Bayesian (probability) interval for an unknown quantity of interest can be directly regarded as having a high probability of containing the unknown quantity, in contrast to a frequentist (confidence) interval, which may strictly be interpreted only in relation to a sequence of similar inferences that might be made in repeated practice. Recently in applied statistics, increased emphasis has been placed on interval estimation rather than hypothesis testing, and this provides a strong impetus to the Bayesian viewpoint, since it seems likely that most users of standard confidence intervals give them a common-sense Bayesian interpretation.

From BDA3



## Frequentist get caught in local minima, especially for ODE models which have unidentifiable parameters 

From [Baclofen paper](https://www.biorxiv.org/content/10.1101/2022.10.25.513675v1.full.pdf+html)

> To estimate unknown quantities, optimization methods (within Frequentist approach) are often used in practice by defining an objective (or a cost) function to score the performance of the model by comparing the observed with the predicted values. However, such a parametric approach results in only a point estimation, and the optimization algorithms may easily get stuck in a local maximum, requiring multi-start strategies to address the potential multi-modalities. Moreover, the estimation depends critically on the form of objective function defined for optimization, and the models involving differential equations often have non-identifiable parameters ([Hashemi et al., 2018](https://hal.archives-ouvertes.fr/hal-03602902)).

Citation is to M. Hashemi, A. Hutt, L. Buhry, and J. Sleigh. Optimal Model Parameter Estimation from EEG Power Spectrum Features Observed during General Anesthesia. Neuroinformatics, 16(2):231–251, April 2018. doi: 10.1007/s12021-018-9369-x. URL https://hal.archives-ouvertes.fr/hal-03602902. 





No approximate covariance matrix, computed directly from samples 

Stricter convergence checks 

## Reasons for Bayesian in R / Stan over NONMEM 

1. Flexibility of the specified **hierarchy**
2. Flexibility of **noise model** specification 
3. Flexibility of the **prior** distributions 
4. Greater ability to use **bayesplot** and other **convergence** checking packages 
5. Parameter **exporting** and posterior **predictive** visualization 
6. Sample size optimization? 
7. Allometric scaling 



## Why to use Stan 

From the Baclofen paper 

>  To conduct Bayesian data analysis, Markov chain Monte Carlo (MCMC) methods have often been used to sample from and hence, approximate the exact posterior distributions. However, MCMC sampling in high-dimensional parameter spaces, which converge to the desired target distribution, is non-trivial and computationally expensive (Betancourt et al., 2014; Betancourt, 2017). In particular, the use of differential equations (such as PK population models) together with noise in data raise many convergence issues (Hashemi et al., 2020; Grinsztajn et al., 2021; Jha et al., 2022). Designing an efficient MCMC sampler to perform principled Bayesian inference on high-dimensional and correlated parameters remains a challenging task. Although the Bayesian inference requires painstaking model-specific derivations and hyper-parameter tuning, probabilistic programming languages such as Stan (Carpenter et al., 2017) provide high-level tools to reliably solve complex parameter estimation problems. Stan (see https://mc-stan.org) is a state-of-the-art platform for high-performance statistical computation and automatic Bayesian data analysis, which provides advanced algorithms (Hoffman and Gelman, 2014), efficient gradient computation (Margossian, 2018), and is enriched with numerous diagnostics to check whether the inference is reliable (Vehtari et al., 2021).



### Sensibility of Bayesian inference - Baclofen 

> Bayesian inference is a principled method to estimate the posterior distribution of unknown quantities given only observed responses and prior beliefs about unobserved hidden states (or latent variables). An advantage of using the Bayesian framework in the context of inference/prediction is the ability to generate not only a single point estimate (e.g., in the Frequentist approach), but also full probability distributions for the quantities of interest (uncertainty quantification for decision-making process). From the latter, one can directly extract quantiles, with the possibility to answer questions such as "what is the probability that the parameter of interest is greater/smaller than a specific value?", with the confidence intervals in estimation. In addition, the **propagation of uncertainty** in the Bayesian framework provides a more robust and reliable predictive capability of the model under study, rather than point estimation with optimization methods. In particular, the out-of-sample prediction accuracy (i.e., the measure of the model’s ability in new data prediction e.g., using WAIC) enables reliable and efficient evaluation of potential hypotheses, as performed in this study. Several previous studies have used a scoring function (such as root mean square error or correlation) to measure the similarity between empirical and fitted data  (Imbert et al., 2015). The choice of scoring function can dramatically affect the ranking of model candidates, and ultimately the decision-making processes (see RMSE in Table 1). Rather, we used non-parametric probabilistic methodology to analyze data, while various convergence diagnostics were monitored to assess when the sampling procedure has converged to sampling from the target distribution.



## Hierarchical distributions 

We can build in uncertainty and iteratively augment our model 



## Diagnostics and parameter uncertainty 

From Baclofen, WAIC citation [Gelman 2013](https://link.springer.com/article/10.1007/s11222-013-9416-2) 

> The Bayesian approach applied to PK analysis provides a fully probabilistic description of unknown quantities that allows not only a straightforward interpretation of the inferred parameters and outcomes, but also the modeling of uncertainty about the inferred values of these quantities. Moreover, it provides us with a principled method for model comparison, selection, and decision-making by measuring the out-of-sample model predictive accuracy (i.e., the measure of the model’s ability in new data prediction). To assess the predictive accuracy, we used Watanabe-Akaike (or widely available) information criterion (WAIC; Gelman et al. (2014b)). This metric is a fully Bayesian information criterion that uses the whole posterior distribution, thus enabling us to integrate our prior knowledge in the model selection. Following Gelman et al. (2014b), WAIC is given by:
>
> WAIC = −2(lppd − peff),
>
> where lppd is the log pointwise predictive density for a new data point (as the accuracy term 3 ), and peff is the effective number of parameters (as penalty term to adjust for overfitting) 4 . In practice, we can replace the expectations with the average over the draws from the full posterior to calculate WAIC (for more details see Gelman et al. (2014b)). Note that WAIC uses the whole posterior distribution rather than the point estimation used in the classical information criteria such as AIC and BIC. Finally, the relative difference in WAIC is used to measure the level or the strength of evidence for each candidate model under consideration. The lower value of WAIC indicates a better model fit. Following Burnham and Anderson (2002), a relative difference larger than 10 between the best model (with minimum WAIC) and other candidate models indicates that an alternative model is very unlikely (i.e., an alternative model has essentially no support). 



## Treatment effects 

Remember the [rule of 16](https://statmodeling.stat.columbia.edu/2018/03/15/need-16-times-sample-size-estimate-interaction-estimate-main-effect/), which is the thought that it takes ~16 times the amount of data to estimate a 