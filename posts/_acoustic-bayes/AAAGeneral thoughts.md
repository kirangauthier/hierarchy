# General thoughts 

Packages to use: patchwork, gt (tables), flextable, here (working directory management), ggthemes, RColorBrewer par(cex = 0.5); display.brewer.all(), 

you can prompt engineer for chatGPT by using all caps and please, see the leaked DALL-E training 

HOCT - hematology, oncology & cell therapy 

ICFN - immunology, cardiovascular, fibrosis, neurology 

Remember that mean estimates are from the heart of the distribution and rare events are from the tails of the distribution 


When QC'ing, first open the `.Rproj` file which properly sets up your `.libPaths()` and then run `renv::init(bare = TRUE)` after which you can immediately load your `library(...)` if the code has been `cp -r` from a previously finalized directory 

Bayesian inference is all about the generative data model, for each possible explanation (proportion of water on the globe) of the sample, count all the ways the sample could happen, explanations with more ways to produce the sample are more plausible 

Nothing magical happens at the boundary of 95% for confidence intervals, does not matter if 95% CI contains 0, it's totally arbitrary, estimates are not "robust", this is not a property of the interval, it's a property of the estimator, only thing NHST is doing is controlling type I error, the fact that an arbitrary interval contains an arbitrary value is not meaningful, we use the whole distribution, posterior distributions don't do Type I error control anyways 

remember the curve function 

```R
post_samples <- rbeta( n=1e3, 6+1, 3+1 ) 
dens( post_samples, lwd=4, col=2, xlab = "proportion water", adj=0.1 ) 
curve( dbeta(x, 6+1, 3+1), add = TRUE, lty=2, lwd=3 )
```

if you're feeling confused, it's because you're paying attention 



Every good idea Andrew's already had from some subsection of one of his papers from the 90s is already published by psychometricians 

Maximized squared jump distance is best transitions jumping rule 

You don't want to jump out of the scale of your principal component axis or you leave your typical set 

Optimal acceptance rate of I think MH on a Gaussian posterior is like 0.234, surprisingly low 

Get into the habit of failing fast 



From the Charles Margossian ODE talk 

> moved from astronomy / physics background at Yale to Metrum, was originally very reductionist, the fundamental particles with the fundamental interactions, but still grapple with uncertainty in looking for exoplanets, how do you prove that something is there 
>
> in PMx, emergent behaviour, we know that the model is wrong and we also have noisy, variable data, both lead to Bayesian interpretations of the model 

There is always a balance between accuracy and convenience. As discussed in Chapter 6, predictive model checks can reveal serious model misfit, but we do not yet have good general principles to justify our basic model choices. As computation of hierarchical models becomes more routine, we may begin to use more elaborate models as defaults.



If the variability of parameters does not decrease with increasing data, this is evidence to suggest that parameters are poorly identified, from [here](https://arxiv.org/pdf/2210.16224.pdf).

> Overall, most parameters are poorly identified as evidenced by their stable (rather than decreasing) variability with more data.

From [here](https://discourse.mc-stan.org/t/trying-to-make-a-three-level-nested-linear-model-run-faster/3405/3).

![image-20230810123553442](C:/Users/dhattgak/OneDrive%20-%20Bristol%20Myers%20Squibb/Documents/BMS%20Projects/assets/image-20230810123553442-1691685354278-1.png)

Julia compiles every function, Python can jit but not everything jits at the same time 

Does explicit elementwise operations, needs .+ for scalar and vector so a lot less errors, enforces scope 

Can also use threads in Julia, Python has something weird where you can't finish until the last thread, and also memory footprint is a lot better, because you get to use the same global variables and all that 

Also specifies maximum compatible version so things don't just randomly break 



You can also try and trick the model by permuting the data, and giving it to the model in a nonsensical order, also from [here](https://arxiv.org/pdf/2210.16224.pdf). 

> A second way to assess the predictive ability (and economic content) of the SW DSGE model is to
> perform a simple permutation test, permuting across the series rather than within them. That is,
> rather than giving the model data in the order it expects, we swap the data series with each other
> and see if the model predicts future data any better. We estimated the model on the properly
> ordered data (presented in the previous section) as well as all 5039 other permutations of the 7
> data series. For each estimation, we used the same estimation procedure as before to minimize the
> (penalized) negative log likelihood. The model is trained using the first 200 time points and its
> predictive performance is tested on the remaining 51.
>
> ![image-20230530120150175](C:/Users/dhattgak/OneDrive%20-%20Bristol%20Myers%20Squibb/Documents/BMS%20Projects/assets/image-20230530120150175.png)

![image-20230524141553627](C:/Users/dhattgak/OneDrive%20-%20Bristol%20Myers%20Squibb/Documents/BMS%20Projects/assets/image-20230524141553627.png)



For example, a regression with many predictors and a flat prior distribution on the coefficients will
tend to overestimate the variation among the coefficients, just as the independent estimates
for the eight schools were more spread than appropriate

The biggest strength of Bayesian is **hierarchical**, and you can explicitly test the "strength" of your priors -> BDA3 

From [here](https://cran.r-project.org/web/packages/brms/vignettes/brms_multilevel.pdf). 

> Usually, however, the functionality of these implementations is limited insofar as it is only
> possible to predict the mean of the response distribution. Other parameters of the response
> distribution, such as the residual standard deviation in linear models, are assumed constant
> across observations, which may be violated in many applications. Accordingly, it is desirable
> to allow for prediction of all response parameters at the same time. Models doing exactly that
> are often referred to as distributional models or more verbosely models for location, scale and
> shape (Rigby and Stasinopoulos 2005). Another limitation of basic MLMs is that they only
> allow for linear predictor terms. While linear predictor terms already offer good flexibility,
> they are of limited use when relationships are inherently non-linear. Such non-linearity can
> be handled in at least two ways: (1) by fully specifying a non-linear predictor term with
> corresponding parameters each of which can be predicted using MLMs (Lindstrom and Bates
> 1990), or (2) estimating the form of the non-linear relationship on the fly using splines (Wood or Gaussian processes (Rasmussen and Williams 2006). **The former are often simply called non-linear models, while models applying splines are referred to as generalized additive**
> **models (GAMs; Hastie and Tibshirani, 1990).**

The all-important point that that skeptics and subjectivists alike [strain on the gnat](https://statmodeling.stat.columbia.edu/2015/01/27/perhaps-merely-accident-history-skeptics-subjectivists-alike-strain-gnat-prior-distribution-swallowing-camel-likelihood/) of the prior distribution while swallowing the camel that is the likelihood.

Bayesian is really good for extrapolating from the means to the extremes -> see Nicolas Simon ["Vancomycin Pharmacokinetics Throughout Life: Results from a Pooled Population Analysis and Evaluation of Current Dosing Recommendations"](https://link.springer.com/article/10.1007/s40262-018-0727-5)

## Blurring in plots to demonstrate uncertainty 

> [Matthew Kay](https://mjskay.com/) on [April 13, 2023 10:03 AM at 10:03 am](https://statmodeling.stat.columbia.edu/2023/04/13/the-percentogram-a-histogram-binned-by-percentages-of-the-cumulative-distribution-rather-than-using-fixed-bin-widths/#comment-2201647) said: 									
>
> I played around with something  similar when trying to create representations of posteriors that also  reflect MCSEs of quantiles. Basic idea was to create bins that keep  quantiles together if they are within 2 MCSEs of each other. See the  bottom of this page: https://github.com/mjskay/uncertainty-examples/blob/master/mcse_dotplots.md
>
> The first half of that page is a very different approach – to use  dotplots of quantiles, but (essentially) blur the dots according to  MCSE. The idea is to try to make it obvious you shouldn’t rely on some  of the quantiles of a posterior depending on the MCSE for that  particular quantile.
>
> ​					[Reply ↓](https://statmodeling.stat.columbia.edu/2023/04/13/the-percentogram-a-histogram-binned-by-percentages-of-the-cumulative-distribution-rather-than-using-fixed-bin-widths/?replytocom=2201647#respond)				
>
> - ​	[img](https://secure.gravatar.com/avatar/3ef672113a958ffd1680367b00a7139e?s=39&d=identicon&r=g)Patrick on [April 13, 2023 3:12 PM at 3:12 pm](https://statmodeling.stat.columbia.edu/2023/04/13/the-percentogram-a-histogram-binned-by-percentages-of-the-cumulative-distribution-rather-than-using-fixed-bin-widths/#comment-2201744) said: 									
>
>   This is awesome, I once had an idea  for showing uncertainty in maps using a blur concept. Basically have  your statistics in whatever color gradient you like but then have this  additional layer of “blur” on top of that, idea being that your eye  would look toward the unblurry statistics and ignore the really  uncertain areas. Cool to see blurry things in ggplot2. I should probably give it a shot again.
>
>   ​					[Reply ↓](https://statmodeling.stat.columbia.edu/2023/04/13/the-percentogram-a-histogram-binned-by-percentages-of-the-cumulative-distribution-rather-than-using-fixed-bin-widths/?replytocom=2201744#respond)				

![mcse_dotplot-1](C:\Users\dhattgak\AppData\Local\Temp\mcse_dotplot-1.png)



## Horseshoe (spike-and-slab) priors as regularization tools 

I think horseshoes are interesting in concept to regularize your parameters and are especially useful when the number of parameters is much larger than the number of observations, spike-and-slab priors are the extension to low dimensional datasets 

> The Horseshoe prior can be applied to models with a large number of  parameters, and is particularly useful when there are many weakly  relevant predictors, but only a few strongly relevant predictors. It has been shown to perform well in high-dimensional settings where the  number of predictors is much larger than the number of observations. In  general, the number of parameters in the model should be larger than the number of observations, and the more parameters in the model, the more  the Horseshoe prior can help to improve performance. 

 ![31a198cad07bc66f1ae9b21897ab135f860713e9](C:\Users\dhattgak\AppData\Local\Temp\31a198cad07bc66f1ae9b21897ab135f860713e9.png)



### Bayesian vs frequentist 

> A primary motivation for Bayesian thinking is that it facilitates a common-sense interpretation of statistical conclusions. For instance, a Bayesian (probability) interval for an unknown quantity of interest can be directly regarded as having a high probability of containing the unknown quantity, in contrast to a frequentist (confidence) interval, which may strictly be interpreted only in relation to a sequence of similar inferences that might be made in repeated practice. Recently in applied statistics, increased emphasis has been placed on interval estimation rather than hypothesis testing, and this provides a strong impetus to the Bayesian viewpoint, since it seems likely that most users of standard confidence intervals give them a common-sense Bayesian interpretation.

From BDA3

### Multilevel modeling is a formalization of empirical Bayes estimation 

From "Bayesian Workflow" 

> Multilevel modeling is a formalization of what has been called empirical Bayes estimation of prior distributions, expanding the model so as to fold inference about priors into a fully Bayesian framework. Exploratory data analysis can be understood as a form of predictive model checking (Gelman, 2003). Regularization methods such as lasso (Tibshirani, 1996) and horseshoe (Piironen et al., 2020) have replaced ad hoc variable selection tools in regression. Nonparametric models such as Gaussian processes (O’Hagan, 1978, Rasumussen and Williams, 2006) can be thought of as Bayesian replacements for procedures such as kernel smoothing. 



![image-20230330075749609](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230330075749609.png)

notation for multivariate normal distributions 



#### More computationally expensive, stiff solvers are sometimes required for the burn-in stage of PMx ODEs 

> In our experience, notably with differential equation based models in pharmacology and epidemiology, we sometime require a more computationally expensive stiff solver to tackle difficult ODEs generated during the warmup phase. 



### VI is a generalization of the EM algorithm 

From "Bayesian Workflow" 

> Variational inference is a generalization fo the expectation-maximization (EM) algorithm, and can, in the Bayesian context provide a fast but possibly inaccurate approximation fo the posterior distribution 



### SMC is a generalization of Metropolis and HMC is a different generalization of Metropolis that uses gradients to move efficiently through probability space 

> Sequential Monte Carlo is a generalization of the Metropolis algorithm that can be applied to any Bayesian computation, and HMC is a different generalization of Metropolis that uses gradient computation to move efficiently through continuous probability spaces.



### Adage / folk theorem of statistical computing 

> Recall that the chains that yielded the lowest log posteriors were also the ones that were the slowest—an instance of the folk theorem of statistical computing (see Section 5.1) There is wisdom is this anecdote: an easy deterministic problem can become difficult in a Bayesian analysis. Indeed Bayesian inference requires us to solve the problem across a range of parameter values, which means we must sometimes confront unsuspected versions of the said problem. In our experience, notably with differential equation based models in pharmacology and epidemiology, we sometime require a more computationally expensive stiff solver to tackle difficult ODEs generated during the warmup phase. Other times slow computation can alert us that our inference is allowing for absurd parameter values and that we need either better priors or more reasonable initial points.



### Reference for Pfizer FFP "right answer the wrong way" 

From "Bayesian Workflow" 

> If our goal is not merely prediction but estimating the latent variables, examining predictions only helps us so much. This is especially true of overparameterized models, where wildly different parameter values can yield comparable predictions (e.g. Section 5.9 and Gelman et al, 1996). 



"relaxation time" is the time required to generate one independent sample, and is a nice estimate of the cost of s

### Tough to compute models are usually misspecified models 

> **5.1. The folk theorem of statistical computing** 
>
> When you have computational problems, often there’s a problem with your model (Yao, Vehtari, and Gelman, 2020). Not always—sometimes you will have a model that is legitimately difficult to fit—but many cases of poor convergence correspond to regions of parameter space that are not of substantive interest or even to a nonsensical model. An example of pathologies in irrelevant regions of parameter space is given in Figure 6. Examples of fundamentally problematic models would be bugs in code or using a Gaussian-distributed varying intercept for each individual observation in a Gaussian or logistic regression context, where they cannot be informed by data. Our first instinct when faced with a problematic model should not be to throw more computational resources on the model (e.g., by running the sampler for more iterations or reducing the step size of the HMC algorithm), but to check whether our model contains some substantive pathology.

### Adjusting for lots of predictors using least squares leads to overfitting - overadjusting for noise / times when models are the same under LS but different under Bayesian 

See ["A bit of harmful advice from “Mostly Harmless Econometrics”"](https://statmodeling.stat.columbia.edu/2023/03/02/a-bit-of-harmful-advice-from-mostly-harmless-economics/) and section 5 of Gelman's cited paper ["Parameterization and Bayesian Modeling"](http://www.stat.columbia.edu/~gelman/research/published/parameterization.pdf) to get a sense of where two models are the same under MLE but different under Bayesian 

> absentminded on [March 2, 2023 10:24 PM at 10:24 pm](https://statmodeling.stat.columbia.edu/2023/03/02/a-bit-of-harmful-advice-from-mostly-harmless-economics/#comment-2183051) said:
>
> Including group-level means of your X in the regression is algebraically the same as running fixed effects (i.e., including a dummy variable for each group). Not only do the authors know about it, it’s precisely what they’re recommending!
>
> - [Andrew](http://www.stat.columbia.edu/~gelman/) on [March 3, 2023 2:50 AM at 2:50 am](https://statmodeling.stat.columbia.edu/2023/03/02/a-bit-of-harmful-advice-from-mostly-harmless-economics/#comment-2183122) said:
>
>   - Absentminded:
>
>     Yes if you’re fitting the model using least squares, no if you’re fitting a hierarchical model or more generally if you have priors on the coefficients. See section 5 of [my 2004 paper on parameterization](http://www.stat.columbia.edu/~gelman/research/published/parameterization.pdf) for some discussion of how you can have two models that are equivalent to each other if you’re fitting least squares or maximum likelihood, but become different when you consider Bayesian models.
>
>     One reason this is important in practice is that if you adjust for lots of predictors using least squares, you will overfit—overadjust for noise.



### $p(\text{aliens on Neptune can rap battle})$ 

from [here](https://statmodeling.stat.columbia.edu/2022/12/23/a-probability-isnt-just-a-number-its-part-of-a-network-of-conditional-statements/), 

> This came up awhile ago in comments, when Justin [asked](https://statmodeling.stat.columbia.edu/2018/12/26/what-is-probability/#comment-934279):
>
> > Is p(aliens exist on Neptune that can rap battle) = .137 valid “probability” just because it satisfies mathematical axioms?
>
> And Martha sagely [replied](https://statmodeling.stat.columbia.edu/2018/12/26/what-is-probability/#comment-934916):
>
> > “p(aliens exist on Neptune that can rap battle) = .137”  in itself isn’t something that can satisfy the axioms of probability.  The axioms of probability refer to a “system” of probabilities that are  “coherent” in the sense of satisfying the axioms. So, for example, the  two statements
> >
> > “p(aliens exist on Neptune that can rap battle) = .137″ and p(aliens exist on Neptune) = .001”
> >
> > are incompatible according to the axioms of probability, because the  event “aliens exist on Neptune that can rap battle” is a sub-event of  “aliens exist on Neptune”, so the larger event must (as a consequence of the axioms) **have probability at least as large as the probability of  the smaller event.**
>
> The general point is that a probability can only be understood as  part of a larger joint distribution; see the second-to-last paragraph of [the boxer/wrestler article](http://www.stat.columbia.edu/~gelman/research/published/augie4.pdf).  I think that confusion on this point has led to lots of general confusion about probability and its applications.



### If you can't simulate from your model, you don't really know it 

From [(What’s So Funny ‘Bout) Fake Data Simulation](https://statmodeling.stat.columbia.edu/2023/02/28/whats-so-funny-bout-fake-data-simulation/#comments)

> I don’t have the energy right now to follow all the details so let me make a generic recommendation which is to set up a reasonable scenario representing “reality” as it might be, then simulate fake data from this scenario, then fit your model to the fake data to see if it does a good job at recovering the assumed truth. This approach of fake-data experimentation has three virtues:
>
> 1. If the result doesn’t line up with the assumed parameter values, this tells you that something’s wrong, and you can do further experimentation to see the source of the problem, which might be a bug in your fitting code, a bug in your simulation code, a conceptual error in your model, or a lack of identifiability in your model.
> 2. If the result does confirm with the assumed parameter values, then it’s time to do some experimentation to figure out when this doesn’t happen. Or maybe your original take, that the inferences didn’t make sense, was itself mistaken.
> 3. In any case, the effort required to simulate the fake data won’t be wasted, because doing the constructive work to build a generative model should help your understanding. If you can’t simulate from your model, you don’t really know it. Ideally the simulation should be of raw data, and then all steps of normalization etc. would come after.





mathematical modeling -> learn about data generating process / generative models make predictions on outcomes (unseen data). and to justify hypotheses 

Although PK modeling has facilitated the drug development process, the accurate and reliable estimation of its parameters from noisy data is a major challenge for conducting clinical research to determine the safety and efficacy of a drug within a particular disease or specific patient population. To estimate unknown quantities, optimization methods (within Frequentist approach) are often used in practice by defining an objective (or a cost) function to score the performance of the model by comparing the observed with the predicted values. However, such a parametric approach results in only a point estimation, and the optimization algorithms may easily get stuck in a local maximum, requiring multi-start strategies to address the potential multi-modalities. Moreover, the estimation depends critically on the form of objective function defined for optimization, and the models involving differential equations often have non-identifiable parameters (Hashemi et al., 2018).

In this study, we use the Bayesian approach to address these challenges in the estimation of the PK population model parameters from synthetic data (generated by known values for validation), and then routine clinical data that were retrospectively collected from 67 adult outpatients treated with oral Baclofen. The Bayesian framework is a principled method for inference and prediction with a broad range of applications, while the uncertainty in parameter estimation is naturally quantified through probability distribution placed on the parameters updated with the information provided by data (Gelman et al., 2014a; Bishop, 2006). Such a probabilistic technique provides the full posterior distribution of unknown quantities in the underlying data generating process given only observed responses and the existing information about uncertain quantities expressed as prior probability distribution (Ferreira et al., 2020; Hashemi et al., 2021). In other words, Bayesian inference provides all plausible parameter ranges consistent with observation by integrating the information from both domain expertise and the experimental data. In the context of clinical trials, using Frequentist approach, prior information (based on evidence from previous trials) is utilized only in the design of a trial but not in the analysis of the data (Jack Lee and Chu, 2012; Gupta, 2012). On the other hand, Bayesian approach provides a formal mathematical framework to combine prior information with available information at the design stage, during the conduct of the experiments, and at the data analysis stage (Spiegelhalter et al., 1999; Berry, 2006; Yarnell et al., 2021). To conduct Bayesian data analysis, Markov chain Monte Carlo (MCMC) methods have often been used to sample from and hence, approximate the exact posterior distributions. However, MCMC sampling in high-dimensional parameter spaces, which converge to the desired target distribution, is non-trivial and computationally expensive (Betancourt et al., 2014; Betancourt, 2017). In particular, the use of differential equations (such as PK population models) together with noise in data raise many convergence issues (Hashemi et al., 2020; Grinsztajn et al., 2021; Jha et al., 2022). Designing an efficient MCMC sampler to perform principled Bayesian inference on high-dimensional and correlated parameters remains a challenging task. Although the Bayesian inference requires painstaking model-specific derivations and hyper-parameter tuning, probabilistic programming languages such as Stan (Carpenter et al., 2017) provide high-level tools to reliably solve complex parameter estimation problems. Stan (see https://mc-stan.org) is a state-of-the-art platform for high-performance statistical computation and automatic Bayesian data analysis, which provides advanced algorithms (Hoffman and Gelman, 2014), efficient gradient computation (Margossian, 2018), and is enriched with numerous diagnostics to check whether the inference is reliable (Vehtari et al., 2021). In the present work, to estimate the posterior distribution of PK model parameters with MCMC, we use a self-tuning variant of Hamiltonian Monte Carlo (HMC) in Stan. This algorithm adaptively tunes the HMC’s parameters, making the sampling strategy more efficient and automatic (Hoffman and Gelman, 2014). The principled Bayesian setting (Gelman et al., 2020) in this study-validated on synthetic data- enables us to efficiently and accurately estimate the Baclofen effect on patients with AUD. This work may pave the way to reliably predict the treatment drug efficacy from longitudinal patient data, optimizing strategies for clinical decision, especially in brain disorders

[Why Bayesian?](https://www.bayes-pharma.org/wp-content/uploads/2014/10/BAYES2015_TW.pdf) - also saved in `/BMS Projects/Software/BauerNONMEMTutorials` 

![image-20230213144529636](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213144529636.png)



Why population approach as opposed to analyzing individual data points? [Fundamentals of Pharmacometrics](https://www.youtube.com/watch?v=fV649fkfhGA) 

1. Most data is not rich 
2. Moving to exposure-response from dose-response will probably break the multimodality / correlation of the $\text{E}_{\text{max}}$ model parameters 
3. A lot during absorption, elimination, distribution (drug is going from central compartment to tissue and back to plasma compartment), then I could get parameters, but unlikely on Phase II, Phase III especially 
4. Data is also *sparse*, motivates **population** approach 
5. Distributions can be lognormally distributed or part of a mixture model 
6. Figure out the distinction between probabilities and confidence intervals, and the sampled results of MCMC and the imposed continuous distributions we assign to them, inspired by [this Gelman blog post](https://statmodeling.stat.columbia.edu/2022/10/22/statistical-methods-that-only-work-if-you-dont-use-them/) 
   1. ??? this needs to be answered *** 
7. *All models are wrong, but some of them are useful*, George Box 
   1. often simple models are adequate and useful 



[Sensitivity analysis of a model](https://www.bayes-pharma.org/wp-content/uploads/2014/10/BAYES2015_TW.pdf) - also saved in `/BMS Projects/Software/BauerNONMEMTutorials` 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213144417483.png" alt="image-20230213144417483" style="zoom:50%;" />

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213144506309.png" alt="image-20230213144506309" style="zoom:50%;" />



[George Box](https://en.wikipedia.org/wiki/All_models_are_wrong)

**2.3 Parsimony**
Since all models are wrong the scientist cannot obtain a "correct" one by excessive elaboration. On the contrary following [William of Occam](https://en.wikipedia.org/wiki/Occam's_Razor) he should seek an economical description of natural phenomena. Just as the ability to devise simple but evocative models is the signature of the great scientist so overelaboration and overparameterization is often the mark of mediocrity.**
**2.4 Worrying Selectively**
Since all models are wrong the scientist must be alert to what is importantly wrong. It is inappropriate to be concerned about mice when there are tigers abroad.




**Model criticism** should be completed by simulating certain key statistics from the sampled parameters and then comparing that to the summary statistics observed from the true data while accounting for multiple comparisons 

​			Apparently Gelman in Bayesian Data Analysis (2013) he suggested **not** to adjust for multiple comparisons when using test statistics 



The opportunity with nonlinear models is to see a non-monotonic increase in the information content as a function of experiment number, validates the use of hierarchical, nonlinear models 



Always estimate parameters in **logspace**, can convert from log-transformed variables to real space using [this](https://stats.stackexchange.com/questions/533804/covariance-of-log-transformed-variable) (code is below for the optimization) 

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221216104238938.png" alt="image-20221216104238938" style="zoom:33%;" />

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221216104250950.png" alt="image-20221216104250950" style="zoom:33%;" />





1. Remember that data quality is paramount, the quantity of data comes second in many settings -> I believe that Bayesian algorithms are most useful when 



### Optimization methods 

Dimitri says that ADVI has very limited uses, really cool to turn a probabilistic inference problem into a optimization problem but because it's so difficult to assess convergence, and low probability regions can be "filled in" by the specific form of the univariate proposal distribution that we prescribe 



### Synthesizing data 

See [here](https://www.biorxiv.org/content/10.1101/2022.10.25.513675v1.full.pdf)

> Using synthetic data for fitting allows us to validate the inference process as we know the ground-truth of the parameters being estimated. Therefore, we can use standard error metrics to measure the similarity between the inferred parameters and those used for data generation. Synthetic data were generated following the same event schedule as empirical clinical data and using a one-compartment population model with first-order absorption (see Eq. (1)). The synthetic data was generated using the R package mrgsolve (Baron, 2022), which enables simulation from ODE-based hierarchical models, including random effects and covariates. The mrgsolve package (Elmokadem et al., 2019) uses Livermore Solver for Ordinary Differential Equations (LSODE; LSO) of the Fortran ODEPACK library (Hindmarsh, 1992) integrated in R through the Rcpp (Eddelbuettel and Francois, 2011) package.

https://github.com/metrumresearchgroup/mrgsolve 

### Bayesian modeling for pharmacometricians 

See [here](https://www.biorxiv.org/content/10.1101/2022.10.25.513675v1.full.pdf) 

> In Bayesian modeling, all model parameters are treated as random variables and the values vary based on the underlying probability distribution (Bishop, 2006). That is, in the case of a one-compartment PK model, kinetic parameters CL, V and ka will be interpreted as random variables, and we aim to infer their probability distributions based on prior knowledge (e.g., derived from physiological information or previous evidence) updated with available information in the observed data through the so-called likelihood function, i.e. the probability of some observed outcomes given a set of parameters. Although the likelihood function can provide the best-fit points (maximum likelihood estimators), we are interested in the whole posterior distribution over parameters as it contains all relevant information about parameters after observing data to perform inference and prediction. 
>
> cont... 

### Predictive accuracy of a Bayesian model 

WAIC uses the whole posterior as opposed to AIC and BIC which just use pointwise, see [here](https://www.biorxiv.org/content/10.1101/2022.10.25.513675v1.full.pdf) 

![image-20230223165944977](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230223165944977.png)

#### MBMA block 

Things to consider when doing MBMA 

> *[Scientists often use](https://arxiv.org/abs/2302.04739) meta-analysis to characterize the impact of an intervention on some outcome of interest across a body of literature. However, threats to the utility and validity of meta-analytic estimates arise when scientists average over potentially important variations in context like different research designs. Uncertainty about quality and commensurability of evidence casts doubt on results from meta-analysis, yet existing software tools for meta-analysis do not necessarily emphasize addressing these concerns in their workflows. We present MetaExplorer, a prototype system for meta-analysis that we developed using iterative design with meta-analysis experts to provide a guided process for eliciting assessments of uncertainty and reasoning about how to incorporate them during statistical inference. Our qualitative evaluation of MetaExplorer with experienced meta-analysts shows that imposing a structured workflow both elevates the perceived importance of epistemic concerns and presents opportunities for tools to engage users in dialogue around goals and standards for evidence aggregation.*  
>
> One way to think about good interface design is that we want to reduce sources of the “friction” like the cognitive effort users have to exert when they go to do some task; in other words minimize the so-called gulf of execution. But then there are tasks like meta-analysis where being on auto-pilot can result in misleading results. We don’t necessarily want to create tools that encourage certain mindsets, like when users get overzealous about suppressing sources of heterogeneity across studies in order to get some average that they can interpret as the ‘true’ fixed effect. So what do you do instead? One option is to create a tool that undermines the analyst’s attempts to combine disparate sources of evidence every chance it gets. 
>
> This is essentially the philosophy behind MetaExplorer. This project started when I was approached by an AI firm pursuing a contract with the Navy, where systematic review and meta-analysis are used to make recommendations to higher-ups about training protocols or other interventions that could be adopted. Five years later, a project that I had naively figured would take a year (this was my first time collaborating with a government agency) culminated in a tool that differs from other software out there primarily in its heavy emphasis on sources of heterogeneity and uncertainty. It guides the user through making their goals explicit, like what the target context they care about is; extracting effect estimates and supporting information from a set of studies; identifying characteristics of the studied populations and analysis approaches; and noting concerns about asymmetries, flaws in analysis, or mismatch between the studied and target context. These sources of epistemic uncertainty get propagated to a forest plot view where the analyst can see how an estimate varies as studies are regrouped or omitted. It’s limited to small meta-analyses of controlled experiments, and we have various ideas based on our interviews of meta-analysts that could improve its value for training and collaboration. But maybe some of the ideas will be useful either to those doing meta-analysis or building software. Codebase is [here](https://github.com/sarahyh/ckm-opencpu).

From that paper, "hierarchical models are the ''gold standard'' ", see M.W. Lipsey and D.B. Wilson. 2001. Practical Meta-Analysis. SAGE Publications (2001 book with 12+k citations). 



I love the idea of using "canary variables", which are explored by Gelman for survey models [here](https://statmodeling.stat.columbia.edu/2023/02/07/checking-survey-representativeness-by-looking-at-canary-variables/) and explained by OpenGPT below, 

> Gelman's "canary variables" are a method of identifying potential issues with a non-survey model. Canary variables are variables not used in the model that are important to the context of the problem. The values of these variables are then monitored to assess whether the model is capturing the relevant aspects of the data. **For example, if the model is predicting sales in a particular region, a canary variable may be the average temperature in the region**. If the model predicts sales will increase when the temperature drops, this could be an indication that the model is not capturing the relevant aspects of the data. Therefore, the canary variable should be monitored to ensure that the model is capturing the relevant aspects of the data.



Project Optimus guidelines **read this and summarize**, FDA wants to use **totality of the data** 

![image-20230208091713835](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230208091713835.png)



Dose optimization case study 

![image-20230208093902736](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230208093902736.png)

![image-20230213155746403](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213155746403.png)

[Stan vs NONMEM for ODE models](https://www.bayes-pharma.org/wp-content/uploads/2014/10/BayesianPmetricsBAYES2015.pdf) 



![image-20230213155938541](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213155938541.png)

pros and cons of NONMEM 

![image-20230213160010313](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230213160010313.png)

pros and cons of Stan 

## Techniques to remember 

### 1. Simulation based calibration 

Not totally sure how it works but I think

### Reasons for / against AI and ML vs mechanistic modeling 

Prediction vs. inference [Thomas Wiecki](https://twiecki.io/blog/2021/02/23/intro_pymc_labs/) 

> As data science has exploded in the last decade, I have always been surprised by the **over-emphasis on prediction-focused machine learning**. For far too long, it has been hailed as the solution to most of our data science problems. 
>
> I believe that the potential of this is way overblown. Not because it doesn't work -- algorithms like deep nets or random forests are extremely powerful at extracting non-linear predictive patterns from large data sets -- but rather because most data science problems are not simple **prediction** but rather **inference** problems. 
>
> In addition, we often already have a lot of knowledge about our problem: knowledge of certain structure in our data set (like nested data, that some variables relate to some but not other parameters) and knowledge of which range of values we expect certain parameters of our model to fall into. Prediction-focused ML does not allow us to include any of this information, that's why it requires so much data. 
>
> With Bayesian statistics, we don't have to learn everything from data as we translate this knowledge into a custom model. Thus, rather than changing our problem to fit the solution, as is common with ML, we can tailor the solution to best solve the problem at hand. 
>
> ... 
>
> Playmobil (ML) v. Lego (Bayes) allows us to build what we actually want using our building block which are probability distributions. 

- Pros 
  - Interpretability 
  - Data-efficiency 
  - Robustness 
  - watch https://youtu.be/S_vefzluy4o 

- Way better than summary statistics 
  - [Wiecki computational psychiatry dissertation](https://twiecki.io/blog/2019/03/15/computational_psychiatry_thesis/) 
- JAX, Pyro, NumPyro (more powerful, less documentation) 

#### Arguments against 

1. Doesn't work on large(ish) datasets 
   1. Slow 
   2. Funneling of hierarchical models 
2. There is no way to recover from a mis-specified model 

---



## Emax modeling 

See [Fitting the Emax model in R (backup copy in Mendeley)](https://www.kristianbrock.com/post/emax-intro/) 

- Logit and probit are terrible fits as they assume that they assume **the event probability tends to 1 as linear predictor tends to $\infty$** 
  - Implies that an event probability of 1.0 is **guaranteed**, given a high enough exposure 



#### Emax model 

$ R_i = E_0 + \frac{D_i^N E_{max}}{D_i^N + ED_{50}^N}$  where 

$R_i$ is the response for experimental unit $i$; 

$D_i$ is the exposure (or dose) of experimental unit $i$; 

$E_0$ is the expected response when exposure is zero, also known as the zero-dose effect, or the *basal* effect; 

$E_{max}$ is the maximum effect attributable to exposure; 

$ED_{50}$ is the exposure that produces half of $E_{max}$; 

$N > 0$ is the slope factor, determining the steepness of the dose-response curve; 

The $E_{max}$ [model](https://www.youtube.com/watch?v=E713BehI2fE) is the 4 parameter logistic model and is 

- ideal for dose modeling 
- physically relevant parameters 
  - captures location $ED_{50}$ 
  - steepness $\lambda$ (Hill coefficient) 
    - $\lambda \equiv 1$ for simple $E_{max}$ model 
    - simple is easy $1/2$ dose, $1/2$ effect, complex, much 
  - placebo $E_0$ 

Criticism of $E_{max}$ 

- monotonic dose response, always increases with increasing dose (major) 
- symmetric about $ED_{50}$ on logscale 
  - Richard's model allows for asymmetry about $ED_{50}$ 
  - ![image-20221024145003232](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221024145003232.png)
    - Probably way harder to estimate 



When there's a lot of uncertainty associated with the Hill parameter, $\lambda$, it is often assumed that $\lambda \sim 1$ which makes no sense, less information and a stricter assumption? Acknowledge the uncertainty and model it 

- Great point - less information, you're more sure that $\lambda \equiv 1$ e.g. $\lambda = 1.5, \text{SE} = 0.3 \textrightarrow \lambda \equiv 1$ ??? 

---

## Normative framework 

Say we have a blinded framework composed of many models and for each model $i$ we get 

- data $y_{ijkl}$ where $j$ is over patient populations, $k$ is over individual patients, and $l$ are the time points 
- and there is both variability at the level of a single patient $\sigma_l$ and also variability between patients $\sigma_j$ 
- balance efficacy and toxicity, especially in settings where there are non-monotonic dosing effects 
- how does it depend on placebo population size 



Can we 

1. recover the correct model that generated the data as a function of $f(\sigma_l, \sigma_j, l)$ where we alter the noise parameters *and* the number of available time-points? 
2. design optimal future time points for sampling? 
   1. what happens if we use conventional 3+3 design vs continual reassessment [which has been shown to outperform 3+3](https://www.semanticscholar.org/paper/How-to-design-a-dose-finding-study-using-the-method-Wheeler-Mander/d694dbe34338c060a9cc856c1201d9e0c05db8c8) 
3. how confident am I in my model? my parameters? what are my model criticism checks? predictive power? 
   1. are there easily detectable failure modes, i.e. parameter falls in tail of prior? 
4. what if you don't get any data (poor coverage) in some sections of dose-response curve, what can we say? 
5. can you do binary and continuous endpoint? 
6. convert from dose to exposure to get rid of multimodality? 
7. can you superimpose two priors, one regular and one Dirac at zero to "switch off" the variable 
8. it's not sexy but I think reviewing the clinical procedures to see how the data is truly being generated and what the sensitivity and sources are would be phenomenally useful 
   1. what are your favourite datasets to work with? which are your least favourite, why? 



Quantification of the uncertainty in an experiment is phenomenally important, check out the visualization from [this](https://vega.github.io/vega/examples/hypothetical-outcome-plots/) archive which can cause us to "hallucinate" variations from an otherwise flat uniform distribution, here we could argue for strong seasonality when in reality there is absolutely none, **to do** simulate data from this distribution and try to recover the parameters, I think this is a really interesting investigation, make the model $y \sim \mathcal{N}(mx + b, \sigma^2)$  and see what the recovered parameters are 

> This would also be really important to show the failures of NLE optimization, the local maximum would be highly variable and specific to the dataset while the Hessian would hopefully cover the true values 

![image-20221104143831510](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221104143831510.png) 



If you work with $E_{max}$ model, recreate this plot from [here](https://www.youtube.com/watch?v=E713BehI2fE) which supposedly shows the range of expected values when conditioning on $\lambda$ as opposed to leaving it uncertain, covariance matrix is **exactly** the same with no covariance for the $\log(\lambda)$ 

![image-20221024144257976](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221024144257976.png)

## [Transformation of log-transformed variables back to real space](https://stats.stackexchange.com/questions/533804/covariance-of-log-transformed-variable) 

![image-20230111160519493](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230111160519493.png)

![image-20230111160545706](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230111160545706.png)

<img src="file:///C:/Users/dhattgak/AppData/Local/Temp/msohtmlclip1/01/clip_image001.jpg" alt="img" style="zoom:50%;" />

```R
temp.llpars <- rstan::extract(out$estanfit, pars = c('led50', 'loglambda', 'emax', 'e0[1]') ) 

temp.lpars.mean <- c(median(temp.lpars$led50), median(temp.lpars$lambda), median(temp.lpars$emax), median(temp.lpars$`e0[1]`)) 
temp.llpars.mean <- c(median(temp.llpars$led50), median(temp.llpars$loglambda), median(temp.llpars$emax), median(temp.llpars$`e0[1]`)) 

temp.lpars.cov <- cov(tibble(led50 = temp.lpars$led50, lambda = temp.lpars$lambda, emax = temp.lpars$emax, e0 = temp.lpars$`e0[1]`)) 
temp.llpars.cov <- cov(tibble(led50 = temp.llpars$led50, loglambda = temp.llpars$loglambda, emax = temp.llpars$emax, e0 = temp.llpars$`e0[1]`))

temp.lpars.tcov <- convert.cov.matrix(temp.lpars.mean, temp.lpars.cov, convert.led50 = TRUE, convert.llambda = FALSE) 
temp.lpars.tcov; sqrt(diag(temp.lpars.tcov)) 

temp.llpars.tcov <- convert.cov.matrix(temp.llpars.mean, temp.llpars.cov, convert.led50 = TRUE, convert.llambda = TRUE) 
temp.llpars.tcov; sqrt(diag(temp.llpars.tcov)) 
```



```R
  cov.trans <- matrix(nrow = 4, ncol = 4)
  
  if (convert.llambda & !convert.led50){
    ## unedited indices 
      ## on-diagonal elements 
    cov.trans[1, 1] <- cov.tmp[1, 1] 
    cov.trans[3, 3] <- cov.tmp[3, 3] 
    cov.trans[4, 4] <- cov.tmp[4, 4] 
    
      ## off-diagonal elements 
    cov.trans[1, 3] <- cov.tmp[1, 3] 
    cov.trans[3, 1] <- cov.trans[1, 3] 
    cov.trans[1, 4] <- cov.tmp[1, 4] 
    cov.trans[4, 1] <- cov.trans[1, 4] 
    cov.trans[3, 4] <- cov.tmp[3, 4] 
    cov.trans[4, 3] <- cov.trans[3, 4] 
    
    ## Get the log lambda parameters 
    mu.llambda <- mean.tmp[2]
    var.llambda <- cov.tmp[2, 2] 
    
    ## edited indices 
      ## on-diagonal elements 
    cov.trans[2, 2] <- (exp(2 * var.llambda) - exp(var.llambda)) * exp(2 * mu.llambda) 
    
      ## off-diagonal elements 
    cov.trans[1, 2] <- cov.tmp[1, 2] * exp(mu.llambda + var.llambda / 2) 
    cov.trans[2, 1] <- cov.trans[1, 2] 
    cov.trans[2, 3] <- cov.tmp[2, 3] * exp(mu.llambda + var.llambda / 2) 
    cov.trans[3, 2] <- cov.trans[2, 3] 
    cov.trans[2, 4] <- cov.tmp[2, 4] * exp(mu.llambda + var.llambda / 2) 
    cov.trans[4, 2] <- cov.trans[2, 4] 
    
    colnames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
    rownames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
  } 
  
  else if (convert.led50 & !convert.llambda){
    ## unedited indices 
      ## on-diagonal elements 
    cov.trans[2, 2] <- cov.tmp[2, 2] 
    cov.trans[3, 3] <- cov.tmp[3, 3] 
    cov.trans[4, 4] <- cov.tmp[4, 4] 
    
      ## off-diagonal elements 
    cov.trans[2, 3] <- cov.tmp[2, 3] 
    cov.trans[3, 2] <- cov.trans[2, 3] 
    cov.trans[2, 4] <- cov.tmp[2, 4] 
    cov.trans[4, 2] <- cov.trans[2, 4] 
    cov.trans[3, 4] <- cov.tmp[3, 4] 
    cov.trans[4, 3] <- cov.trans[3, 4] 
    
    ## Get the log lambda parameters 
    mu.led50 <- mean.tmp[1]
    var.led50 <- cov.tmp[1, 1] 
    
    ## edited indices 
      ## on-diagonal elements 
    cov.trans[1, 1] <- (exp(2 * var.led50) - exp(var.led50)) * exp(2 * mu.led50) 
    
      ## off-diagonal elements 
    cov.trans[1, 2] <- cov.tmp[1, 2] * exp(mu.led50 + var.led50 / 2) 
    cov.trans[2, 1] <- cov.trans[1, 2] 
    cov.trans[1, 3] <- cov.tmp[1, 3] * exp(mu.led50 + var.led50 / 2) 
    cov.trans[3, 1] <- cov.trans[1, 3] 
    cov.trans[1, 4] <- cov.tmp[1, 4] * exp(mu.led50 + var.led50 / 2) 
    cov.trans[4, 1] <- cov.trans[1, 4] 
    
    colnames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
    rownames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
  } 
  
  else if (convert.llambda & convert.led50){
    ## unedited indices 
      ## on-diagonal elements 
    cov.trans[3, 3] <- cov.tmp[3, 3] 
    cov.trans[4, 4] <- cov.tmp[4, 4] 
    
      ## off-diagonal elements 
    cov.trans[3, 4] <- cov.tmp[3, 4] 
    cov.trans[4, 3] <- cov.trans[3, 4] 
    
    ## Get the log lambda and log ED50 parameters 
    mu.led50 <- mean.tmp[1] 
    var.led50 <- cov.tmp[1, 1] 
    mu.llambda <- mean.tmp[2] 
    var.llambda <- cov.tmp[2, 2] 
    
    ## edited indices 
      ## on-diagonal elements 
    cov.trans[1, 1] <- (exp(2 * var.led50) - exp(var.led50)) * exp(2 * mu.led50) 
    cov.trans[2, 2] <- (exp(2 * var.llambda) - exp(var.llambda)) * exp(2 * mu.llambda) 
    
      ## off-diagonal elements 
        ## new elements 
    cov.trans[1, 2] <- cov.tmp[1, 2] * exp(mu.llambda + var.llambda / 2) * exp(mu.led50 + var.led50 / 2) 
    cov.trans[2, 1] <- cov.trans[1, 2] 
        ## repeated elements 
          ## log ED50 
    cov.trans[1, 3] <- cov.tmp[1, 3] * exp(mu.led50 + var.led50 / 2) 
    cov.trans[3, 1] <- cov.trans[1, 3] 
    cov.trans[1, 4] <- cov.tmp[1, 4] * exp(mu.led50 + var.led50 / 2) 
    cov.trans[4, 1] <- cov.trans[1, 4]
          ## log lambda 
    cov.trans[2, 3] <- cov.tmp[2, 3] * exp(mu.llambda + var.llambda / 2) 
    cov.trans[3, 2] <- cov.trans[2, 3] 
    cov.trans[2, 4] <- cov.tmp[2, 4] * exp(mu.llambda + var.llambda / 2) 
    cov.trans[4, 2] <- cov.trans[2, 4] 
    
    colnames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
    rownames(cov.trans) <- c('ed50', 'lambda', 'emax', 'e0') 
  }

  return(cov.trans)
}

```



### Identification is only defined in the limit of infite data 

> In addition, “identification” is formally defined in statistics as an asymptotic property, but in Bayesian inference we care about inference with finite data, especially given that our models often increase in size and complexity as more data are included into the analysis. Asymptotic results can supply some insight into finitesample performance, but we generally prefer to consider the posterior distribution that is in front of us.

## Bernard-von Mises theorem 

From "Bayesian Workflow" 

> Generally, an HMC-based sampler will work best if its mass matrix is appropriately tuned and the geometry of the joint posterior distribution is relatively uninteresting, in that it has no sharp corners, cusps, or other irregularities. This is easily satisfied for many classical models, where results like the Bernstein-von Mises theorem suggest that the posterior will become fairly simple when there is enough data. Unfortunately, the moment a model becomes even slightly complex, we can no longer guarantee that we will have enough data to reach this asymptotic utopia (or, for that matter, that a Bernstein-von Mises theorem holds). For these models, the behavior of HMC can be greatly improved by judiciously choosing a parameterization that makes the posterior geometry simpler. 
>
> For example, hierarchical models can have difficult funnel pathologies in the limit when grouplevel variance parameters approach zero (Neal, 2011), but in many such problems these computational difficulties can be resolved using reparameterization, following the principles discussed by Meng and van Dyk (2001); see also Betancourt and Girolami (2015).



### Zero-avoiding (lognormal, inverse gamma) priors on the group-level variances help avoid funnneling 

> When trying to avoid the funnel pathologies of hierarchical models in the limit when group-level variance parameters approach zero, one could use zero-avoiding priors (for example, lognormal or inverse gamma distributions) to avoid the regions of high curvature of the likelihood; a related idea is discussed for penalized marginal likelihood estimation by Chung et al. (2013, 2014). Zero-avoiding priors can make sense when such prior information is available—such as for the length-scale parameter of a Gaussian process (Fuglstad et al., 2019)—but we want to be careful when using such a restriction merely to make the algorithm run faster. At the very least, if we use a restrictive prior to speed computation, we should make it clear that this is information being added to the model.



### LOO works 

>  Gabry et al. (2019) provide an example where LOO-CV indicated problems that motivated efforts to improve the statistical model.

<img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20230327182633733.png" alt="image-20230327182633733" style="zoom:67%;" />

#### Framing LOO as an augmented problem where the gradient is computed across the product of data points raised to the power $\alpha_i$ 

> An alternative approach to importance weighting is to frame the removal of data points as a gradient in a larger model space. Suppose we have a simple independent likelihood, $\prod^n_{i=1} p(y_i \mid \theta)$, and we work with the more general form, $\prod^n_{i=1} p(y_i \mid \theta)^\alpha_i$ , which reduces to the likelihood of the original model when $\alpha_i = 1$ for all $i$. Leave-one-out cross validation corresponds to setting $\alpha_i = 0$ for one observation at a time. But another option, discussed by Giordano et al. (2018) and implemented by Giordano (2018), is to compute the gradient of the augmented log likelihood as a function of α: this can be interpreted as a sort of differential cross validation or influence function.

That's so cool. 

#### Extension to hierarchical models 

> Cross validation for multilevel (hierarchical) models requires more thought. Leave-one-out is still possible, but it does not always match our inferential goals. For example, when performing multilevel regression for adjusting political surveys, we are often interested in estimating opinion at the state level. A model can show real improvements at the state level with this being undetectable at the level of cross validation of individual observations (Wang and Gelman, 2016). Millar (2018), Merkle, Furr, and Rabe-Hesketh (2019), and Vehtari (2019) demonstrate different cross validation variants and their approximations in hierarchical models, including leave-one-unit-out and leaveone-group-out. In applied problems we have performed a mix, holding out some individual observations and some groups and then evaluating predictions at both levels (Price et al., 1996). 
>
> Unfortunately, approximating such cross validation procedures using importance sampling tends to be much harder than in the leave-one-out case. This is because more observations are left out at a time which implies stronger changes in the posterior distributions from the full to the subsetted model. As a result, we may have to rely on more costly model refits to obtain leave-one-unit-out and leave-one-group-out cross validation results.
