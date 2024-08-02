# BBB Best coding practices 

Clean column names, transmute (mutate columns in place), pretty printing of dataframes with randomly drawn slices sampled from the dataframe, from [here](https://matthewrkaye.com/posts/2023-03-25-balancing-classes/balancing-classes.html)

```
library(readr)
library(dplyr)
library(janitor)
library(purrr)
library(mlbench)

data(BreastCancer)

data <- BreastCancer %>%
  clean_names() %>%
  transmute(
    cl_thickness = as.numeric(cl_thickness), 
    class
  ) %>%
  as_tibble()

data %>%
  slice_sample(n = 5) %>%
  pretty_print()
```

Count and print the proportions 

```
data %>%
  count(class) %>%
  mutate(prop = n / sum(n)) %>%
  pretty_print()
```





Calibration plots for {tidymodels}, from [here](https://matthewrkaye.com/posts/2023-03-25-balancing-classes/balancing-classes.html)



Now, I’ll fit a simple logistic regression model by specifying `logistic_reg()` as the model specification in `fit_model()`.

```
library(probably)

unbalanced_model <- fit_model(
  data,
  logistic_reg()
)

preds <- tibble(
  truth = data$class,
  truth_int = as.integer(data$class) - 1,
  estimate = predict_prob(unbalanced_model, data)
)
```

And now we can make a calibration plot of our predictions. Remember, the  goal is to have the points on the plot lie roughly along the line `y = x`. Lying below the line means that our predictions are too high, and above the line means our predictions are too low.

```
cal_plot_breaks(preds, truth = truth, estimate = estimate, event_level = "second")
```

<img src="C:/Users/dhattgak/AppData/Roaming/Typora/typora-user-images/image-20231030142730192.png" alt="image-20231030142730192" style="zoom:50%;" />

Awesome! Even with the class imbalance, our model’s probability  predictions are well-calibrated. In other words, when we predict that  there’s a 25% chance that a tumor is malignant, it’s actually malignant  about 25% of the time.



Find the arguments of a function 

```
args(sum) 
#function(..., na.rm = FALSE)
```



modify all characters to factors 

```%>%
%>% mutate_if(is.character, as.factor)
```



move columns around 

```
select(iris, Species, everything()) 
## or 
relocate(Out) # to front 
relocate(Out, In) # to front 
relocate(Out, .before = In) %>% 
relocate(Out, .after = In)
```



change single values across columns 

```
 %>% mutate_at(vars(AGE:FORM), replace_na, missing.nm)
```



replace all values at specific indices, with an argument for strictness 

```
mapper <- function (x, from, to, strict = TRUE, ...) {
  stopifnot(length(to) == length(from))
  res <- to[match(x, table = from)]
  if (!strict)
  res[!(x %in% from)] <- x[!(x %in% from)]
res
}

## an example 
## Here's an example to illustrate:

##    x = c(1, 2, 3, 4, 5)
##    from = c(1, 3, 5)
##    to = c('a', 'b', 'c')

## If strict = TRUE, res will be c('a', NA, 'b', NA, 'c').
## If strict = FALSE, res will be c('a', 2, 'b', 4, 'c').
```



reorder such that all numeric columns are first 

```
char.col.at.last <- function(data,col.first=NULL){
  if (!is.data.frame(data)) stop('a data frame is required for this function')
  if (!is.null(col.first)) {
    match(col.first,names(data)) -> temp
    setdiff(1:dim(data)[2],temp) -> temp2
    data[c(temp,temp2)] -> data
  }
#
  grep(FALSE,sapply(data,is.numeric)) -> char.col
  grep(TRUE,sapply(data,is.numeric)) -> num.col

  data[c(num.col,char.col)] -> data
return(data)
}
```



install from yaml 

````
```{r Install packages, include=FALSE}
system("pkgr plan")
system("pkgr install")
```

## Example yaml 
Version: 1
# Specify the version of R used in the analysis
# The packages for the specified R version will be installed
# The available versions of R are listed below ... 
# ... comment out the version that is not used
#RPath: /opt/R/3.5.3/bin/R
#RPath: /opt/R/3.6.3/bin/R
RPath: /opt/R/4.0.3/bin/R

# NOTE: The dash '-' character preceeding each item in the lists under 
# "Packages" and "Repos" should be in column 3 
# 
Packages:
# List the packges to be installed
  - rmarkdown 
# 
# tidyverse related packages
  - tidyverse # dplyr, ggplot2, tibble, readr, tidyr, purrr, stringr, forcats
  - tidyselect
  - haven
  - lubridate
  - hms
  - labelled
  - tidylog
  - tidyvpc
  - vpc
  - mvtnorm
  
#  
#  Analysis related packages
  - xpose4
  - Hmisc
  - reshape2
  - survival
  - gmodels
  - arsenal
  - table1
  - GGally
  - gridExtra
  - ggpubr
  - ggsci
  - PKNCA
  - ggforce
  - DT

# QC related packages
  - styler
  - lintr
  - httpuv
  - xtable
  - sourcetools
  - promises
  - later
  - shiny
  - miniUI
  - sessioninfo
  
Repos:
# Formal analyses should only utilize packages installed from the MPN repository
# Exploratory analyses may utilize packages installed from the CRAN or other 
# repositories
#
#  - MPN: https://mpn.metworx.com/snapshots/stable/2019-10-04
#  - MPN: https://mpn.metworx.com/snapshots/stable/2020-06-08
  - MPN: https://mpn.metworx.com/snapshots/stable/2020-11-21
#
# Only uncomment CRAN or add other repositories to be searched if the analysis is
# exploratory 
#  - CRAN: https://cran.rstudio.com/ 
# 
Library: analysis-library
# All packages should be installed in the ./scripts/analysis-library/ directory
````



converting from `\analysis-library/` to `{renv}` based project 

````
```{r Set up renv, eval = FALSE}

# Install the version of renv in the MPN repository specified in .Rprofile
#install.packages("renv")

#library(renv)

# Initialize a new project (with an empty R library)
#renv::init(bare = TRUE)
```


```{r Install packages, eval = FALSE}

# Check package installation path points to a sub-directory of ./scripts/renv/ 
.libPaths()

# Run pkgr install
# The "update" option will ensure that the package versions will be installed from the repository specified in the pkgr.yml, consistent with the current version of R and other packages
system("pkgr install --update")

# Call snapshot() to create a lockfile capturing the state of a project's R package dependencies. The lockfile can be used to later restore these project's dependencies as required
renv::snapshot()
```
````

