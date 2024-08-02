# R notes 



## Beginner R 



### General notes 

- `summary()` is a nice tool 
- `head()` gives first elements, `tail()` gives last 
- `?dataframe` gives variable meanings 
- `str()` gives you an overview of the data 
- `args(sd)` gives you the args without needing to read through the entire documentation 
- `R` passes arguments **by value** 
  - which means that an `R` function cannot affect the variables passed to the function 

- `install.packages("ggvis")` 
  - `search()` finds any package we want 
  - `library("ggvis")` loads the package we want 
    - useful to use `result <- require("data.table")` where the output is `result = FALSE` if the package is not found but the program does not crash if it is not found 

- `lapply(nyc, class)` is the same as list comprehension in Python (i.e. `print(class(elem)) for elem in elements`) 
  - `lapply()` always returns a list, to convert it to a vector, call `unlist(lapply())` 
  - to include additional arguments, for `multiply <- function(x, mult){ ... }` with `result <- lapply(input, multiply, mult = 4)` 

- You can create anonymous functions if you don't need to reuse the function by calling `function(x) {3*x}` 
- `sapply()` is an alternative to `lapply()` which does not return a `list` , it returns a `list` of vectors (an `array`)! 
  - similar to `unlist(lapply(cities, nchar))` 
  - if all the variables are of the same type, `sapply()` works to `simplify apply` to all elements 
    - the result is a named vector, though the `USE.NAMES = FALSE` flag will have the same behavior

  - `vapply()` explicitly takes the output format of the array 
    - `vapply(X, FUN, FUN.VALUE, ..., USE.NAMES=TRUE)`  

- `cat()` is a nice print statement 
  - `cat("The average temperature is", mean(x), "\n")`




### Lists 

- Lists are like a super data type, you can put anything you want in them `my_list <- list(my_vector, my_matrix, my_df)` 

- **Named lists**, both are equivalent 

- ```
  my_list <- list(name1 = your_comp1, 
                  name2 = your_comp2)
  my_list <- list(your_comp1, your_comp2)
  names(my_list) <- c("name1", "name2")
  ```

- Selecting elements in lists is a bit different, check the double braces, it would look something like 

- ```
  shining_list[["reviews"]]
  shining_list$reviews
  ```

  - or `shining_list[['reviews']][2]` 

### Variable assignment & class declaration, ordering  

- `a <- 2 `
- `class(a)` returns integer 
- R is **1 indexed** 
- `a[order(a)]`, remember that `order(a)` only sorts the *indices* but does not manipulate the original array, behaves similar to `argsort()` 

### Vector creation + indexing tips 

- `c()` is basically a list 

- `1:3` is a shortcut for `c(1, 2, 3)`

  - You can construct a matrix in R with the [`matrix()`](http://www.rdocumentation.org/packages/base/functions/matrix) function. Consider the following example:

    ```
    matrix(1:9, byrow = TRUE, nrow = 3)
    ```

    - `byrow = TRUE` is for row filling, `byrow = FALSE` is for column filling 
    - all elements of first column `matrix[,1]`, first row `matrix[1,]` 

- To select multiple elements from a vector, you can add square brackets at the end of it. You can indicate between the brackets what elements should be selected. For example: suppose you want to select the first and the fifth day of the week: use the vector `c(1, 5)` between the square brackets. For example, the code below selects the first and fifth element of `poker_vector`:

  ```R
  poker_vector[c(1, 5)]
  ```

  - So, another way to find the mid-week results is `poker_vector[2:4]`. Notice how the vector `2:4` is placed between the square brackets to select element 2 up to 4. 

  - Just like you did in the previous exercise with numerics, you can also use the element names to select multiple elements, for example:

    ```
    poker_vector[c("Monday","Tuesday")]
    ```

- `cbind()` is a function that adds columns (or multiple columns) to a matrix by merging them 
  - `rbind()` does the same thing for rows 

### Names 

- You can give a name to the elements of a vector with the `names()` function. Have a look at this example:

```
some_vector <- c("John Doe", "poker player")
names(some_vector) <- c("Name", "Profession")
```



### Factors & categorical variables 

- `factor()` is basically creating a `set` of categorical variables 
- Two orderings of categorical variables 
  - Nominal categorical variables have no implied order 
    - `factor_animals_vector <- factor(animals_vector)` 
  - Ordinal do 
    - "Low", "Medium", "High" 
    - `factor_temperature_vector <- factor(temperature_vector, order = TRUE, levels = c("Low", "Medium", "High"))`
- `levels()` changes the category, *watch out for the ordering here* 



### Dataframes 

- `planets_df[1:3, "rings"]` is perfectly valid and selects the first 3 elements 
- `planets_df[, "rings"]` which is the same as `planets_df$rings`
  - You can mask this to print out the names where `planets_df[planets_df$rings, "name"]` 
  - A shortcut to this is using `subset` (i.e. `subset(planets_df, subset=diameter < 1)`) 



## Intermediate R 

### And (&) and or (|) 

- `y < 5 | y > 14` 
- `c(TRUE, TRUE, FALSE) & c(TRUE, FALSE, FALSE)`
  `TRUE, FALSE, FALSE` 
- But, check out what happens with `&&` , it **only evaluates the first element of each vector**
  - `c(TRUE, TRUE, FALSE) && c(TRUE, FALSE, FALSE)` 
    `TRUE` 
- Same deal with `||` 



### ifs, elses, whiles, fors 

- ``` 
  if (condition){ ... 
  	} else { ... 
  }
  ```

  - be really careful about where the `else` is, it must be in line with closing `{` of if! 

- while loops can use `break` to exit 

- ```
  while (condition){ 
  	if (condition){ 
  		break 
  	}
  	print(...) 
  }
  ```

- ``` 
  for (city in cities){ 
  	print(city) 
  }
  ```

- you can also use `break` and `next` statements in `for` loops 

- these next two statements are equivalent 

- ```R
  for(elem in elements){
      print(elem)
  }
  for(i in 1:length(elements)){
      print(elements[i])
  } 
  ```

- **Remember** that vectors are easy `elements[i]`, **lists are different ** `elements[[i]]` 

  - Also don't forget your `1:length(elements)` 


### Functions 

```R
my_fun <- function(arg1, arg2) { 
	body 
} 

my_fun <- function(arg1, arg2) {
    y <- x^2 
    return(y) 
}
```

- Functions with no explicit `return` **must** `return(TRUE)`! 

### Useful functions 

- `sort(), print(), identical(), is*(), as*(), append(), rev()` 
- `str()` 

### Regular expressions (regex) 

- start with `?regex` if you want 
- `grepl(pattern = "", x = vector)` 
  - to look for starting character `pattern = "^a"` , end of line `pattern = "$a"` 
  - returns a vector of logicals 
- `grep(pattern = "", x = vector)` 
  - returns the **indices** of the `TRUE` results 
  - identical behavior to `which(grepl(pattern = "", x=vector))` 
- Metacharacters 
  - You can use the caret, `^`, and the dollar sign, `$` to match the content located in the start and end of a string, respectively. This could take us one step closer to a correct pattern for matching only the ".edu" email addresses from our list of emails. But there's more that can be added to make the pattern more robust:
    - `@`, because a valid email must contain an at-sign.
    - `.*`, which matches any character (.) zero or more times (*). Both the dot and the asterisk are metacharacters. You can use them to match any character between the at-sign and the ".edu" portion of an email address.
    - `\\.edu$`, to match the ".edu" part of the email at the end of the string. The `\\` part *escapes* the dot: it tells R that you want to use the `.` as an actual character.
    - `.*`: A usual suspect! It can be read as "any character that is matched zero or more times".
    - `\\s`: Match a space. The "s" is normally a character, escaping it (`\\`) makes it a metacharacter.
    - `[0-9]+`: Match the numbers 0 to 9, at least once (+).
    - `([0-9]+)`: The parentheses are used to make parts of the matching string available to define the replacement. The `\\1` in the `replacement` argument of [`sub()`](https://www.rdocumentation.org/packages/base/functions/grep) gets set to the string that is captured by the regular expression `[0-9]+`.
- `sub(pattern = <regex>, replacement = <str> , x = <str>)` 
  - `sub()` changes `impala -> impola`, only replaces the first hit of the pattern 
  - `gsub()` changes `impala -> impolo` and replaces all incidences 
- `gsub(pattern = "a|i", replacement = "_", x = animals)` 
  - or character `|` looks for either character for string replacement 



### Dates 

- ```R
  today <- Sys.Date() 
  today 
  "2022-10-17" 
  ```

- ``` 
  class(today)
  "Date"
  ```

- ```R
  now <- Sys.time() 
  ```

- To create a `Date` object from a simple character string in R, you can use the [`as.Date()`](https://www.rdocumentation.org/packages/base/functions/as.Date) function. The character string has to obey a format that can be defined using a set of symbols (the examples correspond to 13 January, 1982):

  - `%Y`: 4-digit year (1982)
  - `%y`: 2-digit year (82)
  - `%m`: 2-digit month (01)
  - `%d`: 2-digit day of the month (13)
  - `%A`: weekday (Wednesday)
  - `%a`: abbreviated weekday (Wed)
  - `%B`: month (January)
  - `%b`: abbreviated month (Jan)

  The following R commands will all create the same `Date` object for the 13th day in January of 1982:

  ```
  as.Date("1982-01-13")
  as.Date("Jan-13-82", format = "%b-%d-%y")
  as.Date("13 January, 1982", format = "%d %B, %Y")
  ```

  Notice that the first line here did not need a format argument, because by default R matches your character string to the formats `"%Y-%m-%d"` or `"%Y/%m/%d"`.

  In addition to creating dates, you can also convert dates to character strings that use a different date notation. For this, you use the [`format()`](https://www.rdocumentation.org/packages/base/functions/format) function. Try the following lines of code:

  ```R
  today <- Sys.Date()
  format(Sys.Date(), format = "%d %B, %Y")
  format(Sys.Date(), format = "Today is a %A!")
  ```

Similar to working with dates, you can use [`as.POSIXct()`](https://www.rdocumentation.org/packages/base/functions/as.POSIXlt) to convert from a character string to a `POSIXct` object, and [`format()`](https://www.rdocumentation.org/packages/base/functions/format) to convert from a `POSIXct` object to a character string. Again, you have a wide variety of symbols:

- `%H`: hours as a decimal number (00-23)
- `%I`: hours as a decimal number (01-12)
- `%M`: minutes as a decimal number
- `%S`: seconds as a decimal number
- `%T`: shorthand notation for the typical format `%H:%M:%S`
- `%p`: AM/PM indicator

For a full list of conversion symbols, consult the `strptime` documentation in the console 



# Data visualization in R 

- ~~~ R
  # Plot multiple time-series by grouping by species
  ggplot(fish.tidy, aes(Year, Capture)) +
    geom_line(aes(group = Species)) 
  
  # Plot multiple time-series by coloring by species
  ggplot(fish.tidy, aes(x = Year, y = Capture, color = Species)) +
    geom_line()
  ~~~

  - <img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221101105146503.png" alt="image-20221101105146503" style="zoom:15%;" />

- Categorical variables are designated using `factor()` 

- In general `ggplot` plots are called with the syntax `ggplot(<dataset>, aes(x=<xdata>, y=<ydata>)) + geom_point(<geom_point_args>)` 
  - `fill` is different from `color` which typically is the outline 
  - `size` 
  - `alpha` 
  - `linetype` is -- 
  - `labels` 
  - `shape` 
    - The default `shape` is `shape = 19` which is a solid circle 
      - A useful alternative is `shape = 21` which is a circle that allows you to both `fill` the inside *and* `color` the outline  

- `labs()` sets the x and y labels 

- Univariate plots require a 'fake' y-axis by mapping y to zero 

- ![image-20221017152251903](C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221017152251903.png)



### Histograms 

- `ggplot(iris, aes(x = Sepal.Width)) + geom_histogram()` 

### Themes 

- Themes controls all visual parts of the plot that are not data related 
- <img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221101111024145.png" alt="image-20221101111024145" style="zoom:40%;" />
- <img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221101111101713.png" alt="image-20221101111101713" style="zoom:33%;" />

```R
p + theme(axis.line = element_line(color = "red", linetype = "dashed"))

```

Remember when calling say `theme( ..., panel.grid.major.y = (element.line(color = "white", size = 0.5, linetype = "dotted")))` that the `element.line` is inside `panel.grid.major.y` which is inside the `theme` 

- You can call say `plot + theme_classic() + theme(text = element_text(family = "serif"))` to manually **override** any of the elements of the original theme 

- Updating or setting the default theme can be done by `original = theme.update(...)` 

  - <img src="C:\Users\dhattgak\AppData\Roaming\Typora\typora-user-images\image-20221101131748164.png" alt="image-20221101131748164" style="zoom:50%;" />

- Then, `theme_set(original)`, for example 

  - ~~~R
    theme_set <- theme(
      rect = element_rect(fill = "grey92"),
      legend.key = element_rect(color = NA),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.grid.major.y = element_line(color = "white", size = 0.5, linetype = "dotted"),
      axis.text = element_text(color = "grey25"),
      plot.title = element_text(face = "italic", size = 16),
      legend.position = c(0.6, 0.1)
    )
    ~~~

  - Remember that custom themes are **not functions** and are therefore just called by `theme_custom` while themes like Tufte are called using `theme_tufte()` with the parentheses 

  - Apparently this is also ok `labs(title = "Highest and lowest life expectancies, 2007", caption = "Source: gapminder")` 

- `coord_*` controls the dimensions of your plot 

  - Most commonly used is `coord_cartesian()` 
  - Can also use `plot + xlim(c(4.5, 5.5))` but **better** to use `plot + coord_cartesian(xlim = c(4.5, 5.5))` because it doesn't truncate the data 
    - Apparently $xlim$ and $ylim$ both have weird behavior so don't use them until you need it 
    - $xlim$ literally cuts off the data not used in the plot, $coord_cartesian()$ does not so use it instead 

- `coord_cartesian(expand = 0, clip = "off")`, `expand = 0` can be used to make the boundaries of the plot flush with the data and `clip = "off"` means that the data on the boundary will not be cut off, typically paired with `theme(axis.line = element_blank())` to remove the axis lines 



- `scale_*_log10()` functions transform the data *before* inputting them into the `ggplot2` plot 
  - use these 
- `coord_trans()` with `x = "log10"` or `y = "log10"`  transforms the data *after* inputting into plot 
  - do not use this, weird behavior 



### Facets 

Concept of small multiples coined by Tufte in "Visualization of Graphical Information", mainlyu used to add **an additional categorical variable to our plot** 

- basically is `plot + facet_grid(cols = vars(Species))` 

  - this is equivalent to `plot + facet_grid(. ~ Species)` 

    - which is formula notation, everything on the left of the `~` will split according to rows and everything on the right will split according to columns 

  - ```
    | Modern notation                            | Formula notation  |
    |--------------------------------------------|-------------------|
    | facet_grid(rows = vars(A))                 | facet_grid(A ~ .) |
    | facet_grid(cols = vars(B))                 | facet_grid(. ~ B) |
    | facet_grid(rows = vars(A), cols = vars(B)) | facet_grid(A ~ B) |
    ```

- Use the `labeller` argument inside `facet_grid()` to efficiently label axes in a shared axis plot 
  - when using `facet_grid()`, it may be important to use `scale = "free_x", "free_y", or "free"` to get rid of blank data in addition to the `space = "free_x",  ...` argument which typically has the same argument 
  - `fct_recode` allows you to rename variables efficiently, `fct_relevel` allows you to order variables in a non-alphabetical order 



# Bayesian data analysis in R 

- Generally, `rpois` or `runif` are the distributions that can be used to sample values from known, common distributions 

- Calculating probabilities from a vectorized density distribution can be done using `dpois`, the two blocks of code are equivalent and much more efficient 

  - ~~~ 
    n_ads_shown <- 100
    proportion_clicks <- 0.1
    n_visitors <- rbinom(n = 99999, 
        size = n_ads_shown, prob = proportion_clicks)
    prob_13_visitors <- sum(n_visitors == 13) / length(n_visitors)
    ~~~

  - ~~~ 
    dbinom(x = 13, size = n_ads_shown, prob = proportion_clicks)
    ~~~

- To enumerate over a bivariate grid of parameters, say `n_visitors` and `proportion_clicks`, 

  - ~~~ 
    pars <- expand.grid(proportion_clicks = proportion_clicks, 
    n_visitors = n_visitors) 
    ~~~

  - and to add in the prior for say `proportion_clicks` 

  - ~~~ 
    proportion_clicks <- runif(n_samples, min = 0.0, max = 0.2) 
    pars$prior <- dunif(pars$proportion_clicks, min = 0, max = 0.2) 
    ~~~

  - 



### [Dataframe tutorial](https://www.youtube.com/watch?v=oVhDN7TfC08) 

~~~ R
stocks <- data.frame(
	company = company, 
	sector = c("tech", "financial", "healthcare"), 
	buy = c(TRUE, FALSE, TRUE)
	stringsAsFactors = FALSE)
~~~

- Useful to not let `R` treat strings as categorical variables using the `stringsAsFactors` argument 

- You can add columns of identical length using `pe_ratio <- c(14.3, 15.7, 4.2)` 

  - `stocks$pe <- pe_ratio` adds a column! 
  - can also use `stocks <- cbind(ticker, stocks, stringsAsFactors = FALSE)` which accompishes the same goal but with the default column name only 

- Be careful when adding new **rows** with different variable types, for example 

  - `yext <- c("YEXT", "Yext Inc", "tech", 0.23, TRUE, NA)` will be **coerced** such that `0.23` becomes a **string** 
  - however, if you `yext <- list("YEXT", "Yext Inc", "tech", 0.23, TRUE, NA)` then thats fine with `stocks <- rbind(stocks, yext, stringsAsFactors = FALSE)` 

- To delete a column just use `stocks$ticker <- NULL`

  

- `stocks[1]` returns the first column as a **dataframe** 

- `stocks[[1]]` returns the first column as a **vector** 

  - same as `stocks$company` 



- `subset()` seems **phenomenally useful**, say `subset(stocks, select = sector, subset = revenue > 200)` 
  - can also `subset(stocks, select = c(sector, revenue), subset = revenue > 200)`  to select **both** `sector` and `revenue` 



Tibbles 

- Does not change inputs, `stringsAsFactors = FALSE` not needed 
- Better printouts than dataframe 



~~~
library(tibble) 
stocks_tbl <- as.tibble(stocks) 
str(stocks_tbl) 
~~~



