# OLS
# dependent variable - hours studied in a week
# independent variable - credits, work(hours per week), score(SAT), upper(binary var)=1 if more than 2 yr of uni, =0 otherwise
# simulating data
## 1. credits - discrete random variable -> find distribution: pmf for credits
### if 9 credits: 0.2
###   10 credits: 0.03
###   11 credits: 0.05
###   12 credits: 0.23
###   13 credits: 0.05
###   14 credits: 0.07
###   15 credits: 0.17

# assume we have 5000 students

#Sample size 
n<-5000
#Probability
prob <- c(0.2, 0.03, 0.05, 0.23, 0.05, 0.07, 0.17)
# sample of credits
credits <- sample(9:15, n, replace=TRUE, prob=prob)
#barplot
table(credits)
barplot(table(credits))

# work 1-10 hours: 0.45, 11-20 hours: 0.55, equally likely within their subset
# probabilities
probwork <- c(rep(0.045,10),rep(0.055,10))
probwork
# work
work <- sample(1:20, n, replace = TRUE, prob = probwork)
# barplot
barplot(table(work))

# score(SAT score) -> continuous: normal random variable, N(1600,200)
## normal N(1600,200)
score <- rnorm(n, mean = 1600, sd = 200)
## normality check
### plot
hist(score, prob=TRUE)
lines(density(score),col="blue")

# upper: create binomial
upper <- rbinom(n,1,0.45)
# plot
barplot(table(upper))

# Disturbance term
epsilon <- rnorm(n,mean = 1, sd = 1.7)
# plot
plot(density(epsilon))

# simulating the dependent variable
## intercept
beta_0 <- 1.2

# credits
beta_1 <- 0.03

# work
beta_2 <- -0.02

# scores(SAT)
beta_3 <- 0.015

# upper
beta_4 <- 0.5

# Beta
beta <- c(beta_0, beta_1, beta_2, beta_3, beta_4)

## create matrix X
X <- cbind(rep(1,n), credits, work, score, upper)
View(X)

## generate dependent
y <- X%*%beta+epsilon

# performing OLS, to get beta hat
betahat <- solve(t(X)%*%X)%*%t(X)%*%y
betahat
beta

##### easy way: OLS --> lm
regression <- lm(y~credits+work+score+upper)
coefficients(regression)
summary(regression)
regression$coefficients


