######## MLE: Maximum Likelihood Estimation ##############

install.packages("AER")
# Load library
library(AER)

#call the dataset from the library
data("CPS1985")
summary(CPS1985)
head(CPS1985)
#preprocessing data
# level(): categories  #paste: to create new col
for(level in unique(CPS1985[,"married"])){
        CPS1985[paste("dummy_married", level, sep="_")]<-
                ifelse(CPS1985[,"married"]==level,1,0)
}
View((CPS1985))
head(CPS1985$dummy_married_yes)

catego<-c("ethnicity", "gender")
for( i in 1:length(catego)){
        for(level in unique(CPS1985[,catego[i]])) {
                CPS1985[paste("dummy",catego[i], level, sep = "_")] <- ifelse(CPS1985[,catego[i]] == level, 1, 0)
        }
}
names(CPS1985)
View(CPS1985)

#create a vector of the variables I don't want
destroy<-c("wage",
           "married","dummy_married_no",
           "ethnicity" ,"dummy_ethnicity_other", 
           "region",
           "gender", "dummy_gender_male",
           "occupation",
           "union",
           "sector")
#Let's get the number of observations
n<-nrow(CPS1985)
#Dependent
y<-log(CPS1985[,"wage"])
#Independent
X<-cbind(rep(1,n), CPS1985[,-which(colnames(CPS1985) %in% destroy)])

#prepare the log likelihood function
k<-ncol(X)
loglikReg<-function(x,y,X,n,k){
        beta<-x[1:k]
        sig2<-x[(k+1)]^2    #square to keep positive
        X<-as.matrix(X)
        L<- -(n/2)*log(2*pi)-(n/2)*log(sig2)-((1/(2*sig2))*t((y-X%*%beta))%*%(y-X%*%beta)) 
        llik<-sum(L)
        return(-llik)
}

# OLS
Xnew<-data.matrix(X[,2:k])
OLSE<-lm(y~Xnew)
summary(OLSE)

#save the parameter values from OLS to use as initial
#guesses in MLE
BetaOLS<-OLSE$coefficients
u<-OLSE$residuals
SigSq<-(t(u)%*%u)/(n-k)
#set these as initial parameters
Init<-c(BetaOLS, SigSq)

#run the optimization: optimizing the log-likelihood
opt<-optim(Init,loglikReg,gr=NULL,y=y,X=X,n=n,k=k, control=list(maxit=1000,abstol=0.000000001,trace=1),hessian=TRUE, 
           method="BFGS")
#look at the parameters
opt$par
Init
opt$value
