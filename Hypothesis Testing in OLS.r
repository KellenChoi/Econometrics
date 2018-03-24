################## Hypothesis Testing in OLS #######################

setwd("/Users/eunheechoi/R/Econometrics/")
wagedata<-read.csv("wage.csv")
wagedata
getwd()

RegOLS<-lm(wage~female+nonwhite+unionmember+education+experience, data=wagedata)
#Coefficients
b<-coef(RegOLS)
b
#Variance Covariance
Vb<- vcov(RegOLS)
Vb


##### Example 1 HA: “A female worker earns $3 less than male worker”
Rmat<-matrix(c(0, 1, 0, 0, 0, 0),nrow=1)
q<- -3
J<-nrow(Rmat)
n<-nrow(wagedata)
Fstat<-(1/J)* t(Rmat %*% b-q) %*% solve(Rmat %*% Vb %*% t(Rmat)) %*% (Rmat%*%b-q)
Fstat

k<-ncol(Vb)
pval<-1-pf(Fstat,J,n-k)
pval



######## Example 2 HA: “One year of education is worth eight years of experience”
Rmat<-matrix(c(0, 0, 0, 0, 1, -8),nrow=1)
q<- 0
Fstat<-(1/J)* t(Rmat %*% b-q) %*% solve(Rmat %*% Vb %*% t(Rmat)) %*% (Rmat%*%b-q)
Fstat
pval<-1-pf(Fstat,J,n-k)
pval


######## Example 3 HA: “All of the above, plus”A unionized worker earns $1 more than a nonunionized worker“.
Rmat1<-matrix(c(0, 1, 0, 0, 0, 0),nrow=1)
Rmat2<-matrix(c(0, 0, 0, 0, 1, -8),nrow=1)
Rmat3<-matrix(c(0, 0, 0, 1, 0, 0),nrow=1)
Rmat<-rbind(Rmat1,Rmat2,Rmat3)
q<- matrix(c(-3,0,1),nrow=3)
J<-nrow(Rmat)
Fstat<-(1/J)* t(Rmat %*% b-q) %*% solve(Rmat %*% Vb %*% t(Rmat)) %*% (Rmat%*%b-q)
Fstat
pval<-1-pf(Fstat,J,n-k)
pval
