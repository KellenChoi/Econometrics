########## Covergence in Probability#############
##### Approximating a single random variable #####

#### 1. Normal
# Generate normals - > samples
# samples of sizes: 10, 100, 1000
mu<-3 #true population mean
sig<-2 #population std
R<-500 #number of repeated draws
m10<-rep(0,R) #will collect means for samples of size 10
m100<-rep(0,R) #will collect means for samples of size 100
m1000<-rep(0,R) #will collect means for samples of size 1000

for (i in 1:R){
  m10[i]<-mean(rnorm(10,mu,sig))
  m100[i]<-mean(rnorm(100,mu,sig))
  m1000[i]<-mean(rnorm(1000,mu,sig))
}

#Densities
d10<-density(m10, kernel="epanechnikov")
d100<-density(m100, kernel="epanechnikov")
d1000<-density(m1000, kernel="epanechnikov")

#Graphics
plot(d10,type="l",main = "",xlab = "Sample mean",ylab = "density", ylim=c(0, 6))
lines(d100, col=2)
lines(d1000, col=3)
labels<-c("n=10","n=100","n=1000")
legend("topright", inset=.05,
       labels, lwd=1, lty=c(2,3,1), col=c(1,2,3))
######## as the higher the sample size, the higher the accumulation around 3, approaching the mean


#### 2. T-distribution with n-1 degree of freedom 
Re<-500 #number of repeated draws
mu<-0
sig<-1
t10<-rep(0,Re)  #will collect t-draws with DoF 10-1
t100<-rep(0,Re) #will collect t-draws with DoF 100-1
t1000<-rep(0,Re) #will collect t-draws with DoF 1000-1
for (i in 1:Re){
        int10<-rnorm(10,mu,sig)
        t10[i]<-mean(int10)/(sd(int10)/sqrt(10))
        int100<-rnorm(100,mu,sig)
        t100[i]<-mean(int100)/(sd(int100)/sqrt(100))
        int1000<-rnorm(1000,mu,sig)
        t1000[i]<-mean(int1000)/(sd(int1000)/sqrt(1000))
}


#### 3. Standard normal for comparison
int<-rnorm(Re,mu,sig)
de10<-density(t10,kernel="epanechnikov")
de100<-density(t100,kernel="epanechnikov")
de1000<-density(t1000,kernel="epanechnikov")
dnorm<-density(int,kernel="epanechnikov", n=10000)
plot(de10,type="l",main = "",xlab = "t",ylab = "density",
     xlim=c(-5,5),ylim=c(0,0.5))
lines(de100,col=2)
lines(de1000,col=3)
lines(dnorm,col=4)
labels<-c("n=10","n=100","n=1000", "n(0,1)")
legend("topright", inset=.05,
       labels, col=c(1,2,3,4), lwd=1)


########### ggplot ##############################
require(reshape2)
require(ggplot2)
dat<-data.frame(m10, m100,m1000)
#we need to "melt" the data for this to work, to see what I mean, open the "melted" dataset
melted<-melt(dat)
ggplot(melted) + geom_density(aes(x = value,
                                  colour = variable)) 




