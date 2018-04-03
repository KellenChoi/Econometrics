### Linear Regression in Python
##  Simple Linear Regression

# How do we measure institutional differences and economic outcomes?

import pandas as pd
df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')
df1.head()

# scatterplot to see whether any obvious relationship exists between GDP per capita and the protection against expropriation index
# higher protection against expropriation higher GDP per capita
import matplotlib.pyplot as plt
plt.style.use('seaborn')
df1.plot(x='avexpr', y='logpgp95', kind='scatter')
plt.show()

import numpy as np
# Dropping NA's is required to use numpy's polyfit
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])
df1_subset.head()

# Use only 'base sample' for plotting purposes
df1_subset = df1_subset[df1_subset['baseco'] == 1]

X = df1_subset['avexpr']
y = df1_subset['logpgp95']
labels = df1_subset['shortnam']

# Replace markers with country labels
plt.scatter(X, y, marker='')

for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], y.iloc[i]))

# Fit a linear trend line
plt.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

plt.xlim([3.3,10.5])
plt.ylim([4,10.5])
plt.xlabel('Average Expropriation Risk 1985-95')
plt.ylabel('Log GDP per capita, PPP, 1995')
plt.title('Figure 2: OLS relationship between expropriation risk and income')
plt.show()

# To estimate the constant term β0,
# we need to add a column of 1’s to our dataset (consider the equation if β0 was replaced with β0xi and xi=1)

df1['const'] = 1

# construct our model in statsmodels using the OLS function
import statsmodels.api as sm

reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], missing='drop')
type(reg1)

# obtain parameter estimates β̂ 0 and β̂ 1
results = reg1.fit()   # store fitted regression model in 'results'
type(results)

results.summary()

# predict the level of log GDP per capita
# ex) an index value of 7.07 (the average of the dataset)
mean_expr = np.mean(df1_subset['avexpr'])
mean_expr

predicted_logpdp95 = 4.63 + 0.53 * 7.07
predicted_logpdp95

# Alternatively an easier (and more accurate) way to obtain this result 
results.predict(exog=[1, mean_expr])

# Drop missing observations from whole sample
df1_plot = df1.dropna(subset=['logpgp95', 'avexpr'])
df1_plot.head()

# Plot predicted values
plt.scatter(df1_plot['avexpr'], results.predict(), alpha=0.5, label='predicted')

# Plot observed values
plt.scatter(df1_plot['avexpr'], df1_plot['logpgp95'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('avexpr')
plt.ylabel('logpgp95')
plt.show()

### Multivariate regression model
df2 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable2.dta')
df2.head()
# add constant term to the dataset: To estimate the constant term β0,
df2['const'] = 1
df2.head()
# summary
reg2.summary()

# display the results in a single table: summary_col

from statsmodels.iolib.summary2 import summary_col

info_dict = {'R_squared' : lambda x: "{:.2f}".format(x.rsquared),
            'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_table = summary_col(results=[reg1,reg2,reg3],
                           float_format='%0.2f',
                           stars = True,
                           model_names=['Model 1',
                                        'Model 3',
                                        'Model 4'],
                           info_dict=info_dict,
                           regressor_order=['const',
                                            'avexpr',
                                            'lat_abst',
                                            'asia',
                                            'africa'])

results_table.add_title('Table 2 - OLS Regressions')

print(results_table)

## Two-stage least squares(2SLS)regression: for endogeneity issues(biased and inconsistent estimates

# Dropping NA's is required to use numpy's polyfit
df1_subset2 = df1.dropna(subset=['logem4', 'avexpr'])
df1_subset2.head()
X = df1_subset2['logem4']
y = df1_subset2['avexpr']
labels = df1_subset2['shortnam']

# Replace markers with country labels
plt.scatter(X, y, marker='')
for i, label in enumerate(labels):
    plt.annotate(label, (X.iloc[i], y.iloc[i]))
    
# Fit a linear trend line
plt.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

plt.xlim([1.8,8.4])
plt.ylim([3.3,10.4])
plt.xlabel('Log of Settler Mortality')
plt.ylabel('Average Expropriation Risk 1985-95')
plt.title('Figure 3: First-stage relationship between settler mortality and expropriation risk')
plt.show()

### 2SLS Regression 
## First Stage
# Import and select the data
df4 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable4.dta')
df4 = df4[df4['baseco'] == 1]
df4.head()

# add a constant variable
df4['const'] =1 

results_fs = sm.OLS(df4['avexpr'], df4[['const', 'logem4']],
                   missing='drop').fit()
print(results_fs.summary())

## second stage --> give unbiased and consistent estimates
# retrieve the predicted values of avexpri using .predict()
df4['predicted_avexpr'] = results_fs.predict()

results_ss = sm.OLS(df4['logpgp95'],
                    df4[['const', 'predicted_avexpr']]).fit()
print(results_ss.summary())

### 2SLS Regression by IV2SLS
from linearmodels.iv import IV2SLS

iv = IV2SLS(dependent=df4['logpgp95'],
            exog=df4['const'],
            endog=df4['avexpr'],
            instruments=df4['logem4']).fit(cov_type='unadjusted')
print(iv.summary)







