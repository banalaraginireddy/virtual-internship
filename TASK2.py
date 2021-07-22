#!/usr/bin/env python
# coding: utf-8

# In[145]:


import pandas as pd


# In[146]:


#loading the dataset
df=pd.read_csv("/Users/raginibanala/Downloads/QVI_data.csv")


# In[147]:


df.head()


# In[148]:


df.dtypes


# In[149]:


#convert date type to date

df['DATE']=pd.to_datetime(df['DATE'])


# In[150]:


#creating yearmonth column
df["YEARMONTH"] = df["DATE"].dt.strftime("%Y%m").astype("int")


# In[151]:


df.head(2)


# We would want to match trial stores to control stores that are similar to the trial
# store prior to the trial period of Feb 2019 in terms of :
# - Monthly overall sales revenue
# - Monthly number of customers
# - Monthly number of transactions per customer

# For each store and month calculate total sales, number of customers,
# transactions per customer, chips per customer and the average price per unit.

# In[152]:


#monthly sales
month_sales=df.groupby(["STORE_NBR","YEARMONTH"])["TOT_SALES"].sum()

#monthly customers
month_cust= df.groupby(["STORE_NBR","YEARMONTH"])["LYLTY_CARD_NBR"].nunique()

#monthly transactions/cust
month_tnx_percust=df.groupby(["STORE_NBR","YEARMONTH"])["TXN_ID"].nunique()/month_cust

#monthly chips/cust
month_chipscust=df.groupby(["STORE_NBR","YEARMONTH"])["PROD_QTY"].sum()/month_cust

#monthly avg price/unit

month_avgprice=month_sales/df.groupby(["STORE_NBR","YEARMONTH"])["PROD_QTY"].sum()


# In[153]:


#combining the above ones to new dataframe
new_data=pd.concat([month_sales,month_cust,month_tnx_percust,month_chipscust,month_avgprice],axis=1)
new_data.columns=["totsales","customers","txnpercust","chipspertxn","avgprieperunnit"]
new_data=new_data.reset_index()
new_data


# In[154]:


# stores with full observation periods
store_counts=new_data["STORE_NBR"].value_counts()
x=store_counts[store_counts==12].index
stores_with_fullobs = new_data[new_data["STORE_NBR"].isin(x)]


# In[155]:


#Filter to the pre-trial period
pretrial_period = stores_with_fullobs[stores_with_fullobs["YEARMONTH"] < 201902]


# Now we need to work out a way of ranking how similar each potential control store
# is to the trial store. We can calculate how correlated the performance of each
# store is to the trial store.

# In[156]:



def calculateCorrelation(inputTable, metricCol, storeComparison):
    # stores the data of control stores
    control_stores= inputTable.loc[~inputTable["STORE_NBR"].isin([77,86,88])]
    #stores the trial stores data
    trial_store_data = inputTable.loc[inputTable["STORE_NBR"] == storeComparison][metricCol].reset_index()
    #empty dataframe
    calcCorrTable = pd.DataFrame(columns = ["Control_Store", "Trial_Store", "Corr"])
    
    for i in control_stores["STORE_NBR"].unique():

        control = control_stores[control_stores["STORE_NBR"]==i][metricCol].reset_index()              
        correlation = control.corrwith(trial_store_data,axis=0)[1]
        calcCorrTable_i = pd.DataFrame({"Control_Store":i,"Trial_Store":storeComparison,"Corr":[correlation]}) 
        calcCorrTable = pd.concat([calcCorrTable, calcCorrTable_i])
     
    return calcCorrTable


# Apart from correlation, we can also calculate a standardised metric based on the
# absolute difference between the trial store's performance and each control store's
# performance.

# In[157]:


import numpy as np
def calculateMagnitudeDistance(inputTable, metricCol, storeComparison):
    control_stores = inputTable.loc[~inputTable["STORE_NBR"].isin([77,86,88])]
    trial_store_data = inputTable.loc[inputTable["STORE_NBR"] == storeComparison].reset_index()[metricCol]
    calcDistTable = pd.DataFrame(columns=["Control_Store", "Trial_Store", "Magnitude"])
    
    for i in control_stores["STORE_NBR"].unique():
        control = control_stores[control_stores["STORE_NBR"]==i].reset_index()[metricCol]
        diff = abs(trial_store_data - control)
        s_diff = np.mean(1-((diff-min(diff))/(max(diff)-min(diff))))
        calcDistTable_i = pd.DataFrame({"Control_Store":i,"Trial_Store":[storeComparison],"Magnitude": s_diff})
        calcDistTable = pd.concat([calcDistTable, calcDistTable_i])
    return calcDistTable


# Now let's use the functions to find the control stores! We'll select control stores
# based on how similar monthly total sales in dollar amounts and monthly number of
# customers are to the trial stores. So we will need to use our functions to get four
# scores, two for each of total sales and total customers.

# Use the function you created to calculate correlations against
# store 77 using total sales and number of customers.

# In[158]:


#correlation and magnitude for sales column
corr_salesfor77 = calculateCorrelation(pretrial_period,"totsales",77)

magn_sales77=calculateMagnitudeDistance(pretrial_period,"totsales",77) 


# In[159]:


#correlation and magnitude for customer column
corr_custfor77 = calculateCorrelation(pretrial_period,"customers",77)

magn_cust77=calculateMagnitudeDistance(pretrial_period,"customers",77) 


# We'll need to combine the all the scores calculated using our function to create a
# composite score to rank on.
# Let's take a simple average of the correlation and magnitude scores for each
# driver. Note that if we consider it more important for the trend of the drivers to
# be similar, we can increase the weight of the correlation score (a simple average
# gives a weight of 0.5 to the corr_weight) or if we consider the absolute size of
# the drivers to be more important, we can lower the weight of the correlation score.

# In[160]:


#Create a combined score composed of correlation and magnitude, by
#first merging the correlations table with the magnitude table.
# for sales

sales77merge = pd.concat([corr_salesfor77,magn_sales77["Magnitude"]],axis=1)

x= 0.5
sales77merge["scoresales"] = x *sales77merge["Corr"] + (1-x) * sales77merge["Magnitude"]
sales77merge


# In[161]:


#repeat same for customer

cust77merge = pd.concat([corr_custfor77,magn_cust77["Magnitude"]],axis=1)

x = 0.5
cust77merge["scorecustomers"] = x * cust77merge["Corr"] + (1-x) * cust77merge["Magnitude"]
cust77merge


# Now we have a score for each of total number of sales and number of customers.
# Let's combine the two via a simple average.
# 
# Combine scores across the drivers by first merging our sales
# scores and customer scores into a single table

# In[162]:


#merging the tables
finalmerge77 = pd.concat([sales77merge[["Control_Store", "Trial_Store", "scoresales"]],cust77merge["scorecustomers"]],axis=1)


# In[163]:


#final score
x=0.5
finalmerge77["finalscore"]=x*finalmerge77["scoresales"]+(1-x)*finalmerge77["scorecustomers"]
finalmerge77


# The store with the highest score is then selected as the control store since it is
# most similar to the trial store.

# In[164]:


finalmerge77.sort_values(by="finalscore",ascending=False).head(1)


# In[165]:


# 233 is the highest
#Now that we have found a control store, let's check visually if the drivers are
#indeed similar in the period before the trial.
#We'll look at total sales first.


# # visulaizing

# In[166]:


a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["totsales"].unstack()


# In[167]:


a


# In[168]:


othercol = [i for i in a.columns if i not in [77, 233]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b77 = a.loc[:,([77,233,"others"])].reset_index()
b77["YEARMONTH"]= pd.to_datetime(b77["YEARMONTH"], format="%Y%m")
b77 = b77.set_index(["YEARMONTH"])
b77.columns=["Trial_77","Control_233","Others"]


# In[169]:


b77


# In[170]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[171]:


sns.lineplot(data=b77)


# In[172]:


#repeat same thing for customers
a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["customers"].unstack()


# In[173]:


a


# In[174]:


othercol = [i for i in a.columns if i not in [77, 233]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b77 = a.loc[:,([77,233,"others"])].reset_index()
b77["YEARMONTH"]= pd.to_datetime(b77["YEARMONTH"], format="%Y%m")
b77 = b77.set_index(["YEARMONTH"])
b77.columns=["Trial_77","Control_233","Others"]


# In[175]:


b77


# In[176]:


sns.lineplot(data=b77)


# # looks like 233 is a good match for 77

# The trial period goes from the start of February 2019 to April 2019. We now want to
# see if there has been an uplift in overall chip sales.
# We'll start with scaling the control store's sales to a level similar to control
# for any differences between the two stores outside of the trial period. 

# In[177]:


pretrial_period.head(1)


# ## Assessment of trial

# The trial period goes from the start of February 2019 to April 2019. We now want to
# see if there has been an uplift in overall chip sales.
# We'll start with scaling the control store's sales to a level similar to control
# for any differences between the two stores outside of the trial period.
# 
# #### Scale pre-trial control sales to match pre-trial trial store sales 

# In[178]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 77]["totsales"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 233]["totsales"].sum())
y


# In[180]:


scaledcntrlsales = new_data[new_data["STORE_NBR"]== 233]
scaledcntrlsales["controlSales"] = scaledcntrlsales["totsales"] * y
scaledcntrlsales = scaledcntrlsales.reset_index(drop=True)
scaledcntrlsales


# Now that we have comparable sales figures for the control store, we can calculate
# the percentage difference between the scaled control sales and the trial store's
# sales during the trial period.

# Calculate the percentage difference between scaled control sales
# and trial sales

# In[183]:


l=new_data[new_data["STORE_NBR"]==77]
l=l.reset_index(drop=True)
l


# In[186]:


percentdiff = pd.concat([l["YEARMONTH"],l["totsales"],scaledcntrlsales["controlSales"]],axis=1)
percentdiff.columns=["YEARMONTH","trialSales","controlSales"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialSales"]-percentdiff["controlSales"]))/percentdiff["controlSales"]
percentdiff


# In[187]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd


# In[188]:


#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[191]:


from scipy.stats import t
t.ppf(0.95,7)


# We can observe that the t-value is much larger than the 95th percentile value of
# the t-distribution for March and April - i.e. the increase in sales in the trial
# store in March and April is statistically greater than in the control store.

# Let's create a more visual version of this by plotting the sales of the control
# store, the sales of the trial stores and the 95th percentile value of sales of the
# control store.

# In[195]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 77:
        store.append("Trial")
    elif i == 233:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[208]:



d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','totsales']]


# In[209]:


y


# In[219]:


#95th percentile

d_95_77 = d[d["Store_type"] == "Control"]
d_95_77["totsales"] = d_95_77["totsales"] * (1+(sd*2))
d_95_77_1 = d_95_77[['TransactionMonth', 'Store_type','totsales']]
d_95_77_1["Store_type"]='95th percentile cntrl'
d_95_77_1


# In[220]:


#5th percentile
d_5_77 = d[d["Store_type"] == "Control"]
d_5_77["totsales"] = d_95_77_1["totsales"] * (1-(sd*2))
d_5_77_1 = d_5_77[["TransactionMonth", "Store_type","totsales"]]
d_5_77_1['Store_type']='5th percentile cntrl'
d_5_77_1


# In[288]:


#merging all 3
tot_77 = pd.concat([y, d_95_77_1, d_5_77_1])
tot_77.head()


# In[272]:



y


# In[279]:


l=tot_77
l["TransactionMonth"] = l["TransactionMonth"].dt.strftime("%Y-%m").astype("str")


# In[280]:


m = l.set_index(["TransactionMonth","Store_type"])["totsales"].unstack()


# In[281]:


z=m[['5th percentile cntrl','95th percentile cntrl']]


# In[282]:


z


# In[287]:


#plotting everything into one nice graph

fig, ax1 = plt.subplots(1, 1, figsize=(15,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["totsales"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# The results show that the trial in store 77 is significantly different to its
# control store in the trial period as the trial store performance lies outside the
# 5% to 95% confidence interval of the control store in two of the three trial
# months.

# Following the same procedure for customers instead of sales

# In[292]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 77]["customers"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 233]["customers"].sum())
y


# In[293]:


scaledcntrlcust = new_data[new_data["STORE_NBR"]== 233]
scaledcntrlcust["controlcustomer"] = scaledcntrlcust["customers"] * y
scaledcntrlcust = scaledcntrlcust.reset_index(drop=True)
scaledcntrlcust


# In[294]:


l=new_data[new_data["STORE_NBR"]==77]
l=l.reset_index(drop=True)
l


# In[298]:


percentdiff = pd.concat([l["YEARMONTH"],l["customers"],scaledcntrlcust["controlcustomer"]],axis=1)
percentdiff.columns=["YEARMONTH","trialcust","controlcustomer"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialcust"]-percentdiff["controlcustomer"]))/percentdiff["controlcustomer"]
percentdiff


# In[299]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd


# In[300]:


#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[301]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 77:
        store.append("Trial")
    elif i == 233:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[302]:


d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','customers']]


# In[304]:


y


# In[305]:


d_95_77 = d[d["Store_type"] == "Control"]
d_95_77["customers"] = d_95_77["customers"] * (1+(sd*2))
d_95_77_1 = d_95_77[['TransactionMonth', 'Store_type','customers']]
d_95_77_1["Store_type"]='95th percentile cntrl'
d_95_77_1


# In[306]:


d_5_77 = d[d["Store_type"] == "Control"]
d_5_77["customers"] = d_95_77_1["customers"] * (1-(sd*2))
d_5_77_1 = d_5_77[["TransactionMonth", "Store_type","customers"]]
d_5_77_1['Store_type']='5th percentile cntrl'
d_5_77_1


# In[307]:


tot_77 = pd.concat([y, d_95_77_1, d_5_77_1])
tot_77


# In[311]:


l=tot_77
l


# In[312]:


m = l.set_index(["TransactionMonth","Store_type"])["customers"].unstack()

z=m[['5th percentile cntrl','95th percentile cntrl']]


# In[313]:


z


# In[315]:



y["TransactionMonth"] = y["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
y = y.set_index("TransactionMonth")


# In[316]:


fig, ax1 = plt.subplots(1, 1, figsize=(15,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["customers"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# # store 86 

# In[317]:


#correlation and magnitude for sales column
corr_salesfor86 = calculateCorrelation(pretrial_period,"totsales",86)

magn_sales86=calculateMagnitudeDistance(pretrial_period,"totsales",86) 

#correlation and magnitude for customer column
corr_custfor86 = calculateCorrelation(pretrial_period,"customers",86)

magn_cust86=calculateMagnitudeDistance(pretrial_period,"customers",86) 


# In[318]:


#Create a combined score composed of correlation and magnitude, by
#first merging the correlations table with the magnitude table.
# for sales

sales86merge = pd.concat([corr_salesfor86,magn_sales86["Magnitude"]],axis=1)

x= 0.5
sales86merge["scoresales"] = x *sales86merge["Corr"] + (1-x) * sales86merge["Magnitude"]
sales86merge


# In[319]:


#repeat same for customer

cust86merge = pd.concat([corr_custfor86,magn_cust86["Magnitude"]],axis=1)

x = 0.5
cust86merge["scorecustomers"] = x * cust86merge["Corr"] + (1-x) * cust86merge["Magnitude"]
cust86merge


# In[320]:


#merging the tables
finalmerge86 = pd.concat([sales86merge[["Control_Store", "Trial_Store", "scoresales"]],cust86merge["scorecustomers"]],axis=1)
#final score
x=0.5
finalmerge86["finalscore"]=x*finalmerge86["scoresales"]+(1-x)*finalmerge86["scorecustomers"]
finalmerge86
finalmerge86.sort_values(by="finalscore",ascending=False).head(1)


# In[321]:


# 155 is the highest
#Now that we have found a control store, let's check visually if the drivers are
#indeed similar in the period before the trial.
#We'll look at total sales first.


# In[322]:


a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["totsales"].unstack()
othercol = [i for i in a.columns if i not in [86, 155]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b86 = a.loc[:,([86,155,"others"])].reset_index()
b86["YEARMONTH"]= pd.to_datetime(b86["YEARMONTH"], format="%Y%m")
b86 = b86.set_index(["YEARMONTH"])
b86.columns=["Trial_86","Control_155","Others"]
b86


# In[323]:


sns.lineplot(data=b86)


# In[324]:


#repeat same thing for customers
a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["customers"].unstack()
othercol = [i for i in a.columns if i not in [86, 155]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b86 = a.loc[:,([86,155,"others"])].reset_index()
b86["YEARMONTH"]= pd.to_datetime(b86["YEARMONTH"], format="%Y%m")
b86 = b86.set_index(["YEARMONTH"])
b86.columns=["Trial_86","Control_155","Others"]
b86


# In[325]:


sns.lineplot(data=b86)


# ### looks like  is 155 is good match for 86

# In[326]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 86]["totsales"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 155]["totsales"].sum())
y


# In[327]:


scaledcntrlsales = new_data[new_data["STORE_NBR"]== 155]
scaledcntrlsales["controlSales"] = scaledcntrlsales["totsales"] * y
scaledcntrlsales = scaledcntrlsales.reset_index(drop=True)
scaledcntrlsales


# In[328]:


l=new_data[new_data["STORE_NBR"]==86]
l=l.reset_index(drop=True)
l


# In[329]:


percentdiff = pd.concat([l["YEARMONTH"],l["totsales"],scaledcntrlsales["controlSales"]],axis=1)
percentdiff.columns=["YEARMONTH","trialSales","controlSales"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialSales"]-percentdiff["controlSales"]))/percentdiff["controlSales"]
percentdiff


# In[330]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd
#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[331]:


from scipy.stats import t
t.ppf(0.95,7)


# In[332]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 86:
        store.append("Trial")
    elif i == 155:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[333]:



d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','totsales']]


# In[334]:


#95th percentile

d_95_86 = d[d["Store_type"] == "Control"]
d_95_86["totsales"] = d_95_86["totsales"] * (1+(sd*2))
d_95_86_1 = d_95_86[['TransactionMonth', 'Store_type','totsales']]
d_95_86_1["Store_type"]='95th percentile cntrl'
d_95_86_1


# In[335]:


#5th percentile
d_5_86 = d[d["Store_type"] == "Control"]
d_5_86["totsales"] = d_95_86_1["totsales"] * (1-(sd*2))
d_5_86_1 = d_5_86[["TransactionMonth", "Store_type","totsales"]]
d_5_86_1['Store_type']='5th percentile cntrl'
d_5_86_1


# In[336]:


#merging all 3
tot_86_1 = pd.concat([ d_95_86_1, d_5_86_1])
tot_86_1.head()


# In[337]:


y["TransactionMonth"] = y["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
y= y.set_index("TransactionMonth")
l=tot_86_1
l["TransactionMonth"] = l["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
m = l.set_index(["TransactionMonth","Store_type"])["totsales"].unstack()
z=m[['5th percentile cntrl','95th percentile cntrl']]


# In[339]:


fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["totsales"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# The results show that the trial in store 86 is not that significantly different to its
# control store in the trial period as the trial store performance lies inside the
# 5% to 95% confidence interval of the control store in two of the three trial
# months.

# In[340]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 86]["customers"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 155]["customers"].sum())
y


# In[341]:


scaledcntrlcust = new_data[new_data["STORE_NBR"]== 155]
scaledcntrlcust["controlcustomer"] = scaledcntrlcust["customers"] * y
scaledcntrlcust = scaledcntrlcust.reset_index(drop=True)
scaledcntrlcust


# In[342]:


l=new_data[new_data["STORE_NBR"]==86]
l=l.reset_index(drop=True)
l


# In[343]:


percentdiff = pd.concat([l["YEARMONTH"],l["customers"],scaledcntrlcust["controlcustomer"]],axis=1)
percentdiff.columns=["YEARMONTH","trialcust","controlcustomer"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialcust"]-percentdiff["controlcustomer"]))/percentdiff["controlcustomer"]
percentdiff


# In[344]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd


# In[345]:


#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[346]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 86:
        store.append("Trial")
    elif i == 155:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[347]:


d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','customers']]


# In[348]:


d_95_86 = d[d["Store_type"] == "Control"]
d_95_86["customers"] = d_95_86["customers"] * (1+(sd*2))
d_95_86_1 = d_95_86[['TransactionMonth', 'Store_type','customers']]
d_95_86_1["Store_type"]='95th percentile cntrl'
d_95_86_1


# In[349]:


d_5_86 = d[d["Store_type"] == "Control"]
d_5_86["customers"] = d_95_86_1["customers"] * (1-(sd*2))
d_5_86_1 = d_5_86[["TransactionMonth", "Store_type","customers"]]
d_5_86_1['Store_type']='5th percentile cntrl'
d_5_86_1


# In[350]:


tot_86 = pd.concat([ d_95_86_1, d_5_86_1])
tot_86


# In[351]:


l=tot_86
l


# In[352]:


l["TransactionMonth"] = l["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
k= l.set_index(["TransactionMonth","Store_type"])["customers"].unstack()

z=k[['5th percentile cntrl','95th percentile cntrl']]


# In[353]:


y["TransactionMonth"] = y["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
y = y.set_index("TransactionMonth")


# In[355]:


fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["customers"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# # store 88

# In[356]:


#correlation and magnitude for sales column
corr_salesfor88 = calculateCorrelation(pretrial_period,"totsales",88)

magn_sales88=calculateMagnitudeDistance(pretrial_period,"totsales",88) 

#correlation and magnitude for customer column
corr_custfor88 = calculateCorrelation(pretrial_period,"customers",88)

magn_cust88=calculateMagnitudeDistance(pretrial_period,"customers",88) 


# In[357]:


#Create a combined score composed of correlation and magnitude, by
#first merging the correlations table with the magnitude table.
# for sales

sales88merge = pd.concat([corr_salesfor88,magn_sales88["Magnitude"]],axis=1)

x= 0.5
sales88merge["scoresales"] = x *sales88merge["Corr"] + (1-x) * sales88merge["Magnitude"]
sales88merge


# In[358]:


#repeat same for customer

cust88merge = pd.concat([corr_custfor88,magn_cust88["Magnitude"]],axis=1)

x = 0.5
cust88merge["scorecustomers"] = x * cust88merge["Corr"] + (1-x) * cust88merge["Magnitude"]
cust88merge


# In[359]:


#merging the tables
finalmerge88 = pd.concat([sales88merge[["Control_Store", "Trial_Store", "scoresales"]],cust88merge["scorecustomers"]],axis=1)


# In[364]:


#final score
x=0.5
finalmerge88["finalscore"]=x*finalmerge88["scoresales"]+(1-x)*finalmerge88["scorecustomers"]
finalmerge88.sort_values(by="finalscore",ascending=False).head()


# In[361]:


# 178 is the highest
#Now that we have found a control store, let's check visually if the drivers are
#indeed similar in the period before the trial.
#We'll look at total sales first.


# In[362]:


a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["totsales"].unstack()


# In[363]:


othercol = [i for i in a.columns if i not in [88, 237]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b88 = a.loc[:,([88,237,"others"])].reset_index()
b88["YEARMONTH"]= pd.to_datetime(b88["YEARMONTH"], format="%Y%m")
b88 = b88.set_index(["YEARMONTH"])
b88.columns=["Trial_88","Control_178","Others"]


# In[365]:


sns.lineplot(data=b88)


# In[366]:


#repeat same thing for customers
a=pretrial_period.set_index(["YEARMONTH","STORE_NBR"])["customers"].unstack()
othercol = [i for i in a.columns if i not in [88, 237]]
a["others"]=a.loc[:,othercol].mean(axis=1)
b88 = a.loc[:,([88,237,"others"])].reset_index()
b88["YEARMONTH"]= pd.to_datetime(b88["YEARMONTH"], format="%Y%m")
b88 = b88.set_index(["YEARMONTH"])
b88.columns=["Trial_88","Control_237","Others"]
sns.lineplot(data=b88)


# ### looks like  is 237 is  good match for 88

# In[367]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 88]["totsales"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 237]["totsales"].sum())
y


# In[368]:


scaledcntrlsales = new_data[new_data["STORE_NBR"]== 237]
scaledcntrlsales["controlSales"] = scaledcntrlsales["totsales"] * y
scaledcntrlsales = scaledcntrlsales.reset_index(drop=True)
scaledcntrlsales


# In[369]:


l=new_data[new_data["STORE_NBR"]==88]
l=l.reset_index(drop=True)
percentdiff = pd.concat([l["YEARMONTH"],l["totsales"],scaledcntrlsales["controlSales"]],axis=1)
percentdiff.columns=["YEARMONTH","trialSales","controlSales"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialSales"]-percentdiff["controlSales"]))/percentdiff["controlSales"]
percentdiff


# In[370]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd


# In[371]:


#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[372]:


from scipy.stats import t
t.ppf(0.95,7)


# In[373]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 88:
        store.append("Trial")
    elif i == 237:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[374]:


d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','totsales']]


# In[375]:


#95th percentile

d_95_88 = d[d["Store_type"] == "Control"]
d_95_88["totsales"] = d_95_88["totsales"] * (1+(sd*2))
d_95_88_1 = d_95_88[['TransactionMonth', 'Store_type','totsales']]
d_95_88_1["Store_type"]='95th percentile cntrl'
d_95_88_1


# In[376]:


#5th percentile
d_5_88 = d[d["Store_type"] == "Control"]
d_5_88["totsales"] = d_95_88_1["totsales"] * (1-(sd*2))
d_5_88_1 = d_5_88[["TransactionMonth", "Store_type","totsales"]]
d_5_88_1['Store_type']='5th percentile cntrl'
d_5_88_1


# In[377]:


#merging all 3
tot_88_1 = pd.concat([ d_95_88_1, d_5_88_1])
tot_88_1.head()


# In[378]:


y["TransactionMonth"] = y["TransactionMonth"].dt.strftime("%Y-%m").astype("str")


# In[379]:


y= y.set_index("TransactionMonth")
l=tot_88_1
l["TransactionMonth"] = l["TransactionMonth"].dt.strftime("%Y-%m").astype("str")

m = l.set_index(["TransactionMonth","Store_type"])["totsales"].unstack()

z=m[['5th percentile cntrl','95th percentile cntrl']]


# In[381]:


#plotting everything into one nice graph

fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["totsales"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# In[382]:


y = (pretrial_period[pretrial_period["STORE_NBR"] == 88]["customers"].sum()) / (pretrial_period[pretrial_period["STORE_NBR"] == 237]["customers"].sum())
y


# In[383]:


scaledcntrlcust = new_data[new_data["STORE_NBR"]== 237]
scaledcntrlcust["controlcustomer"] = scaledcntrlcust["customers"] * y
scaledcntrlcust = scaledcntrlcust.reset_index(drop=True)
scaledcntrlcust


# In[384]:


l=new_data[new_data["STORE_NBR"]==88]
l=l.reset_index(drop=True)
percentdiff = pd.concat([l["YEARMONTH"],l["customers"],scaledcntrlcust["controlcustomer"]],axis=1)
percentdiff.columns=["YEARMONTH","trialcust","controlcustomer"]
percentdiff["percentageDiff"]= (abs(percentdiff["trialcust"]-percentdiff["controlcustomer"]))/percentdiff["controlcustomer"]
percentdiff


# In[385]:


#calculating standard deviation
sd = percentdiff[percentdiff["YEARMONTH"]<201902]["percentageDiff"].std()
sd


# In[386]:


#calculating t-values for trial months
percentdiff["t-value"] = (percentdiff["percentageDiff"]-0)/sd
percentdiff


# In[387]:


#creating column storetype
store=[]
d=new_data
for i in d["STORE_NBR"]:
    if i == 88:
        store.append("Trial")
    elif i == 237:
        store.append("Control")
    else:
        store.append("Other Stores")
d["Store_type"] = store
d.head()


# In[388]:


d["TransactionMonth"] = pd.to_datetime(d["YEARMONTH"], format = "%Y%m")

d= d.loc[d["Store_type"].isin(["Control","Trial"])]
y=d[['TransactionMonth','Store_type','customers']]


# In[389]:


d_95_88 = d[d["Store_type"] == "Control"]
d_95_88["customers"] = d_95_88["customers"] * (1+(sd*2))
d_95_88_1 = d_95_88[['TransactionMonth', 'Store_type','customers']]
d_95_88_1["Store_type"]='95th percentile cntrl'
d_95_88_1


# In[390]:


d_5_88 = d[d["Store_type"] == "Control"]
d_5_88["customers"] = d_95_88_1["customers"] * (1-(sd*2))
d_5_88_1 = d_5_88[["TransactionMonth", "Store_type","customers"]]
d_5_88_1['Store_type']='5th percentile cntrl'
d_5_88_1


# In[391]:


tot_88 = pd.concat([ d_95_88_1, d_5_88_1])
tot_88


# In[392]:


l=tot_88
l["TransactionMonth"] = l["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
k= l.set_index(["TransactionMonth","Store_type"])["customers"].unstack()

z=k[['5th percentile cntrl','95th percentile cntrl']]


# In[393]:



y["TransactionMonth"] = y["TransactionMonth"].dt.strftime("%Y-%m").astype("str")
y = y.set_index("TransactionMonth")


# In[394]:


fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
ax2 = ax1.twinx()
ax1 = sns.barplot(x=y.index, y=y["customers"], hue=y["Store_type"], data=y,)
ax2=sns.lineplot(data=z,linewidth=5)


# ## conclusion

# Good work! We've found control stores 233, 155, 237 for trial stores 77, 86 and 88
# respectively.
# The results for trial stores 77 and 88 during the trial period show a significant
# difference in at least two of the three trial months but this is not the case for
# trial store 86. We can check with the client if the implementation of the trial was
# different in trial store 86 but overall, the trial shows a significant increase in
# sales. Now that we have finished our analysis, we can prepare our presentation to
# the Category Manager.

# In[ ]:




