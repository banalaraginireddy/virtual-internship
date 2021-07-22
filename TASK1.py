#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


transactionData=pd.read_excel('/Users/raginibanala/Downloads/QVI_transaction_data.xlsx')


# In[5]:


customerData=pd.read_csv('/Users/raginibanala/Downloads/QVI_purchase_behaviour.csv')


# In[6]:


transactionData.info()
# transactiondata.dtypes


# In[7]:


customerData.info()


# In[8]:


from datetime import date,timedelta


# In[9]:


#converting date format from integer to datetime
from datetime import date, timedelta
start = date(1899,12,30)

new_date_format = []

for date in transactionData["DATE"]:
    delta = timedelta(date)
    new_date_format.append(start + delta)


# In[10]:


transactionData["DATE"] = pd.to_datetime(pd.Series(new_date_format))


# In[11]:


transactionData["DATE"].head()


# In[12]:


# summary of prod_name column

transactionData.describe(include='object')


# In[13]:


df=transactionData["PROD_NAME"].unique()
type(df)
import numpy as np
df1=pd.DataFrame(df)
print(df1)


# In[14]:


productWords=df1[0].str.split(" ")


# In[15]:


productWords=transactionData["PROD_NAME"].str.split(" ")


# In[16]:


type(productWords[0])
productWords


# In[17]:


df=[]
for i in range(productWords.shape[0]):
    for j in productWords[i]:
        df.append(j)


# In[18]:


val=''
l = list(filter(lambda x: x != val, df))


# In[19]:


import re
regex = re.compile(r'^[0-9]+')
filtered = [i for i in l if not regex.match(i)]


# In[20]:


import re
regex = re.compile(r'^&')
filtered1 = [i for i in filtered if not regex.match(i)]


# In[21]:


import collections
print(collections.Counter(filtered1))


# In[22]:


# dropping salsa items


# In[23]:


transactionData.drop(transactionData[[("Salsa" in s)  for s in transactionData['PROD_NAME']]].index,inplace=True)


# In[24]:


# checking summary


# In[25]:


transactionData.describe()


# In[26]:


#checking nulls in all columns
transactionData.isna().sum()


# In[27]:


# there are no nulls so checking outliers
#in product quantity


# In[28]:


transactionData["PROD_QTY"].describe()


# In[29]:


#mean is 1.9 so check if there are any entires gretaer than lets say suppose 10
Y=transactionData[transactionData["PROD_QTY"]>10]


# In[30]:


Y


# In[31]:


#there are two entries with prod_qty value as 200 and note that 
#same customer has purchased these
# lets drop these entries


# In[32]:


transactionData.drop(labels=Y.index,inplace=True)


# In[33]:


# checking whether that customer bought any other items


# In[34]:


transactionData[transactionData["LYLTY_CARD_NBR"] == 226000]


# In[35]:


#filtered out the data
# now lets continue
# counitng number of transactions by date


# In[36]:


m=transactionData.groupby('DATE').count()


# In[37]:


m


# In[41]:


# finding the missing date
from datetime import date, timedelta
sdate = date(2018, 7, 1)   # start date
edate = date(2019, 6, 30)   # end date


# In[42]:


pd.date_range(sdate,edate).difference(transactionData["DATE"])


# In[43]:


# the missing date is 2018-12-25


# In[44]:


m.loc['2018-12-25']=np.nan


# In[45]:


m[m.index=='2018-12-25']


# In[46]:


check_null_date = pd.merge(pd.Series(pd.date_range(start=transactionData["DATE"].min(), end=transactionData["DATE"].max()), name="DATE"), transactionData, on="DATE", how="left")


# In[47]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


trans_by_date = check_null_date["DATE"].value_counts()
x=trans_by_date.sort_index()

x= pd.DataFrame({'date':x.index, 'sales':x.values})

y=x.groupby([x['date'].dt.year, x['date'].dt.month]).agg({'sales':sum})


# In[49]:


ax = y.plot(figsize=(15,3))
ax.set_xticks(np.arange(len(y)))
ax.set_xticklabels(y.index)


# In[50]:


trans_by_date = check_null_date["DATE"].value_counts()
dec = trans_by_date[(trans_by_date.index >= pd.datetime(2018,12,1)) & (trans_by_date.index < pd.datetime(2019,1,1))].sort_index()
dec.index = dec.index.strftime('%d')
ax = dec.plot(figsize=(15,3))
ax.set_xticks(np.arange(len(dec)))
ax.set_xticklabels(dec.index)
plt.title("2018 December Sales")
plt.show()


# In[51]:


transactionData["PROD_NAME"] = transactionData["PROD_NAME"].str.replace(r'[0-9]+(G)','g')

pack_sizes = transactionData["PROD_NAME"].str.extract(r'([0-9]+[gG])')[0].str.replace("g","").astype("float")


# In[52]:


# cheking the range of pack sizes and plotting the histogram


# In[53]:


print(pack_sizes.describe())
pack_sizes.plot.hist()


# In[54]:


#creating brand names column
transactionData['BRAND']=[s.split()[0] for s in transactionData['PROD_NAME']]


# In[55]:


transactionData['BRAND'].value_counts()


# In[56]:


# cleaning brand names
#replacing some of the values like dorito with doritos smith with smiths grain with grnwvs rrd with red 
transactionData['BRAND'].replace('Dorito','Doritos',inplace=True)
transactionData['BRAND'].replace('Infzns','Infuzions',inplace=True)
transactionData['BRAND'].replace('Smith','Smiths',inplace=True)
transactionData['BRAND'].replace('Snbts','Sunbites',inplace=True)
transactionData['BRAND'].replace('Red','RRD',inplace=True)
transactionData['BRAND'].replace('Old','Old El Paso',inplace=True)
transactionData['BRAND'].replace('WW','Woolworths',inplace=True)
transactionData['BRAND'].replace('Natural','NCC',inplace=True)


# In[57]:


customerData.describe()


# In[58]:


customerData.head()


# In[59]:


customerData.info()


# In[60]:


customerData.isna().sum()


# In[61]:


# evrything looks good now merging the data
merge_Data=pd.merge(transactionData,customerData,on='LYLTY_CARD_NBR')


# In[62]:


merge_Data.head()


# In[63]:


merge_Data.isna().sum()


# In[64]:


#saving the datafile
merge_Data.to_csv('merge.csv')


# # data analysis 

# Who spends the most on chips (total sales), describing customers by lifestage and
# how premium their general purchasing behaviour is
# - How many customers are in each segment
# - How many chips are bought per customer by segment
# - What's the average chip price by customer segment
# We could also ask our data team for more information. Examples are:
# - The customer's total spend over the period and total spend for each transaction
# to understand what proportion of their grocery spend is on chips
# - Proportion of customers in each customer segment overall to compare against the
# mix of customers who purchase chips
# Let's start with calculating total sales by LIFESTAGE and PREMIUM_CUSTOMER and
# plotting the split by these segments to describe which customer segment contribute
# most to chip sales.

# In[65]:


#Let's start with calculating total sales by LIFESTAGE and PREMIUM_CUSTOMER and
#plotting the split by these segments to describe which customer segment contribute
#most to chip sales.


# In[66]:


a=merge_Data[['LIFESTAGE','PREMIUM_CUSTOMER','TOT_SALES']].groupby(['PREMIUM_CUSTOMER','LIFESTAGE']).sum()
a.sort_values('TOT_SALES',ascending=False)


# In[67]:


#creating barplot
import seaborn as sns
plt.figure(figsize=(20,10))
sns.barplot(y=a.reset_index()['TOT_SALES'],x=a.reset_index()['LIFESTAGE'],hue=a.reset_index()['PREMIUM_CUSTOMER'])


# Sales are coming mainly from Budget - older families, Mainstream - young
# singles/couples, and Mainstream - retirees
# 

# In[68]:


#customers per segment
b=customerData.groupby(['PREMIUM_CUSTOMER','LIFESTAGE']).count()
b.columns=['CUSTOMER_COUNT']
b.sort_values('CUSTOMER_COUNT',ascending=False)


# In[69]:


plt.figure(figsize=(20,10))
sns.barplot(y=b.reset_index()['CUSTOMER_COUNT'],x=b.reset_index()['LIFESTAGE'],hue=a.reset_index()['PREMIUM_CUSTOMER'])


# There are more Mainstream - young singles/couples and Mainstream - retirees who buy
# chips. This contributes to there being more sales to these customer segments but
# this is not a major driver for the Budget - Older families segment.
# Higher sales may also be driven by more units of chips being bought per customer.
# Let's have a look at this next.

# In[70]:


# average number of units per customer


# In[71]:


c=merge_Data[['LIFESTAGE','PREMIUM_CUSTOMER','TOT_SALES']].groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).count()
c.sort_values('TOT_SALES',ascending=False)


# In[72]:



plt.figure(figsize=(20,10))
sns.barplot(y=c.reset_index()['TOT_SALES'],x=c.reset_index()['LIFESTAGE'],hue=a.reset_index()['PREMIUM_CUSTOMER'])


# In[94]:


#average price per unit chips bought for each customer
#segment as this is also a driver of total sales.


# In[93]:


merge_Data['CHIP_PRICE']=merge_Data['TOT_SALES']/merge_Data['PROD_QTY']
d=merge_Data[['LIFESTAGE','PREMIUM_CUSTOMER','CHIP_PRICE']].groupby(['PREMIUM_CUSTOMER','LIFESTAGE']).mean()
d.sort_values("CHIP_PRICE",ascending=False)


# In[ ]:


# performing ttest


# In[95]:


from scipy import stats


# In[96]:


#checking mainstream vs budget


# In[97]:


stats.ttest_ind([4.065642,3.994241],[3.657366,3.743328])


# In[98]:


#checking mainstream vs premium


# In[99]:


stats.ttest_ind([4.065642,3.994241],[3.770698,3.665414])


# the unit price for mainstream,
# young and mid-age singles and couples are significantly higher than
# that of budget or premium, young and midage singles and couples.

# In[100]:


# now let us look at what these mainstream, young and mid-age singles and couples prefer
# let us check what brand sdo they prefer


# In[109]:


merge_Data['PACKAGE_SIZE']=pack_sizes


# In[110]:


midage=merge_Data[(merge_Data['PREMIUM_CUSTOMER']=='Mainstream') & (merge_Data['LIFESTAGE']=='MIDAGE SINGLES/COUPLES')]
young=merge_Data[(merge_Data['PREMIUM_CUSTOMER']=='Mainstream') & (merge_Data['LIFESTAGE']=='YOUNG SINGLES/COUPLES')]
print(f"MIDAGE SINGLES/COUPLES\n{midage['BRAND'].value_counts().head(3)}")
print(f"YOUNG SINGLES/COUPLES\n{young['BRAND'].value_counts().head(3)}")
print(f"MIDAGE SINGLES/COUPLES\n{midage['PACKAGE_SIZE'].value_counts().head(3)}")
print(f"YOUNG SINGLES/COUPLES\n{young['PACKAGE_SIZE'].value_counts().head(3)}")


# In[105]:


# we can observe that kettle is popular among both the groups


# In[111]:


# 175, 150 and 134 packsizes are popular


# In[ ]:




