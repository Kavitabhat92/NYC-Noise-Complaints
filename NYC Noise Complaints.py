
# coding: utf-8

# # NYC Noise Complaints Data Analysis Project
# 
# 
# >- The dataset is based on the NYC's noise complaints which consists of complaints requests for different types of noises within New York City. It also provides us with the information about the agencies that are responsible for responding to these requests and the location where these complaints occur.
# 

# In[1]:

get_ipython().magic('matplotlib inline')

import os
import numpy as np # importing numpy array
import pandas as pd # importing panda library
import seaborn as sns # importing seaborn, advanced of matplot
import matplotlib.pyplot as plt #importing for plotting the graphs 
from statsmodels.formula.api import ols #importing ordinary least squares for linear regression
import scipy.stats as st
import statsmodels.api as sm #importing all the apis in the statmodels

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    import prep # noqa

pd.options.display.max_rows = 30
sns.set(style='ticks', context='talk')


# # DATA IMPORTING

# In[2]:

#read_csv is a function in pandas used to read data from an csv file into a list of DataFrames
tables = pd.read_csv("C:/Users/Kavit/Downloads/Noise_Data_new.csv")
tables.head()


# In[3]:

# Structure of the data
tables.shape


# As we can see, there 53 columns and total 342079 rows in our data.

# # Data Cleaning Process

# # Step 1: 
# Drop all columns with only NaN values

# In[4]:

tables=tables.dropna(axis=1, how='all') #  dropping columns which has only Nan values
tables.head()


# In[5]:

tables.shape


# 11 columns are dropped from our data. Let's proceed to Step 2

# # Removing unnecessary data
# 
# Dropping those columns which are 'UnSpecified'

# In[6]:

tables.drop(tables.columns[29:39],axis=1,inplace=True) # This will delete all the column from index 29 till 38
tables.drop(['Park Facility Name'],axis=1,inplace=True) # This will delete all column 'Park Facility Name' which is not relevant
tables.head()


# So far, we have removed all the data which is blank or null.
# From 53 columns ,we have removed 22 columns so far from our dataset

# # Modify data 
# 
# Modify the column names to make our lives simpler

# In[7]:

tables.rename(columns={'Unique Key': 'Unique_Key'}, inplace=True)
tables.rename(columns={'Complaint Type': 'Complaint_Type'}, inplace=True)
tables.rename(columns={'Location Type': 'Location_Type'}, inplace=True)

# We will work on mostly these column for our Analysis


# # Exploratory Data Analysis

# In[8]:

# Step 1:
# Converting the date columns to an appropriate format for further extraction
tables['Year_Date'] = pd.to_datetime(tables['Created Date'])
tables['Year']=(tables['Year_Date']).dt.year
tables['Hour'] = pd.to_datetime(tables['Created Date']).dt.hour
tables['Created Date'] = pd.to_datetime(tables['Created Date']).dt.date
tables['Closed Date'] = pd.to_datetime(tables['Closed Date']).dt.date
tables['Due Date'] = pd.to_datetime(tables['Due Date']).dt.date
tables['Resolution Action Updated Date']=pd.to_datetime(tables['Resolution Action Updated Date']).dt.date


# In[9]:

# Creating Resolution Time variable
tables['Resolution_Time']=(tables['Closed Date']- tables['Created Date']).dt.days
tables['Resolution_Time'] = tables['Resolution_Time'].fillna(0) # replacing blank values with 0


# In[10]:

tables.head()


# 
# # Problem Statement for our project
# 
# >- What are the factors that affects the resolution time for a complaint?
# 
# 
# 

# # Outlier Analysis
# Here we are trying to determine the data points that 

# In[11]:

fig, ax = plt.subplots(figsize=(20, 8))
sns.boxplot(x="Agency", y="Resolution_Time", data=tables, palette="PRGn")
# Below Boxplot confirms that DEP takes maximum time to resolve issues amongst others


# In[12]:

import scipy.stats as stats
import pylab
#measurements = np.random.normal(loc = 20, scale = 5, size=100)   
#stats.probplot(tables, dist="norm", plot=pylab)
#plt.title("PriceAverage Growing Season Temp of Bordeaux wine bottles")

#plt.grid(True)
plt.show()
plt.scatter(tables['Hour'],tables['Resolution_Time'])
plt.xlabel("Resolution time")
plt.ylabel("Hour of complaint")


# In[13]:

tables = tables[tables.Resolution_Time != 247]
tables = tables[tables.Resolution_Time != 238]
tables = tables[tables.Resolution_Time != 248]


# In[14]:

fig, ax = plt.subplots(figsize=(20, 8))
sns.boxplot(x="Agency", y="Resolution_Time", data=tables, palette="PRGn")
# Below Boxplot confirms that DEP takes maximum time to resolve issues amongst others


# In[15]:

# Distinct plot of resolution timesns.kdeplot(Data['Resolution_Time'],shade=True)
fig, ax = plt.subplots(figsize=(12, 6))
sns.distplot(tables['Resolution_Time'],bins=20,hist=False)


# In[16]:

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(tables['Resolution_Time'],shade=True)


# # Questions to the Data
# 
# >-  1. Which Agency receives the highest number of Complaints?

# In[ ]:




# In[81]:

z=tables.groupby('Agency').Unique_Key.nunique().to_frame()
z.reset_index(level=0, inplace=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='Agency', y= 'Unique_Key', data=z,
            color='#4c72b0', ax=ax)
sns.despine()


# In[17]:

# NYPD receives the highest number of complaints


# >- 2.Which Agency takes the maximum time to resolve a complaint?
# 

# In[18]:

# Bar plot which plots resolution Time vs the Agencies
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Agency', y= 'Resolution_Time', data=tables,color='#4c72b0')
sns.despine()
# We observe that NYPD has lowest resolution time whereas EDC has highest resolution time


# In[21]:

# This code groups by Landmark with resolution time
# To get which landmarks took maximum resolution time
c=tables.groupby('Agency').Descriptor.nunique().to_frame()
c.reset_index(level=0, inplace=True) # this code resets the index to a column for our analysis
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(y='Agency', x= 'Descriptor', data=c,
            color='#4c72b0', ax=ax)
sns.despine()


# In[19]:

# DEP receives maximum different types of Complaints whereas DSNY receives the lowest


# >- 3.Which type of complaints are reported and what are their counts?

# In[20]:

fig, ax = plt.subplots(figsize=(20, 8))
sns.countplot('Complaint_Type',data=tables)


# >- We also analysed which day of the week, which time of the day, causes more noise complaints...

# In[21]:

tables['Created Date'] = pd.to_datetime(tables['Created Date'])
tables['day_of_week'] = tables['Created Date'].dt.weekday_name
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y='day_of_week',data=tables)


# In[22]:

tables.loc[(tables.Hour <= 3) | (tables['Hour']>=23), 'Time_of_Day' ] = 'Late Night'
tables.loc[(tables.Hour <= 6) & (tables['Hour']>3), 'Time_of_Day' ] = 'Early Morning'
tables.loc[(tables.Hour > 6) & (tables['Hour']<=12), 'Time_of_Day' ] = 'Morning'
tables.loc[(tables.Hour > 12) & (tables['Hour']<=16), 'Time_of_Day' ] = 'Afternoon'
tables.loc[(tables.Hour > 16) & (tables['Hour']<=19), 'Time_of_Day' ] = 'Evening'
tables.loc[(tables.Hour >19) & (tables['Hour']<=22), 'Time_of_Day' ] = 'Night'


# In[23]:

max(tables['Hour'])


# In[24]:

fig, ax = plt.subplots(figsize=(20, 8))
sns.countplot('Time_of_Day',data=tables)


# In[25]:

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Time_of_Day', y= 'Resolution_Time', data=tables,color='#4c72b0')
sns.despine()


# In[27]:

a=tables.groupby('Year').Unique_Key.nunique().to_frame()
a.reset_index(level=0, inplace=True)
sns.countplot('Year',data=tables)


# In[65]:

a.head()


# In[26]:

fig, ax = plt.subplots(figsize=(12, 6))
sns.stripplot(x="Resolution_Time", y="day_of_week", data=tables)


# In[27]:

Data=tables[['Agency','Complaint_Type','Descriptor','Location_Type','Resolution_Time','Hour','Year']]


# # Factors Identified...
# >- 
# Unique_key,
# created Date,
# Closed Date,
# Agency,
# Complaint Type,
# Descriptor,
# Location Type,
# Latitude,
# Longitude
# 

# In[28]:

Data.head()


# # Creating Linear Models

# In[29]:

# Creating a Linear Model
model = ols("Resolution_Time ~Agency+Complaint_Type+Descriptor+Location_Type+Hour+Year", Data).fit()


# In[30]:

model.summary()


# In[33]:

# Creating a Linear Model
model = ols("Resolution_Time ~Agency+Complaint_Type+Hour+Year", Data).fit()


# In[34]:

model.summary()


# In[35]:

# Creating a Linear Model
model3 = ols("Resolution_Time ~Agency+Hour+Year", Data).fit()
model3.summary()


# In[36]:

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model3, fig=fig)


# In[37]:

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model3, "Hour", fig=fig)


# In[39]:

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(model3, "Hour", ax=ax)


# In[89]:

# Converting the Categorical Variable Agency to a Continuous variable


# In[40]:

Data1=pd.get_dummies(Data['Agency'])
Data1=pd.concat([Data['Resolution_Time'],Data['Hour'],Data['Year'],Data1],axis=1)
Data1.head()


# In[91]:

Data1.dtypes


# # Using Recursive Feature Estimation

# In[42]:

X=Data1[['Hour','Year','DEP','DSNY','EDC','NYPD']]
Y=Data1['Resolution_Time']


# In[43]:

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(model, 6)
fit = rfe.fit(X, Y)


# In[44]:

print(rfe.support_)
print(rfe.ranking_)


# In[45]:

# Creating a Linear Model
model4 = ols("Resolution_Time ~DEP+DSNY+EDC+NYPD+Hour+Year", Data1).fit()
model4.summary()


# In[46]:

# Creating a Linear Model
model5 = ols("Resolution_Time ~Hour+Year", Data1).fit()
model5.summary()


# In[47]:

# Creating a Linear Model
model6 = ols("Resolution_Time ~Hour", Data1).fit()
model6.summary()


# # Creating Training and Test Data
# >- We divided our data set into training data(80%) and test data(20%) to evaluate our model

# In[48]:

msk = np.random.rand(len(Data)) < 0.8


# In[49]:

train = Data[msk]


# In[50]:

test=Data[~msk]


# In[51]:

len(train)


# In[52]:

len(test)


# In[53]:

train.head()


# In[54]:

Data_train=pd.get_dummies(train['Agency'])
Data_train=pd.concat([train['Resolution_Time'],train['Hour'],train['Year'],Data_train],axis=1)
Data_train.head()


# In[66]:

Data_test=pd.get_dummies(test['Agency'])
Data_test=pd.concat([test['Resolution_Time'],test['Hour'],test['Year'],Data_test],axis=1)
Data_test.head()


# In[56]:

new_model=ols("Resolution_Time ~DEP+DSNY+EDC+NYPD+Hour+Year",Data_train).fit()
new_model.summary()


# In[65]:

#Data_test=pd.get_dummies(test['Agency'])
#Data_test=pd.concat([test['Resolution_Time'],test['Hour'],test['Year']],axis=1)
#Data_test.head()


# In[67]:

y_test=test['Resolution_Time']


# In[68]:

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
X=Data_train[['Hour','DEP','DSNY','NYPD','EDC','Year']]
Y=Data_train['Resolution_Time']
lm.fit(X,Y)


# In[159]:

test['Year'],test['DEP'],test['DSNY'],test['NYPD'],test['EDC']


# In[72]:

#Data_test=pd.get_dummies(test['Agency'])
#Data_test=pd.concat([test['Resolution_Time'],test['Hour']],axis=1)


# In[62]:

Data_test.head()


# In[69]:

X_test=Data_test[['Hour','DEP','DSNY','NYPD','EDC','Year']]
YPred = lm.predict(X_test)


# In[70]:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, YPred))


# In[71]:

print('Variance score: %.2f' % r2_score(y_test, YPred))
# this score is very good


# # Creating another model

# In[73]:

X_2=Data_train[['Hour','Year']]
Y_2=Data_train['Resolution_Time']
lm.fit(X_2,Y_2)


# In[74]:

X_test_2=Data_test[['Hour','Year']]
YPred_2 = lm.predict(X_test_2)


# In[75]:

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, YPred_2))


# In[76]:

print('Variance score: %.2f' % r2_score(y_test, YPred_2)) # Poor


# Efficiency of the above model is very poor.
# This indicates that Hour and Year cannot predict Resolution Time

# In[169]:




# In[77]:

X3=Data_train[['Hour','DEP','DSNY','NYPD','EDC']]
Y3=Data_train['Resolution_Time']
lm.fit(X3,Y3)
X_test_3=Data_test[['Hour','DEP','DSNY','NYPD','EDC']]
YPred_3 = lm.predict(X_test_3)


# In[78]:

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, YPred_3))


# In[79]:

print('Variance score: %.2f' % r2_score(y_test, YPred_3)) 


# The variance Score is good, hence we can say that Resolution Time is dependent on Hour (at which complaint was issued) and the Agencies which are responsible in resolving a complaint

# # Visualization 
# 
# >- We used Bokeh visualization library to create our own interactive map plot.

# In[13]:

from bokeh.io import show #this command is used to import or export a file to the file system
#this command imports the tools required for axis and grids
from bokeh.models import ( GMapPlot, GMapOptions,WheelZoomTool,DataRange1d,BoxSelectTool, PanTool,
    ColumnDataSource, Circle,
    HoverTool,
    LogColorMapper
)

from bokeh.palettes import Viridis6 as palette # this command provide a collection of palettes for color mapping.
from bokeh.plotting import figure,output_file #imports the required figures like lines ,asteriks and circles for plotting data


# In[14]:

#GMapOptions is used to set a latitude, longitude and the type of the map needed to be present.
map_options = GMapOptions(lat=40.8311959, lng =-73.93034856, map_type="roadmap", zoom=11)


# In[15]:

#Google Maps is used underneath Bokeh plot using GMapPlot, which uses Google Maps API key 
plot=GMapPlot(x_range=DataRange1d(),
             y_range=DataRange1d(),
             map_options=map_options,
             api_key="AIzaSyAR6jG-6yNiqLOzzbpA6HfDMX_Jvrq8AFU")

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())


# In[16]:

#Creating another dataframe to filter the data to find the latitude and longtitude where the resolution time is higher than avaerage
df = tables.filter(['Resolution_Time','Longitude','Latitude'], axis=1)
mean = df['Resolution_Time'] > 10.0
df = df[mean]
df.head()


# In[17]:

source=ColumnDataSource(data=dict(
    lat=df['Latitude'],
    lon=df['Longitude'],
    resolving=df['Resolution_Time']
))


# In[18]:

palette.reverse()

color_mapper = LogColorMapper(palette=palette) 
#providing the tools that can be used for interactive bokeh maps
TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="New Jersey Unemployment, 2009", tools=TOOLS,
    x_axis_location=None, y_axis_location=None
)

#returns the model specified in the argument i.e Hovertool
hover = p.select_one(HoverTool)
#Whether the tooltip position should snap to the “center” (or other anchor) position of the associated glyph, or always follow the 
#current mouse cursor position.
hover.point_policy = "follow_mouse"
#hover.
tooltips = [
    ("Resolution Time)", "@resolving%"),
    ("(Lon, Lat)", "($x, $y)"),
]


# In[19]:

circle= Circle(x="lon",
               y="lat",
               fill_color={'field': 'resolving', 'transform': color_mapper},
               fill_alpha=0.7)
circle_renderer = plot.add_glyph(source,circle)

plot.add_tools(HoverTool(tooltips=tooltips, renderers=[circle_renderer]))


# In[20]:

#output_file("NoiseData.html")
show(plot)


# ![image.png](attachment:image.png)

# # Future Work

# >- By analysis of this dataset we can predict which future complaints will take what time to resolve and which agency will take least time to solve the complaints.
