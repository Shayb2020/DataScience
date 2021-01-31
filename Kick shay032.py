#!/usr/bin/env python
# coding: utf-8

# ***
# # Introduction

# **Project aims**
# 
# Start-up Nation - Two items published recently point more than anything to the strength of the local high-tech industry:
# 1. While job vacancies declined sharply since the start of the corona crisis, the demand for employees in the high-tech sector returned very close to its pre-corona figure in November.
# 2. Foreign direct investments (FDI) reached a record level of USD 19 billion in the first three quarters of the year.I believe that most of the investments focused on the high-tech industry...
# 
# Looking ahead, as long as the extremely low interest rate environment continues globally, and the NASDAQ remains at its record levels (evidence of the positive global sentiment towards tech companies), this trend will probably persist....
# 
# In recent years, the range of funding options for projects created by individuals and small companies has expanded considerably. In addition to savings, bank loans, friends and family funding and other traditional options, crowdfunding has become a popular and readily available alternative.
# 
# 
# __[Kickstarter](https://www.kickstarter.com/)__, founded in 2009, is one particularly well-known and popular crowdfunding platform. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.
# 
# A huge variety of factors contribute to the success or failure of a project - in general, but also on Kickstarter. Some of these are able to be quantified or categorised, which allows for the construction of a model to attempt to predict whether a project will succeed or not. The aim of this project is to construct such a model and also to analyse Kickstarter project data more generally, in order to help potential project creators to assess whether or not Kickstarter is a good funding option for them, and what their chances of success are.

# **Dataset**
# 
# The dataset used in this project was downloaded in .csv format from a webscrape conducted by a webscraping site called __[Web Robots](https://webrobots.io/kickstarter-datasets/)__. The dataset contains data on all projects hosted on Kickstarter between the company's launch in April 2009, up until the date of the webscrape on 11 Nov 2020.

# ***
# # Importing the data

# In[45]:


# Importing the required libraries
import pandas as pd
pd.set_option('display.max_columns', 50) # Display up to 50 columns at a time
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
plt.style.use('seaborn')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,5
import glob # To read all csv files in the directory
import seaborn as sns
import calendar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import itertools
import time
import xgboost as xgb


# ***
# # Flat-file generation
# 
# The most recent Kickstarter data from https://webrobots.io/kickstarter-datasets/ (from 11 Nov 2020) is stored in 58 separate csv files. The code below creates a list of all csv files beginning with 'Kickstarter' and concatenates them into one dataframe:

# In[46]:


path = r'C:\Users\shay\Documents\BIU\DATA_SCIENCE\Final Projects\SMB pred success\kickstarter-unzip'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)


# In[4]:


df.head()


# In[47]:


len(df)


# The resulting dataframe contains 213,193 projects.

# ***
# # Exploratory Data Analysis (EDA)
# 

# In this section the data will be cleaned and pre-processed in order to allow for exploratory data analysis and modeling.

# **Inspecting and dropping columns**

# In[48]:


# Checking the columns
df.columns


# Description of each column:
# - backers_count - number of people who contributed funds to the project
# - blurb - short description of the project
# - category - contains the category and sub-category of the project
# - converted_pledged_amount - amount of money pledged, converted to the currency in the 'current_currency' column
# - country - country the project creator is from
# - created_at - date and time of when the project was initially created on Kickstarter
# - creator - name of the project creator and other information about them, e.g. Kickstarter id number
# - currency - original currency the project goal was denominated in
# - currency_symbol - symbol of the original currency the project goal was denominated in
# - currency_trailing_code - code of the original currency the project goal was denominated in
# - current_currency - currency the project goal was converted to
# - deadline - date and time of when the project will close for donations
# - disable_communication - whether or not a project owner disabled communication with their backers
# - friends - unclear (null or empty)
# - fx_rate - foreign exchange rate between the original currency and the current_currency
# - goal - funding goal
# - id - id number of the project
# - is_backing - unclear (null or false)
# - is_starrable - whether or not a project can be starred (liked and saved) by users
# - is_starred - whether or not a project has been starred (liked and saved) by users
# - launched_at - date and time of when the project was launched for funding
# - location - contains the town or city of the project creator
# - name - name of the project
# - permissions - unclear (null or empty)
# - photo - contains a link and information to the project's photo/s
# - pledged - amount pledged in the current_currency
# - profile - details about the project's profile, including id number and various visual settings
# - slug - name of the project with hyphens instead of spaces
# - source_url - url for the project's category
# - spotlight - after a project has been successful, it is spotlighted on the Kickstarter website
# - staff_pick - whether a project was highlighted as a staff_pick when it was launched/live
# - state - whether a project was successful, failed, canceled, suspending or still live
# - state_changed_at - date and time of when a project's status was changed (same as the deadline for successful and failed projects)
# - static_usd_rate - conversion rate between the original currency and USD
# - urls - url to the project's page
# - usd_pledged - amount pledged in USD
# - usd_type - domestic or international

# In[49]:


# Checking for duplicates of individual projects

print(f"Of the {len(df)} projects in the dataset, there are {len(df[df.duplicated(subset='id')])} which are listed more than once.")


# Duplicates are an issue in this dataset and will need to be dealt with.
# 
# Further cleaning of the data will help clarify which duplicates, if any, need to be removed.

# In[50]:


# Checking column information
df.info()


# Some columns contain very few nun-null entries, and can be dropped:

# In[51]:


# Dropping columns that are mostly null
df.drop(['friends', 'is_backing', 'is_starred', 'permissions'], axis=1, inplace=True)


# Other columns are not useful for the purposes of this project, and can also be dropped for these reasons:
# 
# - converted_pledged_amount - most currencies are converted into USD in this column, but not all. Instead, the 'usd_pledged' column will be used as these all use the same currency (the dollar).
# - creator - most projects are by different people, and so this cannot be usefully used to group or categorise projects, and is not useful in a machine learning context.
# - currency - all currency values will be used as/converted to dollars, so that they can be evaluated together. It is not necessary to keep the original record because of this, and because it will be highly correlated with country (which will be kept).
# - currency_symbol - as above.
# - currency_trailing_code - as above.
# - current_currency - as above.
# - fx_rate - this is used to create 'converted_pledged_amount' from 'pledged', but does not always convert to dollars so can be dropped in favour of 'static_usd_rate' which always converts to dollars.
# - photo - image processing/computer vision will not be used in this project.
# - pledged - data in this column is stored in native currencies, so this will be dropped in favour of 'usd_pledged' which is all in the same currency (dollars).
# - profile - this column contains a combination of information from other columns (e.g. id, state, dates, url).
# - slug - this is simply the 'name' column with hyphens instead of spaces.
# - source_url - the sites that the rows were each scraped from is not useful for building a model, as each is unique to an id.
# - spotlight - projects can only be spotlighted after they are already successful, so this will be entirely correlated with successful projects.
# - state_changed_at - this is the same as deadline for most projects. The only exceptions are for projects which were cancelled before their deadline, but they will not be included in this analysis.
# - urls - as with source_url.
# - usd_type - it is unclear what this column means, but it is unlikely to be necessary since all currency values will be converted to dollars, and other currency information has been dropped.

# In[52]:


# Dropping columns that aren't useful
df.drop(['converted_pledged_amount', 'creator', 'currency', 'currency_symbol', 'currency_trailing_code', 'current_currency', 'fx_rate', 'photo', 'pledged', 'profile', 'slug', 'source_url', 'spotlight', 'state_changed_at', 'urls', 'usd_type'], axis=1, inplace=True)


# In[11]:


df.head()


# **Converting datetime columns**

# Columns containing dates are currently stored in unix time, and need to be converted to datetime. Because they have been converted from unix, all times are in GMT( Greenwich Mean Time).

# In[53]:


# Converting dates from unix to datetime
cols_to_convert = ['created_at', 'deadline', 'launched_at']
for c in cols_to_convert:
    df[c] = pd.to_datetime(df[c], origin='unix', unit='s')


# In[54]:


print(f"The dataset contains projects added to Kickstarter between {min(df.created_at).strftime('%d %B %Y')} and {max(df.created_at).strftime('%d %B %Y')}.")


# In[55]:


df.shape


# ***
# #Individual columns will now be pre-processed, and additional features engineered, where necessary.

# **Blurb**
# ##### short description of the project

# Natural language processing can be helpful to find out if the description od the project can effect the number of backers.
# 
# In addition, The length of the blurbs written by project creators will be calculated, in case this is useful for the model (e.g. people preferring to read shorter or longer blurbs when choosing what to fund). The original blurb variable will then be dropped.

# In[56]:


#Text Analytics of The Blurb - Sentiment Analysis
get_ipython().system('pip install textblob ')
get_ipython().system('python -m textblob.download_corpora ')


# In[57]:


from textblob import TextBlob, Word, Blobber


# In[58]:


TextAna = df.copy()
TextAna.drop(['backers_count', 'category', 'country', 'country_displayable_name','created_at','deadline','disable_communication','goal','id','is_starrable','launched_at','location','name','staff_pick','static_usd_rate','usd_pledged'], axis=1, inplace=True)
#TextAna.head(5)


# In[59]:


#TextAna = TextAna[TextAna['blurb'].isin([1,10])]
TextAna = TextAna.reset_index(drop = True)
#TextAna.head()


# In[60]:


TextAna['blurb'] = TextAna['blurb'].str.replace("[^a-zA-Z]", " ")


# In[61]:


TextAna.head(4)


# In[62]:


TextAna['Polarity'] = np.nan
TextAna['Subjectivity'] = np.nan

pd.options.mode.chained_assignment = None

for idx, articles in enumerate(TextAna.blurb.values):  # for each row in our df dataFrame
    articles = str(articles)
    sentA = TextBlob(articles) # pass the text only article to TextBlob to analyse
    TextAna['Polarity'].iloc[idx] = sentA.sentiment.polarity # write sentiment polarity back to df
    TextAna['Subjectivity'].iloc[idx] = sentA.sentiment.subjectivity # write sentiment subjectivity score back to df

TextAna.head()


# In[63]:


#dfsent['sentiment'] = [1 if dfsent.polarity > 0 else -1 if dfsent.polarity < 0 else 0]
TextAna['sentiment'] = 0
TextAna.loc[TextAna.Polarity<-0.05,'sentiment'] = -1
TextAna.loc[TextAna.Polarity>0.05,'sentiment'] = 1
TextAna['sentiment'] = TextAna['sentiment'].astype('category')


# In[64]:


TextAna.groupby('sentiment').count()['Polarity']


# In[65]:


#Distribution of sentiment 
TextAna['sentiment'].value_counts().plot(kind='barh', color='firebrick')
plt.title('sentiment Distribution')
plt.ylabel('Projects')


# In[67]:


df['sentiment'] = TextAna['sentiment']
df.head()


# In[68]:


# Count length of each blurb
df['blurb_length'] = df['blurb'].str.split().str.len()

# Drop blurb variable
df.drop('blurb', axis=1, inplace=True)
df.head()


# In[27]:


df.head()


# **Category**

# The category variable is currently stored as a string, although it was clearly originally a dictionary. The example below shows that each project has both a category (e.g. games) and a sub-category (e.g. tabletop games). Both will be extracted.

# In[69]:


# Example category value
df.iloc[0]['category']


# In[70]:


# Extracting the relevant sub-category section from the string
f = lambda x: x['category'].split('/')[1].split('","position')[0]
df['sub_category'] = df.apply(f, axis=1)

# Extracting the relevant category section from the string, and replacing the original category variable
f = lambda x: x['category'].split('"slug":"')[1].split('/')[0]
df['category'] = df.apply(f, axis=1)
f = lambda x: x['category'].split('","position"')[0] # Some categories do not have a sub-category, so do not have a '/' to split with
df['category'] = df.apply(f, axis=1)


# In[71]:


# Counting the number of unique categories
print(f"There are {df.category.nunique()} unique categories and {df.sub_category.nunique()} unique sub-categories.")


# **Disable_communication**

# 99.7% of project owners did not disable communication with their backers (unsurprisingly). Because nearly all projects have the same value for this variable, it will be dropped as it does not provide much information.

# In[72]:


# Checking the proportions of each category
df.disable_communication.value_counts(normalize=True)


# In[73]:


df.drop('disable_communication', axis=1, inplace=True)


# **Goal**

# The goal amount of funding for each project is currently recorded in native currencies. In order to allow for fair comparisons between projects, goals will be converted into dollars (as amount pledged already is).

# In[74]:


# Calculate new column 'usd_goal' as goal * static_usd_rate
df['usd_goal'] = round(df['goal'] * df['static_usd_rate'],2)


# In[75]:


# Dropping goal and static_usd_rate
df.drop(['goal', 'static_usd_rate'], axis=1, inplace=True)


# In[35]:


df['usd_goal']


# **Is_starrable**

# Only 2.9% of projects were starrable by users. Although this is only a very small proportion, whether or not a project was liked and saved by users is likely to be informative about whether or not a project was successful, so the variable will be kept for now and assessed again once irrelevant rows have been dropped, to check it is still useful.

# In[76]:


# Figure out what this is, and do a count_values() to figure out whether it's worth including or mostly FALSE
df.is_starrable.value_counts(normalize=True)


# **Location**

# The location field contains the town/city that a project originates from, as well as the country. There are a large number (16,865) of unique locations. Because the country is already recorded separately in the country field, and there are such a large number of unique categories (making one-hot encoding not useful, particularly as there are likely to be a lot of smaller towns and cities with very few projects), the column will be dropped.

# In[77]:


# Example location value
df.iloc[0]['location']


# In[78]:


# Counting the number of unique locations
df.location.nunique()


# In[79]:


# Dropping location
df.drop('location', axis=1, inplace=True)


# **Name**

# The length of project names will be calculated, in case this is useful for the model. The original name variable will then be dropped.

# In[80]:


# Count length of each name
df['name_length'] = df['name'].str.split().str.len()
# Drop name variable
df.drop('name', axis=1, inplace=True)


# **Usd_pledged**

# This column requires rounding to two decimal places.

# In[81]:


df['usd_pledged'] = round(df['usd_pledged'],2)


# **Additional calculated features**

# Additional features can be calculated from the existing features, which may also help to predict whether a project is successfully funded. The features to be added are:
# 
# 1) Time from creation to launch.
# 
# 2) Campaign length.
# 
# 3) Launch day of week.
# 
# 4) Deadline day of week.
# 
# 5) Launch month.
# 
# 6) Deadline month.
# 
# 7) Launch time of day.
# 
# 8) Deadline time of day.
# 
# 9) Mean pledge per backer.
# 
# Original datetime values and the mean pledge per backer will be kept in for now for EDA purposes, but will be removed later, before modeling.

# In[82]:


# 1]Time between creating and launching a project

df['creation_to_launch_days'] = df['launched_at'] - df['created_at']
df['creation_to_launch_days'] = df['creation_to_launch_days'].dt.round('d').dt.days # Rounding to nearest days, then showing as number only

# Or could show as number of hours:
# df['creation_to_launch_hours'] = df['launched_at'] - df['created_at']
# df['creation_to_launch_hours'] = df['creation_to_launch_hours'].dt.round('h') / np.timedelta64(1, 'h') 


# In[83]:


# 2]Campaign length
df['campaign_days'] = df['deadline'] - df['launched_at']
df['campaign_days'] = df['campaign_days'].dt.round('d').dt.days # Rounding to nearest days, then showing as number only


# In[84]:


# 3]Launch day of week
df['launch_day'] = df['launched_at'].dt.day_name()


# In[85]:


# 4]Deadline day of week
df['deadline_day'] = df['deadline'].dt.day_name()


# In[86]:


# 5]Launch month
df['launch_month'] = df['launched_at'].dt.month_name()


# In[87]:


# 6]Deadline month
df['deadline_month'] = df['deadline'].dt.month_name()


# In[88]:


# 7]Launch time
df['launch_hour'] = df['launched_at'].dt.hour # Extracting hour from launched_at

def two_hour_launch(row):
    '''Creates two hour bins from the launch_hour column'''
    if row['launch_hour'] in (0,1):
        return '12am-2am'
    if row['launch_hour'] in (2,3):
        return '2am-4am'
    if row['launch_hour'] in (4,5):
        return '4am-6am'
    if row['launch_hour'] in (6,7):
        return '6am-8am'
    if row['launch_hour'] in (8,9):
        return '8am-10am'
    if row['launch_hour'] in (10,11):
        return '10am-12pm'
    if row['launch_hour'] in (12,13):
        return '12pm-2pm'
    if row['launch_hour'] in (14,15):
        return '2pm-4pm'
    if row['launch_hour'] in (16,17):
        return '4pm-6pm'
    if row['launch_hour'] in (18,19):
        return '6pm-8pm'
    if row['launch_hour'] in (20,21):
        return '8pm-10pm'
    if row['launch_hour'] in (22,23):
        return '10pm-12am'
    
df['launch_time'] = df.apply(two_hour_launch, axis=1) # Calculates bins from launch_time

df.drop('launch_hour', axis=1, inplace=True)


# In[89]:


# 8]Deadline time
df['deadline_hour'] = df['deadline'].dt.hour # Extracting hour from deadline

def two_hour_deadline(row):
    '''Creates two hour bins from the deadline_hour column'''
    if row['deadline_hour'] in (0,1):
        return '12am-2am'
    if row['deadline_hour'] in (2,3):
        return '2am-4am'
    if row['deadline_hour'] in (4,5):
        return '4am-6am'
    if row['deadline_hour'] in (6,7):
        return '6am-8am'
    if row['deadline_hour'] in (8,9):
        return '8am-10am'
    if row['deadline_hour'] in (10,11):
        return '10am-12pm'
    if row['deadline_hour'] in (12,13):
        return '12pm-2pm'
    if row['deadline_hour'] in (14,15):
        return '2pm-4pm'
    if row['deadline_hour'] in (16,17):
        return '4pm-6pm'
    if row['deadline_hour'] in (18,19):
        return '6pm-8pm'
    if row['deadline_hour'] in (20,21):
        return '8pm-10pm'
    if row['deadline_hour'] in (22,23):
        return '10pm-12am'
    
df['deadline_time'] = df.apply(two_hour_deadline, axis=1) # Calculates bins from launch_time

df.drop('deadline_hour', axis=1, inplace=True)


# In[90]:


# 9]Mean pledge per backer
df['pledge_per_backer'] = round(df['usd_pledged']/df['backers_count'],2)


# **Checking for null values**

# In[91]:


null = df.isna().sum().sort_values(ascending = False)
null.to_frame().style.background_gradient(cmap = 'cividis')


# There are eight projects without a blurb_length, i.e. without a blurb. These can be replaced with a length of 0.

# In[92]:


# Replacing null values for blurb_length with 0
df.blurb_length.fillna(0, inplace=True)


# In[93]:


# Confirming there are no null values remaining
df.isna().sum().sum()


# **Dropping rows**

# This project aims to predict whether projects succeed or fail. The dataset also includes projects canceled, live (i.e. not yet finished) and suspended projects. 
# 
# These will now be removed.

# In[94]:


#Distribution of project status
df['state'].value_counts().plot(kind='barh', color='firebrick')
plt.title('Kickstarter Project Status')
plt.ylabel('Projects')


# In[95]:


# Number of projects of different states
df.state.value_counts()


# In[96]:


# Dropping projects which are not successes or failures
df = df[df['state'].isin(['successful', 'failed'])]


# In[97]:


# Confirming that the most recent deadline is the day on which the data was scraped, i.e. there are no projects which have yet to be resolved into either successes or failures
max(df.deadline)


# **Dropping duplicates**

# As demonstrated above, some projects are included in the dataset more than once. Duplicates will now be assessed and removed.

# In[98]:


# Checking for duplicates of individual projects, and sorting by id
duplicates = df[df.duplicated(subset='id')]
print(f"Of the {len(df)} projects in the dataset, there are {len(df[df.duplicated(subset='id')])} which are listed more than once.")
print(f"Of these, {len(df[df.duplicated()])} have every value in common between duplicates.")


# In[99]:


# Dropping duplicates which have every value in common
df.drop_duplicates(inplace=True)


# In[100]:


len(df)


# In[101]:


print(len(df[df.duplicated(subset='id')]), "duplicated projects remain.")
duplicated = df[df.duplicated(subset='id', keep=False)].sort_values(by='id')
duplicated


# The above table show that for each pair of duplicates, there are differences in the usd_pledge and usd_goal columns. The differences are only in the order of a few cents/dollars, so it does not make much difference which one is kept. Therefore the first one of each pair will be dropped.

# In[102]:


df.drop_duplicates(subset=['id'], keep='first', inplace = True)


# In[103]:


len(df)


# In[104]:


df_dup = df.copy()

# sorting by first name 
df_dup.sort_values("id", inplace = True) 

# making a bool series 
bool_series = df_dup["id"].duplicated(keep = False) 

# bool series 
bool_series 

# passing NOT of bool series to see unique values only 
df_dup = df_dup[~bool_series] 

# displaying data 
df_dup.info() 
df_dup 


# In[105]:


TextAna.head(5)


# In[106]:


df = df_dup.copy()
df.shape


# Comparing rows for each duplicated project:

# **Setting the index**

# The id will now be set as the index.

# In[108]:


# Setting the id column as the index
df.set_index('id', inplace=True)


# In[110]:


df.head()


# ***
# # Exploring the data

# In this section, exploratory data analysis will be conducted, in order to explore the data and draw useful insights.

# **Key statistics**

# In[111]:


# Summary statistics for the numerical features
df.describe()


# In[112]:


print("Key stats:")
print("\nThe total amount of money that projects have aimed to raise is ${0:,.0f}".format(df.usd_goal.sum()))
print("The total amount of money pledged by backers is ${0:,.0f}".format(df.usd_pledged.sum()))
print("The total amount of money pledged by backers to successful projects is ${0:,.0f}".format(sum(df.loc[df['state'] == 'successful'].usd_pledged)))

print("\nThe total number of successful or failed projects launched on Kickstarter is: {0:,}".format(len(df)))
print("The total number of projects which were successfully funded is: {0:,}".format(len(df.loc[df['state'] == 'successful'])))
print(f"The proportion of completed projects which were successfully funded is: {int(round((len(df.loc[df['state'] == 'successful'])/len(df))*100,0))}%")

print("\nThe mean project fundraising goal is ${0:,.0f}".format(df.usd_goal.mean()))
print("The mean amount pledged per project is ${0:,.0f}".format(df.usd_pledged.mean()))
print("The mean amount pledged per successful project is ${0:,.0f}".format(df.loc[df['state'] == 'successful'].usd_pledged.mean()))
print("The mean amount pledged per failed project is ${0:,.0f}".format(df.loc[df['state'] == 'failed'].usd_pledged.mean()))
      
print("\nThe mean number of backers per project is", int(round(df.backers_count.mean(),0)))
print("The mean pledge per backer is ${0:,.0f}".format(df.pledge_per_backer.mean()))
print("The mean number of days a campaign is run for is", int(round(df.campaign_days.mean(),0)))


# **How do succesful and failed projects differ?**

# The graphs below show how various features differ between failed and successful projects.
# - Unsurprisingly, successful projects tend to have smaller (and therefore more realistic) goals - the median amount sought by successful projects is about half that of failed projects.
# - The differences in the median amount pledged per project are more surprising. The median amount pledged per successful project is considerably higher than the median amount requested, suggesting that projects that meet their goal tend to go on to gain even more funding, and become 'over-funded'.
# - On a related note, the difference between failed and successful companies is much larger in terms of amount pledged and the number of backers, compared to goal amount. Probably once potential backers see that a project looks like it will be successful, they are much more likely to jump on the bandwagon and fund it.
# - Successful projects have slightly shorter durations.
# - Successful projects tend to take slightly longer to launch, measured from the time the project was first created on the site.
# - Average name and blurb lengths are very similar between failed and successful projects.
# - Roughly 20% of successful projects were highlighted on the site as staff picks. It does not seem unreasonable to suggest a causative relationship here, i.e. that projects that are chosen as staff picks are much more likely to go on to be successful, and that only a few staff picks go on to fail. This measurement is possibly polluted by the point at which a project is chosen as a staff pick, however - e.g. a project may already have some backers and funding when it is chosen as a staff pick.

# In[122]:


# Plotting the average amount pledged to successful and unsuccesful projects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))

df['state'].value_counts(ascending=True).plot(kind='bar', ax=ax1, color=['firebrick', 'seagreen'], rot=0)
ax1.set_title('Number of projects')
ax1.set_xlabel('')

df.groupby('state').usd_goal.median().plot(kind='bar', ax=ax2, color=['firebrick', 'seagreen'], rot=0)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')


# In[123]:


fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16,8))
df.groupby('state').usd_pledged.median().plot(kind='bar', ax=ax3, color=['firebrick', 'seagreen'], rot=0)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

df.groupby('state').backers_count.median().plot(kind='bar', ax=ax4, color=['firebrick', 'seagreen'], rot=0)
ax4.set_title('Median backers per project')
ax4.set_xlabel('')


# In[124]:


fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(16,8))
df.groupby('state').campaign_days.mean().plot(kind='bar', ax=ax5, color=['firebrick', 'seagreen'], rot=0)
ax5.set_title('Mean campaign length (days)')
ax5.set_xlabel('')

df.groupby('state').creation_to_launch_days.mean().plot(kind='bar', ax=ax6, color=['firebrick', 'seagreen'], rot=0)
ax6.set_title('Mean creation to launch length (days)')
ax6.set_xlabel('')


# In[126]:


fig, (ax7, ax8, ax9) = plt.subplots(1, 3, figsize=(16,8))
df.groupby('state').name_length.mean().plot(kind='bar', ax=ax7, color=['firebrick', 'seagreen'], rot=0)
ax7.set_title('Mean name length (words)')
ax7.set_xlabel('')

df.groupby('state').blurb_length.mean().plot(kind='bar', ax=ax8, color=['firebrick', 'seagreen'], rot=0)
ax8.set_title('Mean blurb length (words)')
ax8.set_xlabel('')

# Creating a dataframe grouped by staff_pick with columns for failed and successful
pick_df = pd.get_dummies(df.set_index('staff_pick').state).groupby('staff_pick').sum()
# Normalizes counts by column, and selects the 'True' category (iloc[1])
(pick_df.div(pick_df.sum(axis=0), axis=1)).iloc[1].plot(kind='bar', ax=ax9, color=['firebrick', 'seagreen'], rot=0) 
ax9.set_title('Proportion that are staff picks')
ax9.set_xlabel('')


# In[127]:


# Plotting the average amount pledged to successful and unsuccesful projects
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12,12))

df['state'].value_counts(ascending=True).plot(kind='bar', ax=ax1, color=['firebrick', 'seagreen'], rot=0)
ax1.set_title('Number of projects')
ax1.set_xlabel('')

df.groupby('state').usd_goal.median().plot(kind='bar', ax=ax2, color=['firebrick', 'seagreen'], rot=0)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

df.groupby('state').usd_pledged.median().plot(kind='bar', ax=ax3, color=['firebrick', 'seagreen'], rot=0)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

df.groupby('state').backers_count.median().plot(kind='bar', ax=ax4, color=['firebrick', 'seagreen'], rot=0)
ax4.set_title('Median backers per project')
ax4.set_xlabel('')

df.groupby('state').campaign_days.mean().plot(kind='bar', ax=ax5, color=['firebrick', 'seagreen'], rot=0)
ax5.set_title('Mean campaign length (days)')
ax5.set_xlabel('')

df.groupby('state').creation_to_launch_days.mean().plot(kind='bar', ax=ax6, color=['firebrick', 'seagreen'], rot=0)
ax6.set_title('Mean creation to launch length (days)')
ax6.set_xlabel('')

df.groupby('state').name_length.mean().plot(kind='bar', ax=ax7, color=['firebrick', 'seagreen'], rot=0)
ax7.set_title('Mean name length (words)')
ax7.set_xlabel('')

df.groupby('state').blurb_length.mean().plot(kind='bar', ax=ax8, color=['firebrick', 'seagreen'], rot=0)
ax8.set_title('Mean blurb length (words)')
ax8.set_xlabel('')

# Creating a dataframe grouped by staff_pick with columns for failed and successful
pick_df = pd.get_dummies(df.set_index('staff_pick').state).groupby('staff_pick').sum()
# Normalizes counts by column, and selects the 'True' category (iloc[1])
(pick_df.div(pick_df.sum(axis=0), axis=1)).iloc[1].plot(kind='bar', ax=ax9, color=['firebrick', 'seagreen'], rot=0) 
ax9.set_title('Proportion that are staff picks')
ax9.set_xlabel('')

fig.subplots_adjust(hspace=0.3)
plt.show()


# **How has fundraising changed over time?**

# The graph below shows the number of projects launched each month on Kickstarter from 2009 to 2020.
# - The number of projects steadily grows from when the Kickstarter was founded in 2009 up to the start of 2014.
# - From 2012 Kickstarter started expanding into other countries, having launched initially in the US.
# - Growth increased dramatically in 2014, and has remained at a high level since then.
# - Seasonality is also hinted at, with fewer projects appearing to be launched in November.
# - The second graph shows a similar story, with the cumulative amount pledged increasing more quickly from 2012 onwards.

# In[128]:


# Plotting the number of projects launched each month
plt.figure(figsize=(16,6))
df.set_index('launched_at').category.resample('MS').count().plot()
plt.xlim('2009-01-01', '2020-11-12') # Limiting to whole months
plt.xlabel('Launch date', fontsize=12)
plt.ylabel('Number of projects', fontsize=12)
plt.title('Number of projects launched on Kickstarter, 2009-2020', fontsize=16)
plt.show()


# In[129]:


# Plotting the cumulative amount pledged on Kickstarter
plt.figure(figsize=(16,6))
df.set_index('launched_at').sort_index().usd_pledged.cumsum().plot()
plt.xlim('2009-01-01', '2020-11-12') # Limiting to whole months
plt.xlabel('Launch date', fontsize=12)
plt.ylabel('Cumulative amount pledged ($)', fontsize=12)
plt.title('Cumulative pledges on Kickstarter, 2009-2020', fontsize=16)
plt.show()


# The table and graph below show the total and distribution of pledged amounts for each year 2009-2020. Again, the trend can be split into two phases, with a change in 2014. From 2014 onwards there was greater variation in the amounts pledged, with lower median amounts than the period 2009-2014, but generally higher mean amounts (with the exception of 2013) due to some very large projects.

# In[130]:


print("Average amount pledged per project in each year, in $:")
print(round(df.set_index('launched_at').usd_pledged.resample('YS').mean(),2))


# In[131]:


# Plotting the distribution of pledged amounts each year
plt.figure(figsize=(16,6))
sns.boxplot(df.launched_at.dt.year, np.log(df.usd_pledged))
plt.xlabel('Year of launch', fontsize=12)
plt.ylabel('Amount pledged (log-transformed $)', fontsize=12) # Log-transforming to make the trend clearer, as the distribution is heavily positively skewed
plt.title('Amount pledged on Kickstarter projects, 2009-2020', fontsize=16)
plt.show()


# The table and graph below are similar to the ones above, but for the goals of each project. The changes in goals show a similar pattern to the changes in the amounts pledged.

# In[132]:


print("Average fundraising goal per project in each year, in $:")
print(round(df.set_index('launched_at').usd_goal.resample('YS').mean(),2))


# In[133]:


# Plotting the distribution of goal amounts each year
plt.figure(figsize=(16,6))
sns.boxplot(df.launched_at.dt.year, np.log(df.usd_goal))
plt.xlabel('Year of launch', fontsize=12)
plt.ylabel('Goal (log-transformed $)', fontsize=12) # Log-transforming to make the trend clearer, as the distribution is heavily positively skewed
plt.title('Fundraising goals of Kickstarter projects, 2009-2020', fontsize=16)
plt.show()


# The graph below shows the number and proportion of failed and successful projects each year. Once again, there is a change from 2014. From 2009 to 2013, each year about 80% of projects were successful. However, this decreased from 2014, although since then it has mostly been rising again. 

# In[135]:


# Creating a dataframe grouped by year with columns for failed and successful
year_df = df.set_index('launched_at').state
year_df = pd.get_dummies(year_df).resample('YS').sum()
# Plotting the number and proportion of failed and successful projects each year
fig, ax = plt.subplots(1,2, figsize=(12,4))

year_df.plot.bar(ax=ax[0], color=['firebrick', 'seagreen'])
ax[0].set_title('Number of failed and successful projects')
ax[0].set_xlabel('')
ax[0].set_xticklabels(list(range(2009,2021)), rotation=45)

year_df.div(year_df.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax[1], color='seagreen') # Normalizes counts across rows
ax[1].set_title('Proportion of successful projects')
ax[1].set_xlabel('')
ax[1].set_xticklabels(list(range(2009,2021)), rotation=45)

plt.show()


# **What types of projects do people launch and which are more successful?**

# - There are 15 project categories, of which music is the most common, followed by film & video and art.
# - Technology projects have the highest goals by far (in terms of their median goal size), followed by food (e.g. funding for restaurants), with other categories generally much smaller in terms of their funding goals.
# - However, technology projects are in the lower third of the leaderboard in terms of the median amount actually pledged.
# - Comics, Games & Dance projects obtain the greatest amount of funding, on average (median).
# - The most frequently succesful categories are comics and dance (probably at least partly due to their relatively small funding goals), while the least successful are food, journalism (again, probably because of their large funding goals). 
# - Comics and games tend to attract the most backers, but each backer tends to pledge less.
# - Dance and film & video tend to attract the most generous backers.

# In[144]:


# Creating a dataframe grouped by category with columns for failed and successful
cat_df = pd.get_dummies(df.set_index('category').state).groupby('category').sum()

# Plotting
fig, ((ax1, ax2 ,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,16))

color = cm.CMRmap(np.linspace(0.1,0.8,df.category.nunique())) # Setting a colormap

df.groupby('category').category.count().plot(kind='bar', ax=ax1, color=color)
ax1.set_title('Number of projects')
ax1.set_xlabel('')

df.groupby('category').usd_goal.median().plot(kind='bar', ax=ax2, color=color)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

df.groupby('category').usd_pledged.median().plot(kind='bar', ax=ax3, color=color)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

cat_df.div(cat_df.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax4, color=color) # Normalizes counts across rows
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')

df.groupby('category').backers_count.median().plot(kind='bar', ax=ax5, color=color)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')

df.groupby('category').pledge_per_backer.median().plot(kind='bar', ax=ax6, color=color)
ax6.set_title('Median pledged per backer ($)')
ax6.set_xlabel('')

fig.subplots_adjust(hspace=0.6)
plt.show()


# **Where do project owners come from and are some countries more successful than others?**

# - The vast majority of projects are from the US, with more than six times the total number of projects compared to the second most prolific country (the UK).
# -  Poland & Switzerland has the highest median project goal size, although the differences in mean goal sizes are less extreme.
# - Projects from Greece, Slovenia & Poland are the most successful project.
# - Poland have the most backers
# - Greece, Switzerland & HK receive more per backer.

# In[147]:


# Creating a dataframe grouped by country with columns for failed and successful
country_df = pd.get_dummies(df.set_index('country').state).groupby('country').sum()

# Plotting
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16,16))

color = cm.CMRmap(np.linspace(0.1,0.8,df.country.nunique()))

df.groupby('country').country.count().plot(kind='bar', ax=ax1, color=color, rot=0)
ax1.set_title('Number of projects')
ax1.set_xlabel('')

df.groupby('country').usd_goal.median().plot(kind='bar', ax=ax2, color=color, rot=0)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

df.groupby('country').usd_pledged.median().plot(kind='bar', ax=ax3, color=color, rot=0)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

country_df.div(country_df.sum(axis=1), axis=0).successful.plot(kind='bar', ax=ax4, color=color, rot=0) # Normalizes counts across rows
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')

df.groupby('country').backers_count.median().plot(kind='bar', ax=ax5, color=color, rot=0)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')

df.groupby('country').pledge_per_backer.median().plot(kind='bar', ax=ax6, color=color, rot=0)
ax6.set_title('Median pledged per backer ($)')
ax6.set_xlabel('')

fig.subplots_adjust(hspace=0.3)
plt.show()


# **When is the best time to launch a project?**

# - Tuesday appears to be the best day to launch a project. It is the most popular launch day, and has the highest proportion of successful projects, the most backers, the highest median amount pledged per backer, and the highest median pledge amount overall.
# - Weekends are the least popular days to launch a project, attract less money, have fewer backers, receive smaller pledges per backer, and are slightly less successful. They also tend to have lower goals, making it more surprising that they tend to be less successful and receive less funding.

# In[148]:


# Creating a dataframe grouped by the day on which they were launched, with columns for failed and successful
day_df = pd.get_dummies(df.set_index('launch_day').state).groupby('launch_day').sum()

# Plotting
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14,12))

color = cm.CMRmap(np.linspace(0.1,0.8,df.launch_day.nunique()))

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df.groupby('launch_day').launch_day.count().reindex(weekdays).plot(kind='bar', ax=ax1, color=color, rot=0)
ax1.set_title('Number of projects launched')
ax1.set_xlabel('')

df.groupby('launch_day').usd_goal.median().reindex(weekdays).plot(kind='bar', ax=ax2, color=color, rot=0)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')

df.groupby('launch_day').usd_pledged.median().reindex(weekdays).plot(kind='bar', ax=ax3, color=color, rot=0)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')

day_df.div(day_df.sum(axis=1), axis=0).successful.reindex(weekdays).plot(kind='bar', ax=ax4, color=color, rot=0) # Normalizes counts across rows
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')

df.groupby('launch_day').backers_count.median().reindex(weekdays).plot(kind='bar', ax=ax5, color=color, rot=0)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')

df.groupby('launch_day').pledge_per_backer.median().reindex(weekdays).plot(kind='bar', ax=ax6, color=color, rot=0)
ax6.set_title('Median pledged per backer ($)')
ax6.set_xlabel('')

fig.subplots_adjust(hspace=0.3)
plt.show()


# - The most popular month to launch a project is July, and the least common is December.
# - Interestingly, both months have the lowest success rates, lowest median pledge amounts, lowest median backers per project and lowest median amount pledged per backer.
# - Median goal sizes are roughly similar throughout most of the year, but smaller for projects launched in January.
# - The best month to launch in is Sep & Oct, which has the highest median amount pledged per project, the highest success rate and the highest number of backers per project.

# In[149]:


# Creating a dataframe grouped by the month in which they were launched, with columns for failed and successful
month_df = pd.get_dummies(df.set_index('launch_month').state).groupby('launch_month').sum()
month_df


# In[150]:


# Plotting
months = list(calendar.month_name)[1:]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14,12))

color = cm.CMRmap(np.linspace(0.1,0.8,df.launch_month.nunique()))

df.groupby('launch_month').launch_month.count().reindex(months).plot(kind='bar', ax=ax1, color=color, rot=45)
ax1.set_title('Number of projects launched')
ax1.set_xlabel('')
ax1.set_xticklabels(labels=ax1.get_xticklabels(), ha='right')

df.groupby('launch_month').usd_goal.median().reindex(months).plot(kind='bar', ax=ax2, color=color, rot=45)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')
ax2.set_xticklabels(labels=ax2.get_xticklabels(), ha='right')

df.groupby('launch_month').usd_pledged.median().reindex(months).plot(kind='bar', ax=ax3, color=color, rot=45)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')
ax3.set_xticklabels(labels=ax3.get_xticklabels(), ha='right')

month_df.div(month_df.sum(axis=1), axis=0).successful.reindex(months).plot(kind='bar', ax=ax4, color=color, rot=45) # Normalizes counts across rows
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')
ax4.set_xticklabels(labels=ax4.get_xticklabels(), ha='right')

df.groupby('launch_month').backers_count.median().reindex(months).plot(kind='bar', ax=ax5, color=color, rot=45)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')
ax5.set_xticklabels(labels=ax5.get_xticklabels(), ha='right')

df.groupby('launch_month').pledge_per_backer.median().reindex(months).plot(kind='bar', ax=ax6, color=color, rot=45)
ax6.set_title('Median pledged per backer ($)')
ax6.set_xlabel('')
ax6.set_xticklabels(labels=ax6.get_xticklabels(), ha='right')

fig.subplots_adjust(hspace=0.4)
plt.show()


# - Unsurprisingly, the number of projects launched peaks during the day in the US (the times below are in UTC/GMT, so e.g. 2pm-4pm in the chart is equal to 9am-11am EST).
# - More surprisingly, the median amount pledged per project and per backer does vary considerably throughout the day, with projects launched at 12pm-2pm UTC (7am-9am EST) attracting more funding and backers, on average (median), and being more likely to be successful.

# In[151]:


# Creating a dataframe grouped by the time at which they were launched, with columns for failed and successful
time_df = pd.get_dummies(df.set_index('launch_time').state).groupby('launch_time').sum()

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14,12))

color = cm.CMRmap(np.linspace(0.1,0.8,df.launch_time.nunique()))

times = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am']

df.groupby('launch_time').launch_time.count().reindex(times).plot(kind='bar', ax=ax1, color=color, rot=45)
ax1.set_title('Number of projects launched')
ax1.set_xlabel('')
ax1.set_xticklabels(labels=ax1.get_xticklabels(), ha='right')

df.groupby('launch_time').usd_goal.median().reindex(times).plot(kind='bar', ax=ax2, color=color, rot=45)
ax2.set_title('Median project goal ($)')
ax2.set_xlabel('')
ax2.set_xticklabels(labels=ax2.get_xticklabels(), ha='right')

df.groupby('launch_time').usd_pledged.median().reindex(times).plot(kind='bar', ax=ax3, color=color, rot=45)
ax3.set_title('Median pledged per project ($)')
ax3.set_xlabel('')
ax3.set_xticklabels(labels=ax3.get_xticklabels(), ha='right')

time_df.div(time_df.sum(axis=1), axis=0).successful.reindex(times).plot(kind='bar', ax=ax4, color=color, rot=45) # Normalizes counts across rows
ax4.set_title('Proportion of successful projects')
ax4.set_xlabel('')
ax4.set_xticklabels(labels=ax4.get_xticklabels(), ha='right')

df.groupby('launch_time').backers_count.median().reindex(times).plot(kind='bar', ax=ax5, color=color, rot=45)
ax5.set_title('Median backers per project')
ax5.set_xlabel('')
ax5.set_xticklabels(labels=ax5.get_xticklabels(), ha='right')

df.groupby('launch_time').pledge_per_backer.median().reindex(times).plot(kind='bar', ax=ax6, color=color, rot=45)
ax6.set_title('Median pledged per backer ($)')
ax6.set_xlabel('')
ax6.set_xticklabels(labels=ax6.get_xticklabels(), ha='right')

fig.subplots_adjust(hspace=0.45)
plt.show()


# **Checking distributions**

# Most continuous numerical features other than blurb_length and campaign_days are heavily positively skewed. This is not an issue for some machine learning models, so these features will not be log-transformed for the first few models. After that, models will be ru-run using log-transformed data, to see whether this improves model accuracy.

# In[152]:


# Checking the distributions of continuous features
df[df.describe().columns].hist(figsize=(16,10));


# **Preparing the data for machine learning**

# Some features were retained for EDA purposes, but now need to be dropped in order to use machine learning models. This includes datetime features, features that are related to outcomes (e.g. the amount pledged and the number of backers) rather than related to the properties of the project itself (e.g. category, goal, length of campaign), categorical features which would result in too many one-hot encoded features (sub_category), and features that only have one category (is_starrable).

# In[158]:


# Dropping columns and creating new dataframe
df_transformed = df.drop(['backers_count', 'created_at', 'deadline', 'is_starrable', 'launched_at', 'usd_pledged', 'sub_category', 'pledge_per_backer'], axis=1)
df_transformed.head()


# Multi-collinearity will be checked for by assessing correlations between predictor features, as this can cause issues with some models. The multi-collinearity matrix below shows that this is not an issue:

# In[157]:


# Set the style of the visualization
sns.set(style="white")

# Create a covariance matrix
corr = df_transformed.corr()

# Generate a mask the size of our covariance matrix
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (16,16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5});


# The dependent variable will now be converted into 1s (successful) and 0s (failure):

# In[159]:


df_transformed['state'] = df_transformed['state'].replace({'failed': 0, 'successful': 1})


# Categorical features will now be one-hot encoded:

# In[160]:


# Converting boolean features to string to include them in one-hot encoding
df_transformed['staff_pick'] = df_transformed['staff_pick'].astype(str)


# In[163]:


df_transformed.head()


# In[297]:


# Creating dummy variables
df_transformed = pd.get_dummies(df_transformed)


# In[298]:


df_transformed.head()


# Finally, the dependent (y) and independent (X) features will be separated into separate datasets. Because the features are on different scales, independent features will be transformed and normalised using StandardScaler.

# In[299]:


X_unscaled = df_transformed.drop('state', axis=1)
y = df_transformed.state


# In[300]:


# Transforming the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_unscaled), columns=list(X_unscaled.columns))
X.head()


# ***
# # Modeling

# In this section, three different machine learning models for classification will be applied to the data, in order to create a model to classify projects into successes and failures.
# 
# The two categories are of a roughly equal size, so no measures need to be taken to adjust for inbalanced classes (e.g. SMOTE).

# In[301]:


# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)


# It is good practice to choose an evaluation method before running machine learning models - not after. The weighted average F1 score was chosen. The F1 score calculates the harmonic mean between precision and recall, and is a suitable measure because there is no preference for false positives or false negatives in this case (both are equally bad). The weighted average will be used because the classes are of slightly different sizes, and we want to be able to predict both successes and failures.

# ### Model 1: vanilla logistic regression

# Logistic regression can be used as a binary classifier in order to predict which of two categories a data point falls in to. To create a baseline model to improve upon, a logistic regression model will be fitted to the data, with default parameters.

# In[302]:


# Fitting a logistic regression model with default parameters
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[303]:


# Making predictions
y_hat_train = logreg.predict(X_train)
y_hat_test = logreg.predict(X_test)


# In[304]:


# Logistic regression scores
print("Logistic regression score for training set:", round(logreg.score(X_train, y_train),5))
print("Logistic regression score for test set:", round(logreg.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, y_hat_test))


# In[305]:


def plot_cf(y_true, y_pred, class_names=None, model_name=None):
    """Plots a confusion matrix"""
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)
    plt.grid(b=None)
    if model_name:
        plt.title("Confusion Matrix: {}".format(model_name))
    else:
        plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    class_names = set(y_true)
    tick_marks = np.arange(len(class_names))
    if class_names:
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    
    thresh = cf.max() / 2.
    
    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, cf[i, j], horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')

    plt.colorbar()


# In[306]:


# Confusion matrix
plot_cf(y_test, y_hat_test)


# In[307]:


# Plotting the AUC-ROC
y_score = logreg.fit(X_train, y_train).decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)

print('AUC:', round(auc(fpr, tpr),5))

plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# The logistic regression model has a fairly good accuracy score of around 0.78 (weighted average F1 score), with similar results for the test and train set. However, it is worse at predicting failures compared to successes, and the recall rate (ability to correctly predict positives out of all of the actual positives in the data) is notably different between the failure and success categories. The AUC value is pretty high, and the curve is pulled towards the top left of the graph, which is a positive sign. However, this can probably be improved upon.

# In[ ]:


###Feature Selection Strategy


# In[308]:


X.head()


# In[309]:


X.dtypes


# In[310]:


varSel = pd.DataFrame({'Variable': X.columns[0:137]})
varSel.describe


# In[311]:


#from importlib import reload
from pyMechkar.analysis import Table1
#reload(tb1)


# In[315]:


nm = df_transformed.columns[1:137]
nm = nm.append(pd.Index(['state']))
nm


# In[316]:


df2 = df_transformed[nm]
df2.head()


# In[317]:


tab1 = Table1(data=df2, y = "state")


# In[318]:


tab1


# In[319]:


tab1[tab1['p_value']<0.05]


# In[320]:


vn1 = tab1.loc[tab1['p_value']<0.05,'Variables'].unique()
print(len(vn1))
vn1


# In[321]:


#We will add these variables to our variable selection table
varSel['Univarable'] = 0
varSel.loc[varSel['Variable'].isin(vn1), 'Univarable'] = 1
varSel


# In[322]:


## Multivariable Analysis
### drop na
df2 = df2.dropna()


# In[323]:


## remove unnecessary vars
Xs = df2.iloc[:,:]
Xs.head()


# In[324]:


y = df2.iloc[:,-1:]
print([Xs.shape,y.shape])


# In[377]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lassomod = Lasso(alpha=0.01).fit(Xs, y)


# In[378]:


model = SelectFromModel(lassomod, prefit=True)
model.get_support()


# In[379]:


varSel['Lasso'] = model.get_support().astype('int64')
varSel


# In[ ]:


### Variable Selection using Random Forest


# In[380]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rfmod = RandomForestClassifier().fit(Xs, y)
#rfmod.feature_importances_ 


# In[381]:


model = SelectFromModel(rfmod, prefit=True)
model.get_support()


# In[382]:


varSel['RandomForest'] = model.get_support().astype('int64')
varSel


# In[ ]:


### Variable Selection using Gradient Boosting classification


# In[383]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel

gbmod = GradientBoostingClassifier().fit(X, y)


# In[384]:


model = SelectFromModel(gbmod, prefit=True)
model.get_support()


# In[385]:


varSel['GradientBoost'] = model.get_support().astype('int64')
varSel


# In[ ]:


### Variable Selection using SVM classification


# In[386]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

svmmod = LinearSVC(C=0.01, penalty="l1",dual=False).fit(X, y)


# In[387]:


model = SelectFromModel(svmmod, prefit=True)
model.get_support()


# In[388]:


varSel['SVM'] = model.get_support().astype('int64')
varSel


# In[ ]:


### Summarization and Selection of Variables 


# In[389]:


varSel['Sum'] =  np.sum(varSel,axis=1)
varSel


# In[390]:


varSel.groupby('Sum')['Variable'].count()


# In[394]:


#We can now decide a threshold for selecting our variables!


# In[392]:


varSel[varSel['Sum']>=3]


# In[101]:


### Principal Component Analysis and feature weightings


# In[103]:


##There are a large number of features in the dataset. PCA (Principal Component Analysis) can be used to reduce this into a smaller number of components which still explain as much variation in the data as possible.


# In[325]:


pca = PCA()
pca.fit_transform(X)
explained_var = np.cumsum(pca.explained_variance_ratio_)


# ##Plotting the amount of variation explained by PCA with different numbers of components

# In[326]:


# Plotting the amount of variation explained by PCA with different numbers of components
plt.plot(list(range(1, len(explained_var)+1)), explained_var)
plt.title('Amount of variation explained by PCA', fontsize=14)
plt.xlabel('Number of components')
plt.ylabel('Explained variance');


# There is no obvious elbow in this plot, so three different values for n_components will be tried below with logistic regression to see which produces the best score.

# In[327]:


print("Number of components explaining 80% of variance:", np.where(explained_var > 0.8)[0][0])
print("Number of components explaining 90% of variance:", np.where(explained_var > 0.9)[0][0])
print("Number of components explaining 99% of variance:", np.where(explained_var > 0.99)[0][0])


# The number of components to be used will be assessed by running logistic regression models using each of the three numbers of components.

# In[328]:


n_comps = [58,71,94]
for n in n_comps:
    pipe = Pipeline([('pca', PCA(n_components=n)), ('clf', LogisticRegression())])
    pipe.fit(X_train, y_train)
    print("\nNumber of components:", n)
    print("Score:", round(pipe.score(X_test, y_test),5))


# The above results show that the score is highest for 94 components, although the difference is small (c. 6% improvement from 58 components).

# In[329]:


# Feature weightings on each component, in order of average weighting
pca = PCA(n_components=94)
pca.fit_transform(X)
pca_94_components = pd.DataFrame(pca.components_,columns=X.columns).T # Components as columns, features as rows
pca_94_components['mean_weight'] = pca_94_components.iloc[:].abs().mean(axis=1)
pca_94_components.sort_values('mean_weight', ascending=False)


# The graph below plots the average weight of each feature on each component. It shows that there is relatively little variation between the average weights of each feature, i.e. how much each feature is included in each component.

# In[330]:


# Plotting feature importances
plt.figure(figsize=(18,6))
pca_94_components.mean_weight.sort_values(ascending=False).plot(kind='bar',color=color)
plt.show()


# The tables below show the top 10 most important features in the top three most important components.
# 
# - Component 1 - the top two features relate to the country a project is from, primarily the US and the UK (the top two most common countries).
# - Component 2 - the top two features relate to whether or not a project was highlighted as a staff pick.
# - Component 3 - the top two features relate to the timings of the project, specifically whether it was launched in October or had a deadline in November.

# In[374]:


pca_94_components[0].map(lambda x : x).abs().sort_values(ascending = False)[:8]


# In[375]:


pca_94_components[1].map(lambda x : x).abs().sort_values(ascending = False)[:8]


# In[376]:


pca_94_components[2].map(lambda x : x).abs().sort_values(ascending = False)[:8]


# ### Model 2: logistic regression with PCA and parameter optimisation

# The logistic regression model can potentially be further improved by optimising its parameters. GridSearchCV can be used to test multiple different regularisation parameters (values of C), penalties (l1 or l2) and models with and without an intercept.

# In[334]:


# Using GridSearchCV to test multiple different parameters
logreg_start = time.time()

pipe_logreg = Pipeline([('pca', PCA(n_components=90)),
                    ('clf', LogisticRegression())])

params_logreg = [
    {'clf__penalty': ['l1', 'l2'],
     'clf__fit_intercept': [True, False],
        'clf__C': [0.001, 0.01, 1, 10]
    }
]

grid_logreg = GridSearchCV(estimator=pipe_logreg,
                  param_grid=params_logreg,
                  cv=5)

grid_logreg.fit(X_train, y_train)

logreg_end = time.time()

logreg_best_score = grid_logreg.best_score_
logreg_best_params = grid_logreg.best_params_

print("Results from the logistic regression parameter optimisation:")
print(f"Time taken to run: {round((logreg_end - logreg_start)/60,1)} minutes")
print("Best accuracy:", round(logreg_best_score,2))
print("Best parameters:", logreg_best_params)


# #### 
# Results from the logistic regression parameter optimisation:
# 
# #- Time taken to run: 4.7 minutes
# 
# #- Best accuracy: 0.72
# 
# #- Best parameters: {'clf__C': 10, 'clf__fit_intercept': True, 'clf__penalty': 'l2'}

# **Best logistic regression model**

# In[335]:


pipe_best_logreg = Pipeline([('pca', PCA(n_components=94)),
                    ('clf', LogisticRegression(C=10, fit_intercept=True, penalty='l2'))])

pipe_best_logreg.fit(X_train, y_train)

lr_y_hat_train = pipe_best_logreg.predict(X_train)
lr_y_hat_test = pipe_best_logreg.predict(X_test)

print("Logistic regression score for training set:", round(pipe_best_logreg.score(X_train, y_train),5))
print("Logistic regression score for test set:", round(pipe_best_logreg.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, lr_y_hat_test))
plot_cf(y_test, lr_y_hat_test)


# After hyperparameter tuning, the model's accuracy score is the same as the logistic regression model using default parameters (0.70 weighted average F1 score).

# ### Model 3: Random Forests

# The Random Forest algorithm is a supervised learning algorithm that can be used for classification. It works by building multiple different decision trees to predict which category a data point belongs to.

# In[336]:


# Random Forests with default settings
pipe_rf = Pipeline([('pca', PCA(n_components=92)),
                    ('clf', RandomForestClassifier())])
pipe_rf.fit(X_train, y_train)
print("Score:", round(pipe_rf.score(X_test, y_test),5))


# In[337]:


# Reporting the depths of each tree in the model created by the default Random Forest classifier, to get a sense of the
# maximum depth that a tree can be if depth is not limited, to help with choosing parameters to try with GridSearchCV
[estimator.tree_.max_depth for estimator in pipe_rf.named_steps['clf'].estimators_]


# In[338]:


# Using GridSearchCV to test multiple different parameters
rf_start = time.time()

pipe_rf = Pipeline([('pca', PCA(n_components=90)),
                    ('clf', RandomForestClassifier())])

params_rf = [ 
  {'clf__n_estimators': [100],
   'clf__max_depth': [20, 30, 40],    
   'clf__min_samples_split':[0.001, 0.01]
  }
]

grid_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params_rf,
                  cv=5)

grid_rf.fit(X_train, y_train)

rf_end = time.time()

rf_best_score = grid_rf.best_score_
rf_best_params = grid_rf.best_params_

print("Results from the Random Forest parameter optimisation:")
print(f"Time taken to run: {round((rf_end - rf_start)/60,1)} minutes")
print("Best accuracy:", round(rf_best_score,2))
print("Best parameters:", rf_best_params)


# **Best Random Forest model**

# In[339]:


pipe_best_rf = Pipeline([('pca', PCA(n_components=92)),
                    ('clf', RandomForestClassifier(max_depth=30, min_samples_split=0.001, n_estimators=100))])

pipe_best_rf.fit(X_train, y_train)

rf_y_hat_train = pipe_best_rf.predict(X_train)
rf_y_hat_test = pipe_best_rf.predict(X_test)

print("Random Forest score for training set:", round(pipe_best_rf.score(X_train, y_train),5))
print("Random Forest score for test set:", round(pipe_best_rf.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, rf_y_hat_test))
plot_cf(y_test, rf_y_hat_test)


# 
# After hyperparameter tuning, the model's accuracy score has increased from 0.65 to 0.69 (weighted average f1 score). However, the difference between the accuracy score for the training set and the test set suggests there might be some over-fitting.

# ### Model 4: XGBoost

# XGBoost is a form of gradient boosting algorithm. Similar to Random Forests, it is an ensemble method that produces multiple decision trees to improve classification of data points, but it uses gradient descent to improve the performance of the model for the data points that are particularly difficult to classify.

# In[340]:


# XGBoost with default settings
pipe_xgb = Pipeline([('pca', PCA(n_components=92)),
                    ('clf', xgb.XGBClassifier())])
pipe_xgb.fit(X_train, y_train)
print("Score:", round(pipe_xgb.score(X_test, y_test),5))


# In[341]:


## Using GridSearchCV to test multiple different parameters
xgb_start = time.time()

pipe_xgb = Pipeline([('pca', PCA(n_components=92)),
                    ('clf', xgb.XGBClassifier())])

params_xgb = [ 
  {'clf__n_estimators': [100],
   'clf__max_depth': [25, 35],
   'clf__learning_rate': [0.01, 0.1],
   'clf__subsample': [0.7, 1],
   'clf__min_child_weight': [20, 100]
  }
]

grid_xgb = GridSearchCV(estimator=pipe_xgb,
                  param_grid=params_xgb,
                  cv=5)

grid_xgb.fit(X_train, y_train)

xgb_end = time.time()

xgb_best_score = grid_xgb.best_score_
xgb_best_params = grid_xgb.best_params_

print(f"Time taken to run: {round((xgb_end - xgb_start)/60,1)} minutes")
print("Best accuracy:", round(xgb_best_score,2))
print("Best parameters:", xgb_best_params)


# Results from the XGBoost parameter optimisation:
# - Time taken to run: 172.5 minutes (3 hours)
# - Best accuracy: 0.71
# - Best parameters: {'clf__learning_rate': 0.1, 'clf__max_depth': 35, 'clf__min_child_weight': 100, 'clf__n_estimators': 100, - 'clf__subsample': 1}

# **Best XGBoost model**

# In[342]:


pipe_best_xgb = Pipeline([('pca', PCA(n_components=90)),
                    ('clf', xgb.XGBClassifier(learning_rate=0.1, max_depth=35, min_child_weight=100, n_estimators=100, subsample=0.7))])

pipe_best_xgb.fit(X_train, y_train)

xgb_y_hat_train = pipe_best_xgb.predict(X_train)
xgb_y_hat_test = pipe_best_xgb.predict(X_test)

print("XGBoost score for training set:", round(pipe_best_xgb.score(X_train, y_train),5))
print("XGBoost score for test set:", round(pipe_best_xgb.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, xgb_y_hat_test))
plot_cf(y_test, xgb_y_hat_test)


# After hyperparameter tuning, the model's accuracy has increased from 0.69 to 0.7, although this is a very small increase for a very computationally-expensive hyperparameter tuning process. There also appears to be some over-fitting, from the difference in train and test scores.

# ### Model 5: vanilla logistic regression with log-transformed data

# Previously, positively skewed data was not log-transformed. Now a log transformation will be applied to the skewed features, and a logistic regression model fitted again to see whether this improved accuracy.

# In[343]:


# Assessing skewed distributions
cols_to_log = ['creation_to_launch_days', 'name_length', 'usd_goal']
df_transformed[cols_to_log].hist(figsize=(8,6));


# In[344]:


# Replacing 0s with 0.01 and log-transforming
for col in cols_to_log:
    df_transformed[col] = df_transformed[col].astype('float64').replace(0.0, 0.01)
    df_transformed[col] = np.log(df_transformed[col])


# In[345]:


# Checking new distributions
df_transformed[cols_to_log].hist(figsize=(8,6));


# In[346]:


df_transformed.head()


# Now the data can be prepared again for machine learning by separating X and y, and scaling:

# In[347]:


X_unscaled_log = df_transformed.drop('state', axis=1)
y_log = df_transformed.state


# In[348]:


# Transforming the data
scaler = StandardScaler()
X_log = pd.DataFrame(scaler.fit_transform(X_unscaled_log), columns=list(X_unscaled_log.columns))
X_log.head()


# In[349]:


# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.3, random_state=123)


# In[350]:


# Fitting a logistic regression model with default parameters
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

# Making predictions
lr_y_hat_train2 = logreg.predict(X_train)
lr_y_hat_test2 = logreg.predict(X_test)

# Logistic regression scores
print("Logistic regression score for training set:", round(logreg.score(X_train, y_train),5))
print("Logistic regression score for test set:", round(logreg.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, lr_y_hat_test2))
# Confusion matrix
plot_cf(y_test, lr_y_hat_test2)


# Log-transforming the data has increased the accuracy by 0.03 to 0.73.

# ### Model 6: Random Forests without PCA

# A Random Forest model will now be fitted using the log-transformed data, but without using PCA. This is to allow for the construction of a list of feature importances.

# In[162]:


# Using GridSearchCV to test multiple different parameters
rf_start2 = time.time()

rf2 = RandomForestClassifier(min_samples_split=0.001, verbose=2)

params_rf2 = [ 
  {'n_estimators': [200, 400],
   'max_depth': [20, 35]
  }
]

grid_rf2 = GridSearchCV(estimator=rf2, param_grid=params_rf2, cv=5)

grid_rf2.fit(X_train, y_train)

rf_end2 = time.time()

rf_best_score2 = grid_rf2.best_score_
rf_best_params2 = grid_rf2.best_params_

print(f"Time taken to run: {round((rf_end2 - rf_start2)/60,1)} minutes")
print("Best accuracy:", round(rf_best_score2,2))
print("Best parameters:", rf_best_params2)


# Results:
# - Time taken to run: 19.2 minutes
# - Best accuracy: 0.75
# - Best parameters: {'max_depth': 35, 'n_estimators': 400}

# In[351]:


best_rf = RandomForestClassifier(max_depth=35, min_samples_split=0.001, n_estimators=400)

best_rf.fit(X_train, y_train)

rf_y_hat_train2 = best_rf.predict(X_train)
rf_y_hat_test2 = best_rf.predict(X_test)

print("Random Forest score for training set:", round(best_rf.score(X_train, y_train),5))
print("Random Forest score for test set:", round(best_rf.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, rf_y_hat_test2))
plot_cf(y_test, rf_y_hat_test2)


# By log-transforming features and increasing the number of trees created (n_estimators), it was possible to improve the weighted average F1 score to 0.74. Overfitting does not appear to be an issue.
# 
# Because PCA was not used, it was possible to plot feature importance (see graph below).
# - Goal size is the most important feature, followed by the number of days taken from project creation to launch, and whether or not the project was a staff pick.
# - Campaign length and name length were also fairly important.
# - Project type (category) was less important, although whether or not a project was a technology and food project does seem to be fairly important.
# - Launch and deadline time, day and month is not very important.
# - Country of origin is not very important.

# In[352]:


# Plotting feature importance
n_features = X_train.shape[1]
plt.figure(figsize=(8,20))
plt.barh(range(n_features), best_rf.feature_importances_, align='center') 
plt.yticks(np.arange(n_features), X_train.columns.values) 
plt.title("Feature importances in the best Random Forest model")
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


# ### Model 7: XGBoost without PCA

# An XGBoost model will now be fitted using the log-transformed data, but without using PCA. This is to attempt to improve upon the Random Forest model, and to see whether the feature importances are similar.

# In[353]:


# Using GridSearchCV to test multiple different parameters
xgb_start2 = time.time()

xgb2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=35, verbose=2)#max_depth=35

params_xgb2 = [ 
  {'n_estimators': [100, 200],
   'subsample': [0.7, 0.9],
   'min_child_weight': [100, 200]
  }
]

grid_xgb2 = GridSearchCV(estimator=xgb2, param_grid=params_xgb2, cv=5)

grid_xgb2.fit(X_train, y_train)

xgb_end2 = time.time()

xgb_best_score2 = grid_xgb2.best_score_
xgb_best_params2 = grid_xgb2.best_params_

print(f"Time taken to run: {round((xgb_end2 - xgb_start2)/60,1)} minutes")
print("Best accuracy:", round(xgb_best_score2,2))
print("Best parameters:", xgb_best_params2)


# Results:
# - Time taken to run: 52.9 minutes
# - Best accuracy: 0.76
# - Best parameters: {'min_child_weight': 100, 'n_estimators': 100, 'subsample': 0.7}

# In[355]:


best_xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=35, min_child_weight=100, n_estimators=100, subsample=0.7)

best_xgb.fit(X_train, y_train)

xgb_y_hat_train2 = best_xgb.predict(X_train)
xgb_y_hat_test2 = best_xgb.predict(X_test)

print("XGBoost score for training set:", round(best_xgb.score(X_train, y_train),5))
print("XGBoost score for test set:", round(best_xgb.score(X_test, y_test),5))
print("\nClassification report:")
print(classification_report(y_test, xgb_y_hat_test2))
plot_cf(y_test, xgb_y_hat_test2)


# By log-transforming features it was possible to improve the weighted average F1 score to 0.75, which makes it the highest performing model so far. Overfitting does not appear to be an issue.
# 
# The graph below shows the feature importances for the model.
# - Goal size is the most important feature, followed by the number of days taken from project creation to launch, followed by the name and blurb lengths.
# - Campaign length was also fairly important.
# - Most other categories were relatively unimportant, although project categories and whether the project was from the US are slightly important.

# In[356]:


# Plotting feature importance
n_features = X_train.shape[1]
plt.figure(figsize=(8,20))
plt.barh(range(n_features), best_xgb.feature_importances_, align='center') 
plt.yticks(np.arange(n_features), X_train.columns.values) 
plt.title("Feature importances in the best XGBoost model")
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


# Plotting the feature importances for the Random Forest model and the XGBoost model side-by-side shows the differences between them (see graph below).
# - Goal size followed by the number of days from creation to launch are the two most important features in each model.
# - The main difference is in the importance of whether or not a project was a staff pick - this was very important in the Random Forest model, and not important at all in the XGBoost model.
# - There are some other differences, including category being more important in Random Forests, and name and blurb length being more important in XGBoost.
# - Overall, the Random Forest model seems to consider more features more important, whereas XGBoost depends more heavily on only five features.

# In[373]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,22))
n_features = X_train.shape[1]
ax1.barh(range(n_features), best_rf.feature_importances_, align='center')
ax1.set_yticks(np.arange(n_features))
ax1.set_yticklabels(X_train.columns.values) 
ax1.set_xlabel('Feature importance')
ax1.set_title('Random Forest', fontsize=14)
ax2.barh(range(n_features), best_xgb.feature_importances_, align='center') 
ax2.set_yticks(np.arange(n_features))
ax2.set_yticklabels(X_train.columns.values)
ax2.set_xlabel('Feature importance')
ax2.set_title('XGBoost', fontsize=14)
fig.subplots_adjust(wspace=0.5);


# ### Additional investigation of PCA

# The graphs of feature importances above indicate that launch and deadline months, days and times are not very important to either model. Filtering out unimportant feature prior to PCA might help create a more effective PCA.

# In[358]:


# Dropping columns beginning with 'deadline'
X_filtered = X_log[[c for c in X_log.columns if c[:8] != 'deadline']]

# Dropping columns beginning with 'launch'
X_filtered = X_filtered[[c for c in X_filtered.columns if c[:6] != 'launch']]

X_filtered.head()


# In[359]:


# Conducting PCA
pca = PCA()
principal_comps = pca.fit_transform(X_filtered)
explained_var = np.cumsum(pca.explained_variance_ratio_)

# Plotting the amount of variation explained by PCA with different numbers of components
plt.plot(list(range(1, len(explained_var)+1)), explained_var)
plt.title('Amount of variation explained by PCA', fontsize=14)
plt.xlabel('Number of components')
plt.ylabel('Explained variance');


# Unfortunately, this appears to show the same pattern as the previous PCA analysis, with no clear 'elbow' point and without the first few components explaining the majority of variation.

# In[361]:


# Creating a list of PCA column names
pca_columns = []
for i in range(1,76):
    pca_columns.append("PC"+str(i))

# Creating a dataframe of principal components
principal_comps_df = pd.DataFrame(principal_comps, columns=pca_columns)


# In[362]:


# Adding target (success/fail) to the principal components dataframe
principal_comps_df = pd.concat([principal_comps_df, y_log.reset_index()], axis=1)
principal_comps_df.drop('id', inplace=True, axis=1)
principal_comps_df.head()


# In[363]:


# Plotting the first two principal components, coloured by target
plt.figure(figsize=(8,6))
sns.scatterplot(x=principal_comps_df.PC1, y=principal_comps_df.PC2, data=principal_comps_df, hue='state')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# The graph above reiterates the point that even the first two components do not explain much variation alone. There is also an interesting pattern in the data, although the reason is unclear.
# 
# As a final experiment, countries will also be dropped from the dataframe, as they also do not explain much variance.

# In[364]:


# Dropping columns beginning with 'country'
X_filtered = X_filtered[[c for c in X_filtered.columns if c[:7] != 'country']]

X_filtered.head()


# In[365]:


# Conducting PCA
pca = PCA()
principal_comps = pca.fit_transform(X_filtered)
explained_var = np.cumsum(pca.explained_variance_ratio_)

# Plotting the amount of variation explained by PCA with different numbers of components
plt.plot(list(range(1, len(explained_var)+1)), explained_var)
plt.title('Amount of variation explained by PCA', fontsize=14)
plt.xlabel('Number of components')
plt.ylabel('Explained variance');


# Unfortunately this shows a similar pattern.

# ### Potential directions for future work

# Potential additional improvements to the models include:
# - Further explanation of PCA, or possibly using the original PCA but with 58 components instead, as this only had slightly lower accuracy scores in the logistic regression test case than using 90 components
# - Further tuning of Random Forest and XGBoost parameters

# ***
# # Conclusions and recommendations

# ### Choosing a final model

# In[366]:


# Extracting weighted average precision, recall and test scores for each best model
# Logistic regression
lr_test_precision, lr_test_recall, lr_test_f1score, lr_test_support = precision_recall_fscore_support(y_test, lr_y_hat_test2, average='weighted')
lr_train_precision, lr_train_recall, lr_train_f1score, lr_train_support = precision_recall_fscore_support(y_train, lr_y_hat_train2, average='weighted')
# Random Forest
rf_test_precision, rf_test_recall, rf_test_f1score, rf_test_support = precision_recall_fscore_support(y_test, rf_y_hat_test2, average='weighted')
rf_train_precision, rf_train_recall, rf_train_f1score, rf_train_support = precision_recall_fscore_support(y_train, rf_y_hat_train2, average='weighted')
# XGBoost
xgb_test_precision, xgb_test_recall, xgb_test_f1score, xgb_test_support = precision_recall_fscore_support(y_test, xgb_y_hat_test2, average='weighted')
xgb_train_precision, xgb_train_recall, xgb_train_f1score, xgb_train_support = precision_recall_fscore_support(y_train, xgb_y_hat_train2, average='weighted')


# In[367]:


# Logistic regression results
lr_results = {'Precision':[lr_train_precision, lr_test_precision], 'Recall':[lr_train_recall, lr_test_recall], 'F1_score': [lr_train_f1score, lr_test_f1score]}
lr_results = pd.DataFrame(lr_results, index=['Train', 'Test'])
print("Best logistic regression results (Model 5):")
lr_results


# In[368]:


# Random Forest results
rf_results = {'Precision':[rf_train_precision, rf_test_precision], 'Recall':[rf_train_recall, rf_test_recall], 'F1_score': [rf_train_f1score, rf_test_f1score]}
rf_results = pd.DataFrame(rf_results, index=['Train', 'Test'])
print("Best Random Forest results (Model 6):")
rf_results


# In[369]:


# XGBoost results
xgb_results = {'Precision':[xgb_train_precision, xgb_test_precision], 'Recall':[xgb_train_recall, xgb_test_recall], 'F1_score': [xgb_train_f1score, xgb_test_f1score]}
xgb_results = pd.DataFrame(xgb_results, index=['Train', 'Test'])
print("Best XGBoost results (Model 7):")
xgb_results


# ### Final model evaluation and interpretation

# Each model was able to achieve an accuracy of 73-75% after parameter tuning. Although it was relatively easy to reach an accuracy level of about 70% for each model, parameter tuning and other adjustments were only able to increase accuracy levels by a small amount. Possibly the  large amount of data for each of only two categories meant that there was enough data for even a relatively simple model (e.g. logistic regression with default settings) to achieve a good level of accuracy.
# 
# The final chosen model is the tuned XGBoost model, which had the highest test set weighted average F1 score of 0.747.
# 
# Interestingly, each model performed worse at predicting failures compared to successes, with a lower true negative rate than true positive rate (see calculations below). I.e. it classified quite a few failed projects as successes, but relatively few successful projects as failures. Possibly the factors that might cause a project to fail are more likely to be beyond the scope of the data, e.g. poor marketing, insufficient updates, or not replying to messages from potential backers.
# 
# The false positive and false negative rates mean that, if the data about a new project is fed through the model to make a prediction about its success or failure:
# - if the project is going to end up being a success, the model will correctly predict this as a success about 80% of the time
# - if the project is going to end up being a failure, the model will only correctly predict this as a failure about 65% of the time, and the rest of the time will incorrectly predict it as a success

# In[370]:


r_cf = confusion_matrix(y_test, xgb_y_hat_test2)
print("Evaluation of the final model:")
print("\nIf the true value is failure, what proportion does the model correctly predict as a failure? \n(True negative rate/specificity):\n", round(r_cf[0][0]/sum(r_cf[0]),4))
print("If the true value is success, what proportion does the model correctly predict as a success? \n(True positive rate/recall/sensitivity):\n", round(r_cf[1][1]/sum(r_cf[1]),4))
print("\nIf the model predicts a failure, what proportion are actually failures? \n(Negative prediction value):\n", round(r_cf[0][0]/sum(r_cf[:,0]),4))
print("If the model predicts a success, what proportion are actually successes? \n(Positive prediction value/precision):\n", round(r_cf[1][1]/sum(r_cf[:,1]),4))


# ### Recommendations

# Some of the factors that had a **positive effect** on success rate and/or the amount of money received (as deduced from a mixture of data exploration and model feature importances) are:
# 
# **Most important:**
# - Smaller project goals
# - Being chosen as a staff pick (a measure of quality)
# - Shorter campaigns
# - Taking longer between creation and launch
# - Comics, dance and games projects
# 
# **Less important:**
# - Projects from Hong Kong
# - Film & video and music projects are popular categories on the site, and are fairly successful
# - Launching on a Tuesday (although this is also the most common day to launch a project, so beware the competition)
# - Launching in October
# - Launching between 12pm and 2pm UTC (this is related to the country a project is launched from, but backers could come from all over the world)
# - Name and blurb lengths (shorter blurbs and longer names are preferred)
# 
# Factors which had a **negative effect** on success rate and/or the amount of money received are:
# 
# **Most important:**
# - Large goals
# - Longer campaigns
# - Food and journalism projects
# - Projects from Italy
# 
# **Less important:**
# - Launching on a weekend
# - Launching in July or December
# - Launching between 6pm and 4am UTC
# 
# Overall, Kickstarter is well suited to small, high-quality projects, particularly comics, dances and games. It is less suited to larger projects, particularly food and journalism projects.
