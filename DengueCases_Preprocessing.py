#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Team Introduction</h2>
# </div>
# 
# <b>1. Kew Jing Sheng - S2021452</b>
# 
# <b>2. Ezlan - S</b>
# 
# <b>3. Yugavashini - S</b>
# 
# <b>4. Puvee - S</b>
# 
# <b>5. Ghayathri - S</b>
# 

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Agenda</h2>
# </div>
# 
# <img src="img/agenda.png" alt="agenda" height="400" width="800">

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Intitial Questions</h2>
# </div>

# <img src="img/mosquito.jpg" alt="mosquito" height="400" width="400">
# 
# <h3>Can we predict local epidemics of <b>Dengue</b> cases?</h3>
# 
# <b>Dengue</b> fever is a <code>mosquito-borne</code> disease that occurs in tropical and sub-tropical parts of the world. In mild cases, symptoms are similar to the flu: fever, rash, and muscle and joint pain. In severe cases, <b>Dengue</b> fever can cause severe bleeding, low blood pressure, and even death.
# 
# <img src="img/dengue_outbreak.png" alt="outbreak" height="500" width="1000">
# 
# Because it is carried by mosquitoes, the transmission dynamics of <b>Dengue</b> are related to climate variables such as temperature and precipitation. Although the relationship to climate is complex, a growing number of scientists argue that climate change is likely to produce distributional shifts that will have significant public health implications worldwide.
# 
# It's critical to enhance understanding about the network’s notion about <b>Dengue</b> as well as their vector manipulation practices before and after the launching of any network-based vector manipulation applications. There had been numerous troubles related to the problem of controlling <b>Dengue</b> in those hotspots, as cases of <b>Dengue</b> have remained uncontrolled and vary due to climate alternate and tropical seasons in Malaysia (District Health Office, Gombak, 2010, unpublished records). Thus, these issues boost the motivation to conduct research based on factors affecting factors that affect the <b>Dengue</b> cases in Malaysia.

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Overview & Motivations</h2>
# </div>

# <h3>Overview</h3>
# 
# The general overview of this project is to study <code>variables</code> affecting the <b>Dengue</b> cases and their weightage by analysing public data set regarding the DEN-V field. 
# 
# Our goal is to predict the <code>total_cases</code> of <b>dengues</b> for each <code>city</code>, <code>year</code> and <code>weekofyear</code> in the datset. There are 2 cities, San Juan, <b>sj</b> and Iquitos, <b>iq</b>, with the data spanning from period of 2001 to 2021.
# 
# <img src="img/data_intro.png" alt="data_intro" height="1000" width="1000">
# 
# <b> List of Features </b>
# We are provided the following information on a <code>year</code> & <code>weekofyear</code> timescale:
# 
# 1. <font size="3px" face="Dosis" color=""><b>City & Date Indicators</b></font>
#     - <code>city</code>: City abbreviations : <b>sj</b> for San Juan and <b>iq</b> for Iquitos
#     - <code>week_start_date</code> : Date given in <b>yyyy-mm-dd</b> format
# 2. <font size="3px" face="Dosis"><b>PERSIANN satellite precipitation measurements (0.25x0.25 degree scale)</b></font>
#     - <code>precipitation_amt_mm</code> : Total precipitation
# 3. <font size="3px" face="Dosis"><b>NOAA's (National Oceanic and Atmospheric Administration) NCEP Climate Forecast System Reanalysis measurements (0.5x0.5 degree scale)</b></font>
#     - <code>reanalysis_sat_precip_amt_mm</code> : Total precipitation
#     - <code>reanalysis_dew_point_temp_k</code> : Mean dew point temperature
#     - <code>reanalysis_air_temp_k</code> : Mean air temperature
#     - <code>reanalysis_relative_humidity_percent</code> : Mean relative humidity
#     - <code>reanalysis_specific_humidity_g_per_kg</code> : Mean specific humidity
#     - <code>reanalysis_precip_amt_kg_per_m2</code> : Total precipitation
#     - <code>reanalysis_max_air_temp_k</code> : Maximum air temperature
#     - <code>reanalysis_min_air_temp_k</code> : Minimum air temperature
#     - <code>reanalysis_avg_temp_k</code> : Average air temperature
#     - <code>reanalysis_tdtr_k</code> : Diurnal temperature range
# 3. <font size="3px" face="Dosis"><b>Satellite vegetation - Normalized difference vegetation index (NDVI) - NOAA's CDR Normalized Difference Vegetation Index (0.5x0.5 degree scale) measurements</b></font>
#     - <code>ndvi_se</code> : Pixel <b>southeast</b> of city centroid
#     - <code>ndvi_sw</code> : Pixel <b>southwest</b> of city centroid
#     - <code>ndvi_ne</code> : Pixel <b>northeast</b> of city centroid
#     - <code>ndvi_nw</code> : Pixel <b>northwest</b> of city centroid
# 
# Each variable will tabulate each set of data accordingly, will then visualize/plot in determining which variables lower or increase <b>Dengue</b> cases (map its correlation either positive or negative) by looking at its <b>Pearson correlation</b>. From the correlation, we can grasp a good idea on how these features correlate with the Dengue cases. But before moving to correlation study, series of preprocessing are conducted to clean the data.
# 
# <h3>Motivation</h3>
# 
# <font size="3px">How might we gain better prevention on <b>Dengue</b> outbreak by having good understanding between information that would cause <b>Dengue</b> cases outbreak?</font>

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Related Work</h2>
# </div>

# Numerous studies on finding the significant risk factors of <b>Dengue</b> outbreaks have been carried out globally. 
# 
# In particular, the Institute for Medical Research experimented to discover the aspects causing the outbreak in Malaysia from 2007 to 2009 which directed towards significant factors such as entomological, epidemiological and environmental. 
# 
# An experiment that was conducted in Kuala Lumpur, Pahang, Kedah and Johor identified the previous week rainfall, optimum relative humidity and temperature affected the increase in the breeding of mosquitoes. Nevertheless, there was only a correlation between the entomology, reproduction of mosquitoes and natural attributes such as temperature, rainfall and relative humidity level (Ahmad et al., 2018). 
# 
# Some research has exhibited different factors such as ecological, meteorological and socioeconomic features also highly affect the outbreak of <b>Dengue</b> cases in Yunnan and Guangdong Provinces, China. However, most past research has not thoroughly identified the consequence of multiple correlations of different features on single factors (Liu et al., 2021).

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Data Cleaning & Preprocessing</h2>
# </div>

# <img src="img/processing_workflow.png" alt="workflow" height="500" width="1000">

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Install Packages</h2>
# </div>

# In[1]:


import os
import pathlib
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno
from collections import OrderedDict

from IPython.display import display, Markdown, clear_output, HTML
import ipywidgets as widgets
from ipywidgets import interact, interactive

from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

pd.set_option('display.max_columns', 1000)
pd.options.display.float_format = '{:,.4f}'.format


# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Import Data</h2>
# </div>

# In[2]:


BASE_DIR = pathlib.Path().resolve()
DATASET_DIR = BASE_DIR / "data/dengue"
EXPORT_DIR = DATASET_DIR / "exports"


# In[3]:


train_df = pd.read_csv(DATASET_DIR / 'train_df.csv', index_col=0)
train_df["week_start_date"] = pd.to_datetime(train_df["week_start_date"])

test_df = pd.read_csv(DATASET_DIR / 'test_df.csv', index_col=0)
test_df["week_start_date"] = pd.to_datetime(test_df["week_start_date"])


# <font size="5px" face="sans-serif"><ins>Feature Data Understanding</ins></font>

# In[4]:


display(train_df.head())
display(Markdown("There are a total of $1456$ number of observations in training data samples."))
display(train_df.info())


# In[5]:


display(test_df.head())
display(Markdown("There are a total of $416$ number of observations in testing data samples."))
display(test_df.info())


# <font size="5px" face="sans-serif"><ins>Target Variable Analysis: <code>total_cases</code></ins></font>

# In[6]:


def horizontal_bar_plot(df, var, normalize=True, title=None):
    var_df = df[var].value_counts(normalize=normalize).reset_index().rename(columns={'index': 'category'})
    n_colors = len(var_df)

    fig, ax = plt.subplots(figsize=(15,3))
    sns.barplot(x=var, y="category", data=var_df, ax=ax)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("City", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="y", which='major', labelsize=14)
    ax.tick_params(axis="y", which='minor', labelsize=14)
    ax.tick_params(axis="x", which='major', labelsize=14)
    ax.tick_params(axis="x", which='minor', labelsize=14)

    return plt.show()


def line_plot(df, x_var, y_var, title=None):
    fig, ax = plt.subplots(figsize=(15,3))
    plt.plot(pd.to_datetime(df[x_var]), df[y_var], color="black")
    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel(x_var, fontsize=14)
    plt.ylabel(y_var, fontsize=14)
    ax.grid()

    return plt.show()


def distribution_plot(df, x_var, title=None):
    fig, ax = plt.subplots(figsize=(15,3))
    sns.distplot(df[x_var], bins=50, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_var)
    ax.set_ylabel("Density")
    ax.grid()
    
    return plt.show()


def multiple_distribution_plot(df, var, xticks=True, xticks_value=[], title=None):
    fig, ax = plt.subplots(figsize=[15,5])
    sns.distplot(a=df[df["city"]=="sj"][var],
                 bins=df[df["city"]=="sj"][var].nunique(),
                 kde=False, label='SJ')
    sns.distplot(a=df[df["city"]=="iq"][var],
                 bins=df[df["city"]=="iq"][var].nunique(),
                 kde=False, label='IQ')
    if xticks:
        plt.xticks(labels=xticks_value, ticks=xticks_value)
    else:
        plt.xticks()
    plt.title("Training data by year and city", fontweight="bold", fontsize="14")
    plt.legend()
    
    return plt.show()


# <font size="3px">Since there are data from $2$ different cities, <b>iq</b> and <b>sj</b>, in the dataset, we need to understand the data nature from these $2$ different cities, then we can make the decision whether we can process the data from $2$ different cities together or separately.</font>

# In[7]:


display(Markdown('''<font size="3px">1. Visualize the occurences of <code>total_cases</code> across 2 cities, to have a grasp on how the data spread between 2 cities.</font>
'''))

for city, data in train_df.groupby(["city"]):
    print(data["week_start_date"].min(), data["week_start_date"].max())
    line_plot(data, "week_start_date", "total_cases", title="Dengue Cases by week in {}".format(city))


# In[8]:


display(Markdown('''<font size="3px">2. Visualize the amount of <code>total_cases</code> data across 2 cities, to have an idea on how our much data we have for two cities.</font>
'''))

multiple_distribution_plot(train_df, "year", xticks=True, xticks_value=[1998, 2003, 2008, 2013, 2018], title=None)

horizontal_bar_plot(train_df, "city", normalize=False, title="Count of Dengue cases Across 2 Different States: San Juan & Iquitos")


# <font size="4px">We can notice that the <b>occurences</b> and <b>amount</b> of <code>total_cases</code> of dengues are different between $2$ cities. Besides, the values problems and cleaning processes would also be different between $2$ cities. Hence, we will <b>separate</b> the data into $2$ different cities and treat the data separately.</font>

# In[9]:


display(Markdown('''<font size="3px">Splitting data based two different cities.</font>
'''))

sj_train_df = train_df[train_df["city"]=="sj"]
sj_train_df.reset_index(drop=True, inplace=True)
display(Markdown("<b>sj</b> dataset: {}".format(sj_train_df.shape)))

iq_train_df = train_df[train_df["city"]=="iq"]
iq_train_df.reset_index(drop=True, inplace=True)
display(Markdown("<b>iq</b> dataset: {}".format(iq_train_df.shape)))


# In[10]:


for city, data in train_df.groupby(["city"]):
    distribution_plot(data, "total_cases", title="Distribution of Dengue Total Cases in {}".format(city))
    print("Mean: ", data["total_cases"].mean())
    print("Median: ", data["total_cases"].median())


# <font size="3px">Eventhough the <code>total_cases</code> data nature are different across period between $2$ cities. Yet, we can observe that the <code>total_cases</code> data distribution are quite resemble, where both cities data are also <b>right-skewed</b>, this making the target variable is not <b>normal</b>. Hence, perhaps in the modelling stage later, some resampling to a much normally distributed data is required.</font>

# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Data Preprocessing</h2>
# </div>
# 
# <font size="3px">There are few preprocesses that we will be performing to wrangle the data:</font>
# <img src="img/data_preprocessing_steps.png" alt="Mountain">

# <font size="3px">1. To process the data, we will split the data into both different cities, <b>sj</b> and <b>iq</b> and perform preprocessing on each of it separately.</font>
# 
# <font size="3px">2. After we done processing the data, we will merge the data from both cities back into the training dataset for modelling later on.</font>

# <font size="5px" face="sans-serif"><ins>1. Correcting Inconsistent Data</ins></font>
# 
# <font size="3px">The feature <code>weekofyear</code> has some inconsistent as in certain years, the 1st week of the year is labelled as $53$ week, instead of $1$.</font>

# - <font size="3px" face="sans-serif"><b>sj</b> city data</font>

# In[11]:


for year in sj_train_df["year"].unique():
    print(f'{year} max: {max(sj_train_df[sj_train_df["year"]==year]["weekofyear"])}      | min: {min(sj_train_df[sj_train_df["year"]==year]["weekofyear"])}      | uniques: {sj_train_df[sj_train_df["year"]==year]["weekofyear"].nunique()}')
    
display(Markdown("""<font size="3px">Some weeks have a mistaken weeknumber value, this is because the maximum number of weeks in certain years (eg. $2004$, $2010$, $2016$) is $53$ but one year only has $52$ unique weeks. This <b>inconsistent</b> naming of week number need to be corrected by changing all the maximum number of weeks in each year is only $52$.</font>
"""))


# A closer look shows that it's the first week of the year that is registering as week 53:

# In[12]:


for year in [2001, 2007, 2013]:
    display(sj_train_df[sj_train_df["year"]==year][["city", "year", "weekofyear", "week_start_date"]].head(3))
    
display(Markdown("""<font size="3px">This error on <b>labelling the first week of the year as $53$</b> will be corrected using these 2 steps:</font>
<ul>
    <li><font size="2.5px">Incrementing all the week values by one. This would make the first week 54, and the \
    rest of the weeks should be correct</font></li>
    <li><font size="2.5px">Then, reduce the any week values over 52 (etc. 53-1, 54-1) by 1.</font></li>
</ul>
"""))


# In[13]:


display(Markdown("Increment all week values by one for all problem years"))
for year in [2001, 2007, 2013]:
    sj_train_df.loc[:,'weekofyear'] = np.where(sj_train_df["year"]==year,
                                               sj_train_df["weekofyear"]+1, 
                                               sj_train_df["weekofyear"])

display(Markdown("Reduce any week values over $52$ down to $1$"))
sj_train_df.loc[:,'weekofyear'] = np.where(sj_train_df["weekofyear"]>52,
                                           1, 
                                           sj_train_df["weekofyear"])

for year in sj_train_df["year"].unique():
    print(f'{year} max: {max(sj_train_df[sj_train_df["year"]==year]["weekofyear"])}    | min: {min(sj_train_df[sj_train_df["year"]==year]["weekofyear"])}    | uniques: {sj_train_df[sj_train_df["year"]==year]["weekofyear"].nunique()}')


# - <font size="3px" face="sans-serif"><b>iq</b> city data</font>

# In[14]:


for year in iq_train_df["year"].unique():
    print(f'{year} max: {max(iq_train_df[iq_train_df["year"]==year]["weekofyear"])}    | min: {min(iq_train_df[iq_train_df["year"]==year]["weekofyear"])}    | uniques: {iq_train_df[iq_train_df["year"]==year]["weekofyear"].nunique()}')
    
display(Markdown("""<font size="3px">Some weeks have a mistaken weeknumber value, since the max for some years (eg. $2016$, $2021$) is 53 despite still only having 52 unique weeks.</font>
"""))


# A closer look shows that it's the first week of the year that is registering as week 53:

# In[15]:


for year in [2013, 2018]:
    display(iq_train_df[iq_train_df["year"]==year][["city", "year", "weekofyear", "week_start_date"]].head(3))


# In[16]:


display(Markdown("Increment all week values by one for all problem years"))
iq_train_df.loc[:, "weekofyear"] = np.where(iq_train_df["year"]==2013,
                                            iq_train_df["weekofyear"]+1,
                                            iq_train_df["weekofyear"])
iq_train_df.loc[:, "weekofyear"] = np.where(iq_train_df["year"]==2018,
                                            iq_train_df["weekofyear"]+1,
                                            iq_train_df["weekofyear"])

display(Markdown("Reduce any week values over $52$ down to $1$"))
iq_train_df.loc[:,'weekofyear'] = np.where(iq_train_df["weekofyear"]>52,
                                           1, 
                                           iq_train_df["weekofyear"])

for year in iq_train_df["year"].unique():
    print(f'{year} max: {max(iq_train_df[iq_train_df["year"]==year]["weekofyear"])}  | min: {min(iq_train_df[iq_train_df["year"]==year]["weekofyear"])}  | uniques: {iq_train_df[iq_train_df["year"]==year]["weekofyear"].nunique()}')


# <font size="5px" face="sans-serif"><ins>2. Missing Values Checking</ins></font>

# In[17]:


display(pd.isnull(train_df).any())

display(Markdown("""There are misisng values across the dataframe, besides from <code>city</code>, <code>year</code>, <code>weekofyear</code>, <code>week_start_date</code> and <code>total_cases</code>.
"""))


# In[18]:


def missingno_matrix(df, title):
    fig, ax = plt.subplots(figsize=(15,5))
    msno.matrix(df, ax=ax)
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    return plt.show()


# In[19]:


missingno_matrix(sj_train_df, "Missing Values Distrbution Matrix of sj City")

print("Total Missing Rows", sj_train_df.isnull().sum().max())
print("% of Missing Rows Throughout the Data", (sj_train_df.isnull().sum().max() / sj_train_df.shape[0])*100)

display(Markdown("""<font size="3px"><code>ndvi_ne</code> has the most missing values, we will select this feature as the reference on how our missing values imputation work</font>
"""))


# In[20]:


distribution_plot(sj_train_df, "ndvi_ne", title="Distribution of ndvi_ne Before Missing Value Imputation")

display(Markdown("""<font size="3px">Understand the <b>distribution of data</b> is essential before missing value handling, to give us some glance that our imputation is making sense. We will be using <b>KNN Data Imputation</b> to handle the mising values.</font>
"""))


# <font size="4px" face="sans-serif"><b>KNN Data Imputation:</b></font>
# 
# <font size="3px"><b>KNN</b>, stands for <b>K-Nearest Neighbour</b>, utilizes data from neighbouring points to replace the missing values in the datasets with the mean values from the parameter <code>n_neighbors</code>.</font>
# 
# <img src="https://www.k2analytics.co.in/wp-content/uploads/2020/06/K_NN.png" alt="KNN Model" height="500" width="500">
# 
# <font size="3px"><b>KNN</b> will measure the distance and find the nearest neighbours to impute the missing values between the data points</font>

# In[21]:


display(Markdown("""Setting the number of neighbors, <code>n_neighbors</code> to $5$. 
"""))
imputer = KNNImputer(n_neighbors=5)

display(Markdown("""<font size="3px">All the numeric columns in <b>sj</b> will be imputed using <b>KNN</b></font>
"""))
num_sj_train_df = sj_train_df.select_dtypes(include=np.number)
num_sj_train_df = pd.DataFrame(imputer.fit_transform(num_sj_train_df), columns=num_sj_train_df.columns)
for col in num_sj_train_df.columns:
    sj_train_df[col] = num_sj_train_df[col]
    
display(Markdown("""Done imputation..."""))


# In[22]:


missingno_matrix(sj_train_df, "Missing Values Distrbution Matrix of sj City")

print("Total Missing Rows", sj_train_df.isnull().sum().max())
print("% of Missing Rows Throughout the Data", (sj_train_df.isnull().sum().max() / sj_train_df.shape[0])*100)

display(Markdown("""<font size="3px">All missing values have been imputed.</font>"""))


# In[23]:


distribution_plot(sj_train_df, "ndvi_ne", title="Distribution of ndvi_ne After Missing Value Imputation")

display(Markdown("""<font size="3px">We can see that the data distribution after <b>KNN Data Imputation</b> does not change much. Hence, our imputation method is working well.</font>
"""))


# In[24]:


display(Markdown('<font size="3px">Perform the same missing values data imputation in <b>iq</b> city dataset.</font>'))

missingno_matrix(iq_train_df, "Missing Values Distrbution Matrix of iq City")

print("Total Missing Rows", iq_train_df.isnull().sum().max())
print("% of Missing Rows Throughout the Data", (iq_train_df.isnull().sum().max() / iq_train_df.shape[0])*100)


# In[25]:


display(Markdown("""<font size="3px">All the numeric columns in <b>iq</b> will be imputed using <b>KNN</b></font>
"""))
num_iq_train_df = iq_train_df.select_dtypes(include=np.number)
num_iq_train_df = pd.DataFrame(imputer.fit_transform(num_iq_train_df), columns=num_iq_train_df.columns)
for col in num_sj_train_df.columns:
    iq_train_df[col] = num_iq_train_df[col]
    
display(Markdown("""Done imputation..."""))


# In[26]:


missingno_matrix(iq_train_df, "Missing Values Distrbution Matrix of iq City")

print("Total Missing Rows", iq_train_df.isnull().sum().max())
print("% of Missing Rows Throughout the Data", (iq_train_df.isnull().sum().max() / iq_train_df.shape[0])*100)


# <font size="5px" face="sans-serif"><ins>3. Outliers Data Handling</ins></font>
# 
# <font size="3px">Outliers data will be checked visually using box-plot, and then if there are present of extreme values, the outliers will be removed using <b>statistical IQR filtering</b>. There are few steps to run the <b>statistical IQR filtering</b>:</font>
# 1. Calculate the first ($Q1$) and third quantile ($Q3$)
# 2. Find the interquartile range ($Q3-Q1$)
# 3. Find the <b>lower bound</b>, where $Q1 * 1.5$
# 4. Find the <b>upper bound</b>, where $Q3 * 1.5$
# 5. Mask anything outside <b>lower bound</b> and <b>upper bound</b>
# 
# <img src="https://www.dataquest.io/wp-content/uploads/2019/01/whm_elements_of_a_boxplot_en_wikimedia.png" alt="KNN Model" height="500" width="500">

# In[27]:


train_df = pd.concat([sj_train_df, iq_train_df])
train_df.reset_index(drop=True, inplace=True)


# In[28]:


def boxplot(df, x_var, y_var, title=None):
    fig, ax = plt.subplots(figsize=(15,1))
    sns.boxplot(x=x_var, y=y_var, data=df, ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y_var, fontsize=11)
    
    return plt.show()


def iqr_based_filtering(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound  = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
        
    df = df[(df[col]<lower_bound) |(df[col]>upper_bound)]
    
    return df


# In[29]:


ANALYSIS_COLS = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                 'reanalysis_precip_amt_kg_per_m2',
                 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                 'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                 'station_min_temp_c', 'station_precip_mm']

for col in ANALYSIS_COLS:
    boxplot(train_df, col, "city", title="Boxplot of {} Based on City".format(col))


# In[30]:


OUTLIERS_COLS = ["precipitation_amt_mm", "reanalysis_precip_amt_kg_per_m2",
                 "reanalysis_sat_precip_amt_mm", "station_precip_mm"]

for col in OUTLIERS_COLS:
    train_outlier_removed_df = iqr_based_filtering(train_df, col)


# In[31]:


fig = plt.figure(figsize=(13, 12), constrained_layout=True)
spec = gridspec.GridSpec(nrows=4, ncols=2, figure=fig)

# # precipitation_amt_mm
ax0 = fig.add_subplot(spec[0, 0])
sns.boxplot(x="precipitation_amt_mm", y="city", data=train_df, ax=ax0)
ax0.set_title("precipitation_amt_mm - Before", fontweight="bold", fontsize=13)
ax0.set_xlabel("precipitation_amt_mm", fontsize=11)
ax0.set_ylabel("city", fontsize=11)

ax1 = fig.add_subplot(spec[0, 1])
sns.boxplot(x="precipitation_amt_mm", y="city", data=train_outlier_removed_df, ax=ax1)
ax1.set_title("precipitation_amt_mm - After", fontweight="bold", fontsize=13)
ax1.set_xlabel("precipitation_amt_mm", fontsize=11)
ax1.set_ylabel("city", fontsize=11)

# # reanalysis_precip_amt_kg_per_m2
ax2 = fig.add_subplot(spec[1, 0])
sns.boxplot(x="reanalysis_precip_amt_kg_per_m2", y="city", data=train_df, ax=ax2)
ax2.set_title("reanalysis_precip_amt_kg_per_m2 - Before", fontweight="bold", fontsize=13)
ax2.set_xlabel("reanalysis_precip_amt_kg_per_m2", fontsize=11)
ax2.set_ylabel("city", fontsize=11)

ax3 = fig.add_subplot(spec[1, 1])
sns.boxplot(x="reanalysis_precip_amt_kg_per_m2", y="city", data=train_outlier_removed_df, ax=ax3)
ax3.set_title("reanalysis_precip_amt_kg_per_m2 - After", fontweight="bold", fontsize=13)
ax3.set_xlabel("reanalysis_precip_amt_kg_per_m2", fontsize=11)
ax3.set_ylabel("city", fontsize=11)

# # reanalysis_sat_precip_amt_mm
ax4 = fig.add_subplot(spec[2, 0])
sns.boxplot(x="reanalysis_sat_precip_amt_mm", y="city", data=train_df, ax=ax4)
ax4.set_title("reanalysis_sat_precip_amt_mm - Before", fontweight="bold", fontsize=13)
ax4.set_xlabel("reanalysis_sat_precip_amt_mm", fontsize=11)
ax4.set_ylabel("city", fontsize=11)

ax5 = fig.add_subplot(spec[2, 1])
sns.boxplot(x="reanalysis_sat_precip_amt_mm", y="city", data=train_outlier_removed_df, ax=ax5)
ax5.set_title("reanalysis_sat_precip_amt_mm - After", fontweight="bold", fontsize=13)
ax5.set_xlabel("reanalysis_sat_precip_amt_mm", fontsize=11)
ax5.set_ylabel("city", fontsize=11)

# # station_precip_mm
ax6 = fig.add_subplot(spec[3, 0])
sns.boxplot(x="station_precip_mm", y="city", data=train_df, ax=ax6)
ax6.set_title("station_precip_mm - Before", fontweight="bold", fontsize=13)
ax6.set_xlabel("station_precip_mm", fontsize=11)
ax6.set_ylabel("city", fontsize=11)

ax7 = fig.add_subplot(spec[3, 1])
sns.boxplot(x="station_precip_mm", y="city", data=train_outlier_removed_df, ax=ax7)
ax7.set_title("station_precip_mm - After", fontweight="bold", fontsize=13)
ax7.set_xlabel("station_precip_mm", fontsize=11)
ax7.set_ylabel("city", fontsize=11)

fig.suptitle("Before & After Outliers Removal", fontsize=16, fontweight="bold")


# In[32]:


display(Markdown("""<font size="3px">Remove outliers in the dataset</font>
"""))

ol_col_list = ["city", "year", "week_start_date"] + OUTLIERS_COLS
sub_train_outlier_removed_df = train_outlier_removed_df[ol_col_list]

merge_train_df = train_df.merge(sub_train_outlier_removed_df, 
                                    on=["city", "year", "week_start_date"], 
                                    how="left", 
                                    suffixes=["", "_src"], 
                                    indicator="_join_ind")

for col in OUTLIERS_COLS:
    train_df.loc[merge_train_df["_join_ind"]=="both",col] = merge_train_df[col+"_src"]


# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Merging Back The sj & iq Dataset</h2>
# </div>

# In[33]:


train_df["total_cases"] = train_df["total_cases"]
train_corr = (train_df.corr().total_cases.drop(["total_cases", "weekofyear", "year"], axis=0).sort_values())


# In[34]:


def correlation_bar(series, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x=series, y=series.index, ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=15)
    
    return plt.show()


# In[35]:


correlation_bar(train_corr, title="Correlation of Numeric Features with total_cases of Dengue")

display(Markdown("""Based on <b>Pearson correlations</b>, certain features like <code>reanalysis_tdtr_k</code>, <code>station_diur_temp_rng_c</code> and <code>reanalysis_max_air_temp_k</code> have strong <b>negative</b> relationship. In contrast, features like <code>reanalysis_min_air_temp_k</code>, <code>station_min_temp_c</code> and <code>reanalysis_air_temp_k</code> have strong <b>positive</b> relationship.
"""))


# <div class="alert alert-info" style="background-color:#006a79; color:white; padding:0px 10px; border-radius:5px;"><h2 style='margin:10px 5px'>Next Way Forward</h2>
# </div>
# 
# <font size="3px">We have successfully process the data to a relatively ready data to move into further steps. We have completed:</font>
# 
# 1. Understanding the data ,b>nature of target variable</b> <code>total_cases</code> in the data, and perform processing by splitting data into $2$ different cities, <b>sj</b> and <b>iq</b>
# 
# 
# 2. Fixing the <b>data inconsistency</b> issue in the feature <code>weekofyear</code> where the $1st$ week of certain years are labelled as $53$, not $52$, which is not right, as the maximum weeks per year is $52$.
# 
# 
# 3. Handle the missing values by imputing the missing values using <b>KNN (K-Nearest Neighbors)</b> model, by defining the <code>n_neighbors</code> as $5$. The imputatation is good as the data distribution before and after imputation do not change much.
# 
# 
# 4. Handle of outliers and noisy data using <b>IQR Based Filtering</b>. We identified there are presence of extreme values in certain features like <code>precipitation_amt_mm</code>, <code>reanalysis_precip_amt_kg_per_m2</code>, <code>"reanalysis_sat_precip_amt_mm"</code>, <code>"station_precip_mm"</code> using visualization method like <b>Boxplot</b>. Then, these extreme values are further masked. 

# <img src="img/next_way_forward.png" alt="agenda" height="500" width="1000">
