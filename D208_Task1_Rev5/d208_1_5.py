'''
########################################
    IMPORT REQUIRED PACKAGES
######################################## '''

# standard
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

# statistics
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# plots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# sklearn
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA

# regression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# other
from IPython.display import Latex
import sys
import os
import datetime

# personal library
from mm_lib import *

# remove the ellipsis from terminal pandas tables
pd.set_option("display.width", None)
np.set_printoptions(threshold=np.inf)

# reconfigure sys.stdout to utf-8
sys.stdout.reconfigure(encoding="utf-8")

'''
########################################
    STUDENT INFORMATION
######################################## '''
''' Example:
       Student | Mike Mattinson
    Student ID | 001980761
         Class | D208 Predictive Analysis
          Task | Task 1 Multiple Regression
       Dataset | Churn
    Submission | 1st Submission
        School | WGU
    Instructor | Keiona Middleton
Today is August 31, 2021
'''
# print student information
print(heading_toString("STUDENT INFORMATION"))
stud_info = {
  "Student": "Mike Mattinson",
  "Student ID": "001980761",
  "Class": "D208 Predictive Analysis",
  "Task" : "Task 1 Multiple Regression",
  "Dataset": "Churn",
  "Submission": "1st Submission",
  "School": "WGU",
  "Instructor": "Keiona Middleton"
}
for key in stud_info:
    print('{:>14} | {}'.format(key,stud_info[key]))

print("Today is {}".format(datetime.date.today().strftime("%B %d, %Y")))


# print python enivronment
''' Example:
Version: 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21)
 [MSC v.1929 64 bit (AMD64)] located at 
 P:\workspace-wgu\wgu_local_py\wgu_venu\Scripts\python.exe
'''
print(heading_toString("PYTHON ENVIRONMENT"))
print("Version: {} located at {}".format(sys.version, sys.executable))


# global variables
count_tables = 0  # keep track of the number of tables generated
count_figures = 0  # keep track of the number of figures generated
course = "d208"
dep_mlr = ''  # use continuous variable
dep_lr = 'Churn' # use categorical variable


# check and create folder to hold all figures
figure_folder='figures\\' + course + '\\'
if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

table_folder='tables\\' + course + '\\'
if not os.path.exists(table_folder):
        os.makedirs(table_folder)

'''
########################################
    SETUP DATAFRAME
######################################## '''
''' Example:
  Customer_id         City Churn  MonthlyCharge     Tenure
0     K409198  Point Baker    No     172.455519   6.795513
1     S120509  West Branch   Yes     242.632554   1.156681
2     K191035      Yamhill    No     159.947583  15.754144
3      D90850      Del Mar    No     119.956840  17.087227
(10000, 50)
'''
# create dataframe
print(heading_toString("DATAFRAME (DF)"))
df = pd.read_csv("data\churn_clean.csv")
print(df[["Customer_id", "City", "Churn", "MonthlyCharge", "Tenure"]].head(4))
print(df.shape)
print(df.info())


'''
########################################
    CLEAN DATA
######################################## '''
# remove unwanted columns
print(heading_toString("REMOVE UNWANTED COLUMNS"))

# remove unwanted columns
unwanted_columns = ["UID", "Interaction", "Lat", "Lng", "CaseOrder",
                "Customer_id"]

for uc in unwanted_columns:
    if uc in df.columns:
        df.drop(columns=uc, inplace=True)
        print("[{}] column removed.".format(uc))

# show remaining columns
print("Remaining columns: \n{}".format(df.columns))


# Rename columns
print(heading_toString("RENAME COLUMNS"))
df.rename(columns = {
  'Population':'POP',
  'Children':'CHI',
  'Age':'AGE',
  'Email':'EML',
  'Contacts':'CON',
  'Contract':'CONTR',
  'Yearly_equip_failure':'YEF',
  'Gender':'GEN',
  'Tablet':'TAB',
  'MonthlyCharge':'MC',
  'Bandwidth_GB_Year':'BGY',
  'Tenure':'TEN',
  'Income':'INC',
  'Marital':'MAR',
  'Churn':'CHRN',
  'Outage_sec_perweek':'OPW',
  'Techie':'TEC',
  'InternetService':'IS',
  'Phone':'PHN',
  'Multiple':'MULT',
  'OnlineSecurity':'SEC',
  'OnlineBackup':'BACK',
  'DeviceProtection':'DP',
  'TechSupport':'TS',
  'StreamingTV':'TV',
  'StreamingMovies':'MOV',
  'PaperlessBilling':'PB',
  'PaymentMethod':'PAY',
  },
inplace=True)

print(heading_toString("CONTINUOUS DATA"))
df_cont = df.select_dtypes(include="float")
print(df_cont.info())

print(heading_toString("INTEGER DATA"))
print(df.select_dtypes(include="integer").info())

print(heading_toString("CATEGORICAL DATA"))
print(df.select_dtypes(include="object").info())

# output dataframe as .csv table
print(heading_toString("C5 - PROVIDE COPY OF CLEAN DATA"))
count_tables += 1
table_title = "churn_clean_data"
fname = (
    "tables\\" + course + "\\" + "Tab_" + str(count_tables) + "_" + table_title + ".csv"
)
print("table saved at: {}".format(fname))
df.to_csv(fname, index=False, header=True)
print(df.columns)

'''
########################################
    GOAL - CONTRACT VS CHURN
######################################## '''
# analyze contract types 
print(heading_toString("Contract (IV) vs Churn (DV)"))

# first plot of all customers
univariate_categorical = {
  "1": "CONTR",
}

for key in univariate_categorical:
    fig, ax = plt.subplots()
    c = univariate_categorical[key]
    ax = sns.countplot(x=c, data=df, order = df[c].value_counts().index)
    plt.xlabel(c)
    plt.ylabel('Count')
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'All','countplot')
    title = '{}'.format(c)
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    for p in ax.patches:
        ax.annotate('{:d}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+50))
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))


# second plot of churned customers
df_churned = df.query('CHRN == "Yes"')
univariate_categorical = {
  "2": "CONTR",
}

for key in univariate_categorical:
    fig, ax = plt.subplots()
    c = univariate_categorical[key]
    ax = sns.countplot(x=c, data=df_churned, order = df[c].value_counts().index)
    plt.xlabel(c)
    plt.ylabel('Count')
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'CHRN','countplot')
    title = '{}'.format(c)
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    for p in ax.patches:
        ax.annotate('{:d}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+50))
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))



'''
########################################
    BIVARIATE - MONTHLYCHARGE VS OUTAGE
######################################## '''
print(heading_toString("PLOT BIVARIATE SCATTER PLOT"))  
bivariate = {
    "3": "MC",
}

for key in bivariate:
    c = bivariate[key]
    target = 'OPW'
    fig, ax = plt.subplots()
    ax = plt.plot(df[c], df[target], 'o', color='black');
    plt.xlabel(c)
    plt.ylabel(target)
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'bivariate','scatter plot')
    title = '{} vs. {}'.format(c,target)
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))


'''
########################################
  GOAL - CHI-SQUARE INDEPENDENCE TESTS
######################################## '''
# CHI-SQUARE INDEPENDENCE TEST
target = 'CHRN' 
prob = 0.999 
predictor = 'MC' # continuous
print(heading_toString("CHI-SQUARE INDEPENDENCE TEST - " + target + " vs. " + predictor))
bins = 6 # continuous variable
chi_square_analysis(target,predictor,bins,prob,df)

# CHI-SQUARE INDEPENDENCE TEST
target = 'CHRN' 
prob = 0.999 
predictor = 'CONTR' # categorical
print(heading_toString("CHI-SQUARE INDEPENDENCE TEST - " + target + " vs. " + predictor))
bins = 0 # categorical variable
chi_square_analysis(target,predictor,bins,prob,df)


'''
########################################
  GOAL - CORRELATION ANALYSIS
######################################## '''
print(heading_toString("CORRELATION ANALYSIS"))

# example boston
#boston = load_boston()
#dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset = df.select_dtypes(include="float")
dataset['target'] = df['CHRN']
X = dataset.iloc[:,:-1]
variables = dataset.columns[:-1]
correlation_matrix = X.corr()
print (correlation_matrix)
print(dataset.info())
print(dataset.head(10))

def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar=0)
    R[np.where(np.abs(R)<hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap=mpl.cm.coolwarm, alpha=0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor=False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor=False)
    heatmap.axes.set_xticklabels(variables, minor=False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(variables, minor=False)
    plt.tick_params(axis='both', which='both', bottom='off', \
    top='off', left = 'off', right = 'off')
    plt.colorbar()
    plt.show()

#visualize_correlation_matrix(X, hurdle=0.5)

'''
########################################
  SIMPLE LINEAR REGRESSION
######################################## '''
print(heading_toString("SIMPLE LINEAR REGRESSION"))
y = df['BGY']
X = df['TEN']

Xc = sm.add_constant(X)
linear_regression = sm.OLS(y,Xc)
fitted_model = linear_regression.fit()
print(fitted_model.summary())
print(df.info())




'''
########################################
  BOSTON - CHAP 3
######################################## '''
print(heading_toString("BOSTON - CHAP 3"))
#https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
#https://www.statsmodels.org/devel/examples/notebooks/generated/contrasts.html
#https://www.weirdgeek.com/2018/12/linear-regression-to-boston-housing-dataset/
#https://www.kaggle.com/jayateerthas/boston-dataset-analysis
#https://www.ritchieng.com/machine-learning-project-boston-home-prices/





boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

print(boston.head())
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
print(boston.isnull().sum())
correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

# remove unwanted columns
#unwanted_columns = ["AGE", "INDUS", "CHAS", "CRIM", "ZN", "TAX", "RAD"]

#for uc in unwanted_columns:
#    if uc in dataset.columns:
#        dataset.drop(columns=uc, inplace=True)
#        print("[{}] column removed.".format(uc))


observations = len(boston)
variables = boston.columns[:-1]
#X = dataset.ix[:,:-1] # ix deprecated
X = boston.iloc[:,:-1]
y = boston['MEDV'].values
Xc = sm.add_constant(X)
linear_regression = sm.OLS(y, Xc)
fitted_model = linear_regression.fit()
print(fitted_model.summary())
print(boston.info()) 



'''
########################################
  CHURN DATA - MULTIPLE REGRESSION
######################################## '''
print(heading_toString("CHURN DATA - MULTIPLE REGRESSION"))
#https://medium.com/data-science-on-customer-churn-data/predictive-analysis-using-multiple-linear-regression-b6b3b79b36b6
#https://towardsdatascience.com/predicting-customer-churn-using-logistic-regression-9543c60f6d47


features = ['TEN', 'BGY']
dataset = pd.DataFrame(df, columns = features)
dataset['target'] = df.MC
observations = len(dataset)
variables = dataset.columns[:-1]
#X = dataset.ix[:,:-1] # ix deprecated
X = dataset.iloc[:,:-1]
y = dataset['target'].values
Xc = sm.add_constant(X)
linear_regression = sm.OLS(y, Xc)
fitted_model = linear_regression.fit()
print(fitted_model.summary())
print(dataset.info()) 
print(features)



