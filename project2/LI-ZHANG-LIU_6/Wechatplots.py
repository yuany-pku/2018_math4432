# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:17:45 2018

@author: Junrong
"""
#draw the plots for data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['Familysize'] = train_data['SibSp'] + train_data['Parch'] + 1

sns.set_style('whitegrid')
train_data.head()

train_data.info()
print("-" * 40)
test_data.info()
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')

train_data.groupby(['Sex','Survived'])['Survived'].count()
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

train_data.groupby(['Pclass','Survived'])['Pclass'].count()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

train_data.groupby(['SibSp','Survived'])['SibSp'].count()
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()

train_data.groupby(['Parch','Survived'])['Parch'].count()
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()

train_data.groupby(['Age','Survived'])['Age'].count()
train_data[['Age','Survived']].groupby(['Age']).mean().plot.bar()

train_data.groupby(['Familysize','Survived'])['Familysize'].count()
train_data[['Familysize','Survived']].groupby(['Familysize']).mean().plot.bar()

train_data.groupby(['Age','Survived'])['Age'].count()
train_data[['Age','Survived']].groupby(['Age']).mean().plot.bar()