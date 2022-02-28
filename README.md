# Email classification into themes using NLP

*This paper presents different approaches to classifying e-mails into different categories using natural language processing (NLP).
You can fin the source (notebook file)* [here](https://github.com/MohamedBouzidGit/giskard/blob/2091fd6705ac90ed6cc130dd281b3642ab856288/GiskardExercice.ipynb).  


## Introduction

We spend a lot of time sorting out our emails at work. And it's enough to come back from vacations to spend hours, even days! It would be convenient for us to know what topic is associated with our mail in order to treat certain subjects in priority. 

While one can find projects showing how to classify emails as spam / non-spam, it is very difficult to find documentation on classifying emails into different themes. 


## Challenge

The project aims at classifying emails into different themes using some headers and the body of the email. This results in an NLP task that manages to understand the meaning of the text via the Bag of Words and Term Frequency Inverse Document Frequency (TF-IDF) models. 

### Dataset

*giskard.csv* : a dataset of over 800 emails, labeled in 13 themes: 

  1. `regulations and regulators (includes price caps)`
  2. `internal projects -- progress and strategy`
  3. `company image -- current`
  4. `corporate image -- changing / influencing`
  5. `political influence / contributions / contacts`
  6. `california energy crisis / california politics`
  7. `internal company policy`
  8. `internal company operations`
  9. `alliances / partnerships`
  10. `legal advice`
  11. `talking points`
  12. `meeting minutes`
  13. `trip reports`

### Strategy

We plan the following methodology : 
1. Recovery of data and python libraries
2. Data exploration
3. Data pre-processing (check the need for cleaning and feature engineering)
4. Modeling with Bag of Words and TF-IDF then optimization of the best model


## 1. Recovery of data and python libraries

The dataset, once downloaded, has the following form: 

```
df = pd.read_csv('giskard_dataset.csv', sep=';', index_col=[0]) # extract dataset
df = df.reset_index().drop(axis=1, columns = 'index') # this avoid index corruption after removing

df.head()
```
![image](https://user-images.githubusercontent.com/74253587/155990320-1132994c-54b2-43e4-ae2c-2caca343acb3.png)


## 2. Data exploration

Exploratory Data Analysis (EDA) is an analysis approach that identifies general patterns in the data. Here, the data are mostly string values, which means that there are no numerical patterns, but we can still examine the ratio of each target class.

![image](https://user-images.githubusercontent.com/74253587/155991740-004af801-f1d2-4c28-886f-8e8968155fe7.png)

We can see that half of the messages are related to :

- regulations and regulators (20.9%)
- california energy crisis / california politics (17%)
- internal projects -- progress and strategy (12.4%)





