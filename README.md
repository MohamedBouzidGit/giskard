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


## 3. Data pre-processing (check the need for cleaning and feature engineering)

### Cleaning
On such a dataset, we check in particular that there are no duplicates or missing values. 

```
# check duplicates to clean
df.duplicated().value_counts()

# calculation of the filling ratio per column
ratio = pd.DataFrame(
    data=(100 - 100*(df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)),
    columns=['values']).reset_index()

# color criteria (red = not full / green = full)
colors = ['red' if y<100 else 'green' for y in ratio['values']]

# chart
sns.barplot(x='index', y='values', data=ratio, palette=colors)

plt.title('Filling ratio (%)')
plt.tick_params(axis='x', rotation=45)
```
![image](https://user-images.githubusercontent.com/74253587/155992823-dd07b190-b435-466b-8c26-e0475e505e73.png)
![image](https://user-images.githubusercontent.com/74253587/155992982-eea23c8a-6cb2-42bd-a97a-22e06a9b4c70.png)

### Feature engineering

Let's see how a message looks like : 
```
Message-ID: <1637509.1075843546651.JavaMail.evans@thyme>
Date: Thu, 31 May 2001 04:19:00 -0700 (PDT)
From: susan.mara@enron.com
To: arem@electric.com, tracy.fairchild@edelman.com, erica.manuel@edelman.com,
	nplotkin@tfglobby.com
Subject: Ackerman to talk to Wolak
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Susan J Mara
X-To: arem@electric.com, tracy.fairchild@edelman.com, erica.manuel@edelman.com, nplotkin@tfglobby.com
X-cc: 
X-bcc: 
X-Folder: \Jeff_Dasovich_June2001\Notes Folders\All documents
X-Origin: DASOVICH-J
X-FileName: jdasovic.nsf

Gary has been seeing Frank Wolak all over the place on the media scene.  I 
asked him to ask Frank to do something POSITIVE and under the state's own 
control and push DA.  Gary is also speaking to a bunch of CEOs (with Anjali 
of the ISO and Carl Wood) and he said he would carry our message to them as 
well.

Sue Mara
Enron Corp.
Tel: (415) 782-7802
Fax:(415) 782-7854
```

We can see that the message respect the *MIME 1* format. The headers such as `Date`, and `Subject`, `From` and the content of the message body can be extracted as characteristics. Fortunately, python has a package called email that can handle the MIME format, so we can use it for our dataset!

```
# Nested function in 'get_features' used for preparing features contents 
def insert_value(dictionary, key, value):
    if key in dictionary:
        values = dictionary.get(key)
        values.append(value)
        dictionary[key] = values
    else:
        dictionary[key] = [value]
    return dictionary


# Function used for extracting features into our dataframe 
def get_headers(df, header_names):
    headers = {}
    messages = df['Message']
    for message in messages:
        e = email.message_from_string(message)
        for item in header_names:
            header = e.get(item)
            insert_value(dictionary = headers, key = item, value = header) 
    return headers
header_names = ['date', 'subject', 'from']    
headers = get_headers(df, header_names)


# Function used for extracting body text from emails
def get_messages(df):
    messages = []
    for item in df['Message']:
        # Return a message object structure from a string
        e = email.message_from_string(item)    
        # get message body  
        message_body = e.get_payload()
        messages.append(message_body)
    return messages
msg_body = get_messages(df)
df['body'] = msg_body


# Function to add headers and values to the main dataframe
def add_headers(df, header_list):
    for label in header_list:
        df_new = pd.DataFrame(headers[label], columns = [label]) # make dataframe based on header_list
        if label not in df.columns: # avoid duplicates
            df = pd.concat([df, df_new], axis = 1)
    return df

header_list = ['date', 'subject', 'from']
df = add_headers(df, header_list)


# We also want to manipulate the date column by converting it to datetime format, before keeping only the months (alphabetical) and years (numerical)
df['date'] = pd.to_datetime(df['date'], utc=True) # utc conversion argument needed otherwise reading error is raised

# Splitting of the Date column
df['year'] = df['date'].dt.year
df['month'] = pd.to_datetime(df['date'], format='%m').dt.month_name()

df.head()

```
![image](https://user-images.githubusercontent.com/74253587/155994148-ed3b1e3c-01be-43a1-a52c-264cb7b38f0c.png)


After getting rid of the useless columns, we tackle the text processing using Natural Language Toolkit (nltk). 

```
nltk_ponct = nltk.RegexpTokenizer(r'\w{2,}') # tokenize (split words):
                                                        # {2,} : 2+ words
        
stopword = stopwords.words('english') # remove unuseful words (to, the, etc.)

wordnet_lemmatizer = WordNetLemmatizer() # reduce words to singular nouns

snowball_stemmer = SnowballStemmer('english') # find the root of the word

# Function
def nlp(text):
    '''Processes the text to extract the keywords needed for data analysis'''
    # keep original messages only (truncate '-----Original Message-----')
    keyword = '-----'
    if keyword in text:
        text=text[:text.index(keyword) + len(keyword)]    
    # switch to lower case letters
    text = text.lower()
    # split words and removes bridging
    text = nltk_ponct.tokenize(text)
    # remove unnecessary words
    text = [word for word in text if word not in stopword]
    # reduce words to singular nouns
    text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    # find the root of the word
    text = [snowball_stemmer.stem(word) for word in text]

    #return text : if we want the tokenized format (list)
    return ' '.join(text) # to get the str format
    

# apply the above cleaning function to our text column
df['clean_text'] = df['text'].map(nlp)



# merge from & clean_text columns and drop from & text columns
df['features'] = df['from'] + ' ' + df['clean_text']

# drop the columns 'subject' and 'body'
df.drop(['from', 'text', 'clean_text'], axis=1, inplace=True)

df.head()
```
![image](https://user-images.githubusercontent.com/74253587/155995110-68f94714-763f-4689-8620-5bb4e03f31e9.png)


## 4. Modeling with Bag of Words and TF-IDF then optimization of the best model
After encoding the `Target` column containing the class labels, we can compare several machine learning models in order to choose one, then tune its hyperparameters to optimize it. 

```
# split the data
features = df['features']
X = features
y = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
### Bag of Words
A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. It is called a "bag" of words because any information about the order or structure of words in the document is discarded.

```
# vecorization
vectorizer = CountVectorizer(min_df=5, max_features=5000)
X_train_bow = vectorizer.fit_transform(X_train)
X_train_bow = X_train_bow.toarray()

# modeling
models = [GaussianNB(), MultinomialNB(), DecisionTreeClassifier(), LinearSVC(), 
          AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=5),
         MLPClassifier(hidden_layer_sizes=(10,)), KNeighborsClassifier(), LGBMClassifier()]

names = ["Gaussian NB", "Multinomial NB", "Decision Tree", "SVC", "AdaBoost", "ANN", 'KNN', 'LGB']

acc_scores = []
f1_scores = []
exec_times = []

for model, name in zip(models, names):
    print(name)
    start = time.time()
    scoring = {
        'acc': 'accuracy',
        'f1_mac': 'f1_macro',
    }
    scores = cross_validate(model, X_train_bow, y_train, cv=10, n_jobs=4, scoring=scoring)
    training_time = (time.time() - start)
    print("accuracy: ", scores['test_acc'].mean())
    print("f1_score: ", scores['test_f1_mac'].mean())
    print("time (sec): ", training_time)
    print("\n")
    
    acc_scores.append(scores['test_acc'].mean())
    f1_scores.append(scores['test_f1_mac'].mean())
    exec_times.append(training_time)
    
acc_df['BoW'] = acc_scores
f1_df['BoW'] = f1_scores
acc_df['time'] = exec_times
acc_df
```
![image](https://user-images.githubusercontent.com/74253587/155996072-f6d43974-9fe6-42a8-b25f-30df2e6ec4d0.png)

### TF-IDF
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents . This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. 

```
# vectorization
vectorizer = TfidfVectorizer(min_df=5, max_features=5000)
X_train_tf = vectorizer.fit_transform(X_train)
X_train_tf = X_train_tf.toarray()

# modeling
models = [GaussianNB(), MultinomialNB(), DecisionTreeClassifier(), LinearSVC(), 
          AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=5),
         MLPClassifier(hidden_layer_sizes=(10,)), KNeighborsClassifier(), LGBMClassifier()]

names = ["Gaussian NB", "Multinomial NB", "Decision Tree", "SVC", "AdaBoost", "ANN", 'KNN', 'LGB']

acc_scores = []
f1_scores = []
exec_times = []

for model, name in zip(models, names):
    print(name)
    start = time.time()
    scoring = {
        'acc': 'accuracy',
        'f1_mac': 'f1_macro',
    }
    scores = cross_validate(model, X_train_tf, y_train, cv=10, n_jobs=4, scoring=scoring)
    training_time = (time.time() - start)
    print("accuracy: ", scores['test_acc'].mean())
    print("f1_score: ", scores['test_f1_mac'].mean())
    print("time (sec): ", training_time)
    print("\n")
    
    acc_scores.append(scores['test_acc'].mean())
    f1_scores.append(scores['test_f1_mac'].mean())
    exec_times.append(training_time)
    
acc_df['TfIdf'] = acc_scores
f1_df['TfIdf'] = f1_scores
acc_df['TfIdf_time'] = exec_times
acc_df
```
![image](https://user-images.githubusercontent.com/74253587/155998557-5ab7a109-4f83-4bbd-928c-2d361d083ef6.png)



We conclude that LinearSVC with TF-IDF has the best performance-duration ratio. The score is low, possibly due to the low data volume (only 879 rows for 13 classes). We decide to optimize the model by tuning the hyperparameters using GridSearchCV.

### Model tuning



# optimizing the "best" estimator

```
param_grid = [{'C': [1, 10, 100, 1000],
               'class_weight': ['balanced', 'None'],  
               'fit_intercept': ['True', 'False'], 
               'intercept_scaling': [1, 5, 10], 
               'tol': [0.1, 0.2, 0.4, 0.8, 1]
              }]

grid = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=8)

# vectorization
vectorizer = TfidfVectorizer(min_df=5, max_features=5000)
X_train_f = vectorizer.fit_transform(X_train)
X_test_f = vectorizer.transform(X_test)
X_train_f = X_train_f.toarray()
X_test_f = X_test_f.toarray()

# training
grid.fit(X_train_f, y_train)

# Best estimator/hyperparameters
model = grid.best_estimator_
model
```
`LinearSVC(C=1, class_weight='balanced', fit_intercept='True', intercept_scaling=5, tol=0.8)` is our best model. Let's check how it performs with test data. 

```
model.score(X_test_f, y_test)
```
`0.4034090909090909`. This shows the need to better train this model with more emails per theme. We can see this with the following confusion matrix showing that only themes 2 and 10 are the best estimated by our model: 

```
# Confusion matrix of correct/incorrect estimates
fig, ax = plt.subplots(figsize=(10, 10))

matrix_lr = plot_confusion_matrix(grid, X_test_f, y_test, cmap=plt.cm.Blues, ax=ax)

plt.title('Confusion matrix for LinearSVC')
plt.show(matrix_lr)
plt.show()
```
![image](https://user-images.githubusercontent.com/74253587/156000512-3c16eefc-392a-4a35-aa53-f84c32ceebcb.png)
