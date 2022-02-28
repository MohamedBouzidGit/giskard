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
