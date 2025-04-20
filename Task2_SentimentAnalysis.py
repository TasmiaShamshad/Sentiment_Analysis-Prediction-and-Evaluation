import numpy as np 
import pandas as pd 
import seaborn as sns 
import nltk
import matplotlib.pyplot as plt

plt.style.use("ggplot")

## Quick EDA
df = pd.read_csv(r"E:\Reviews.csv")
df.shape
df = df.head(500)
df.shape

ax = df['Score'].value_counts().sort_index().plot(kind= 'bar', title= "Reviews", figsize= (8,5))
plt.xlabel('Review stars by count')
plt.show()

##Tokeinzation 
example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
tagged[:10]
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

## SENTIMENT ANALYSIS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores(example)

from tqdm.notebook import tqdm
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()    

## VISUALISING THE SENTIMENT ANALYSIS RESULT

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], palette = 'Greens')
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], palette = 'Blues')
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], palette = 'Reds')
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

## TRAINING THE DATA SET
'''Train a classifier such as Logistic Regression predict sentiment.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv(r"E:\Reviews.csv")  # Update with your actual file path
df = df.head(500)  # Use the first 500 rows for testing

# Preprocessing: Vectorization of text using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Text'])  # Convert text data to numerical format
y = df['Score']  # Assuming 'Score' is the target variable (1 to 5 rating)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

## MODEL EVALUATION
'''Evaluate the model's performance using metrics like precision, recall, and F1-score.'''
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


