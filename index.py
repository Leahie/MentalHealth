import pandas as pd 
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
nltk.download("punkt")
from nltk.tokenize import word_tokenize

# Model Creation
from sklearn.feature_extraction.text import TfidfVectorizer # 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn .metrics import classification_report

#svc ml model - support vector classifier 
#huggingface for nlp

df = pd.read_csv("./mental_health.csv")

print(df)
msg=df.text

# Remove Special Characters
msg=msg.str.replace('[^a-zA-Z0-9]+'," ")
print(msg)

# NLTK, natural language tool kit --> hugging face 
stemmer=PorterStemmer() 
msg=msg.apply(lambda line:[stemmer.stem(token.lower()) for token in word_tokenize(line)]).apply(lambda token:" ".join(token))

msg=msg.apply(lambda line:[token for token in word_tokenize(line) if len(token)>2]).apply(lambda y:" ".join(y))

# Vectorization 
tf = TfidfVectorizer()
data_vec = tf.fit_transform(msg)
print(f"Vectorized vector: {data_vec}")

y=df['label'].values 
print(y)

x_train,x_test,y_train,y_test=train_test_split(data_vec,y,test_size=0.3,random_state=1)

sv=SVC()
# nb=GaussianNB()
rf=RandomForestClassifier()
ab= AdaBoostClassifier()
models=[sv,rf,ab]
for model in models:
  print(model)
  model.fit(x_train,y_train)
  y_pred=model.predict(x_test)
  print(classification_report(y_test,y_pred))