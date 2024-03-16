import pandas as pd 
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer # 

#svc ml model - support vector classifier 
#huggingface for nlp

df = pd.read_csv("mental_health.csv")

print(df)
msg=df.text

# Remove Special Characters
msg=msg.str.replace('[^a-zA-Z0-9]+'," ")
print(msg)

# NLTK, natural language tool kit --> hugging face 
stemmer=PorterStemmer() 
msg=msg.apply(lambda line:[stemmer.stem(token.lower()) for token in word_tokenize(line)]).apply(lambda token:" ".join(token))

msg=msg.apply(lambda line:[token for token in word_tokenize(line) if len(token)>2]).apply(lambda y:" ".join(y))

