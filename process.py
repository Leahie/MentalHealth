import pandas as pd
from unidecode import unidecode
from spellchecker import SpellChecker
import contractions
import re
import en_core_web_sm
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("./.data/Suicide_Detection.csv", index_col=0)
df.reset_index(drop=True, inplace=True)
i = 1

# Remove Accented Characters 
def remove_accent(text):
    return unidecode(text)

# Expand Contractions 
def expand_contractions(text):
    return contractions.fix(text)

# Convert to Lowercase
def convert_lower(text):
    return text.lower()

# Remove Special Digits
def special_figures(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove symbols, digits, and special characters
    text = re.sub(r'[^\w\s]', '', text)

    return text

# whitespace 
def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text)

# Word Length 
def word_length(text):
    pattern = re.compile(r'(.)\1+')
    return pattern.sub(r'\1\1', text)

# Spell Correction 
def spell_correct(text):
    spell = SpellChecker()
    return ' '.join(spell.correction(word) for word in text.split() if spell.correction(word)!=None)
    # return spell.correction(text)

# Space Correction
def lemmatiz(txt):
    nlp = en_core_web_sm.load()

    doc = nlp(txt)
    
    x=[token.lemma_ for token in doc]
    lemmatized_text = " ".join(x)

    return lemmatized_text

# Remove Stopworks 
nltk.download('stopwords')
def remove_stopwords(text):
    nltk_stopwords = set(stopwords.words('english'))
    custom_stopwords = nltk_stopwords - {"no", "not"}
    filtered_text = ' '.join(word for word in text.split() if word not in custom_stopwords)
    
    return filtered_text

# Lemmatization



def process_text(text, ra=True, ec=True, low=True, sf = True, rw=True, wl = True, lemma=True, sc=True, sw=True):
    if ra: 
        text = remove_accent(text)
    if ec:
        text = expand_contractions(text)
    if low:
        text = convert_lower(text)
    if sf:
        text = special_figures(text)
    if rw:
        text = remove_whitespace(text)
    if wl: 
        text = word_length(text)
    if sw: 
        text = remove_stopwords(text)
    if lemma: 
        text = lemmatiz(text)
    if sc: 
        text = spell_correct(text)

    print(text)
    global i 
    print(i/df.shape[0], i, "out of",df.shape[0] )
    i+=1
    return text

print(df.shape)
df = df[df['text'].apply(lambda x: len(x.split())<=100)]
df.reset_index(drop=True, inplace=True)
df['text_processed'] = df['text'].apply(process_text)
df.to_csv('.data/Suicide_Detection_Clean.csv', index=False)