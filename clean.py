import pandas as pd 
import re

df = pd.read_csv('./.data/Suicide_Detection.csv', index_col = 0)
df.reset_index(drop=True, inplace=True)


def has_emojis(text):
    # Regular expression pattern to match emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

df = df[df['text'].apply(lambda x: not has_emojis(x))]

df = df[df['text'].apply(lambda x: len(x.split())!=0)]
df.reset_index(drop=True, inplace=True)

df = df[df['text'].apply(lambda x: len(x.split())<=62)]
df.reset_index(drop=True, inplace=True)

df = df[df['class'].apply(lambda x: x=="suicide" or x=="non-suicide")]
df.reset_index(drop=True, inplace=True)

df.to_csv('./.data/Suicide_Detection_Final_Clean.csv')