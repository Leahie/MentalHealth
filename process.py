import pandas as pd
from unidecode import unidecode

df = pd.read_csv("Suicide_Detection.csv", index_col=0)
df.reset_index(drop=True, inplace=True)

# remove accented characters 
def remove_accent(text):
    return unidecode(text)

def process_text(text, ra=True):
    if ra: 
        text = remove_accent(text)
    return text

df['text_processed'] = df['text'][:20].apply(process_text)
print(df[:20])