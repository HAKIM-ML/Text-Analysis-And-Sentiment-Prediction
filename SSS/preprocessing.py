import re

import pandas as pd


class Preprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def clean_text(self, text):
        text = re.sub(r'[a-zA-Z\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', '', text).strip()
        
        return text
    
    def preprocess_data(self):
        df = pd.read_csv(self.file_path, encoding='ISO-8859-1')
        df.dropna(inplace=True)
        df['text'] = df['text'].apply(self.clean_text)
        return df
    
    