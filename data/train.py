import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from SSS.model import SentimentModel
from SSS.preprocessing import Preprocessor

preprocessor = Preprocessor('data/train.csv')
df = preprocessor.preprocess_data()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen = max_length, padding = 'post')

labels = pd.get_dummies(df['sentiment']).values
xtrain, xtest,ytrain,ytest = train_test_split(padded_sequences, labels, test_size = 0.1)


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = SentimentModel(vocab_size, embedding_dim, max_length)
model = model.create_model()
model.fit(xtrain, ytrain, epochs = 5, validation_data =(xtest,ytest))

model.save('models/sentiment_analysis_model.h5')