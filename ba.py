import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


df = pd.read_csv('Fake_Real_data.csv')
df['Label'] = df['label'].map({'Fake': 0, 'Real': 1})

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
x = pad_sequences(sequences, maxlen=300)
y = np.array(df['Label'])
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=300),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1)

def classify(article):
    sequence = tokenizer.texts_to_sequences([article])
    padded = pad_sequences(sequence, maxlen=300)
    pred = model.predict(padded, verbose=0)[0][0]
    return 'REAL' if pred > 0.5 else 'FAKE'


user = st.text_input('Enter News')
if user:
    result= classify(user)
    st.write(f'the news is:{result}')
