import os
import pickle
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(page_title="Text Visualization and Sentiment Analysis", layout="wide")

# Function to download files from URL
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Load tokenizer
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_tokenizer():
    tokenizer_path = 'tokenizer.pkl'
    if not os.path.exists(tokenizer_path):
        download_file('https://github.com/AH-ML/Text-Analysis-And-Sentiment-Prediction/raw/master/tokenizer.pkl', tokenizer_path)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Load model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_sentiment_model():
    model_path = 'lstm.h5'
    if not os.path.exists(model_path):
        download_file('https://github.com/AH-ML/Text-Analysis-And-Sentiment-Prediction/raw/master/lstm.h5', model_path)
    model = load_model(model_path)
    max_length = 32
    return model, max_length

# Function to predict sentiment
def predict_sentiment(input_text, tokenizer, model, max_length):
    # Preprocess the input text
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    # Get the prediction
    prediction = model.predict(padded_input_sequence)

    # Convert the prediction to sentiment label
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_label_index = np.argmax(prediction)
    predicted_sentiment = sentiment_labels[predicted_label_index]

    return predicted_sentiment

# Function to visualize word length distribution
def visualize_word_length_distribution(text):
    word_lengths = [len(word) for word in text.split()]
    word_lengths_counts = {length: word_lengths.count(length) for length in set(word_lengths)}
    sorted_word_lengths = sorted(word_lengths_counts.items(), key=lambda x: x[0])

    colorscale = [[0.0, 'rgb(255, 0, 0)'], [0.5, 'rgb(0, 255, 0)'], [1.0, 'rgb(0, 0, 255)']]  # Cyberpunk palette
    bar_trace = go.Bar(
        x=[length for length, count in sorted_word_lengths],
        y=[count for length, count in sorted_word_lengths],
        marker=dict(color=[word_lengths_counts[length] / max(word_lengths_counts.values()) for length, _ in sorted_word_lengths],
                    colorscale=colorscale),
        hovertemplate='Word Length: %{x}<br>Count: %{y}<extra></extra>'
    )

    light_effect_trace = go.Scatter(
        x=[length for length, count in sorted_word_lengths],
        y=[count * 1.05 for length, count in sorted_word_lengths],
        mode='lines',
        line=dict(color='rgb(0, 255, 255)', width=5),
        hoverinfo='skip'
    )

    layout = go.Layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(title='Word Length', titlefont=dict(color='rgb(0, 255, 255)')),
        yaxis=dict(title='Count', titlefont=dict(color='rgb(0, 255, 255)')),
        font=dict(color='rgb(0, 255, 255)'),
        title='Word Length Distribution',
        title_font=dict(color='rgb(0, 255, 255)', size=20)
    )

    fig = go.Figure(data=[bar_trace, light_effect_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

# Function to visualize sentence length distribution
def visualize_sentence_length_distribution(text):
    sentence_lengths = [len(sentence.split()) for sentence in text.split('.')]
    sorted_sentence_lengths = sorted(set(sentence_lengths))

    colorscale = [[0.0, 'rgb(255, 0, 0)'], [0.5, 'rgb(0, 255, 0)'], [1.0, 'rgb(0, 0, 255)']]  # Cyberpunk palette
    bar_trace = go.Bar(
        x=sorted_sentence_lengths,
        y=[sentence_lengths.count(length) for length in sorted_sentence_lengths],
        marker=dict(color=[sentence_lengths.count(length) / max(sentence_lengths) for length in sorted_sentence_lengths],
                    colorscale=colorscale),
        hovertemplate='Sentence Length: %{x}<br>Count: %{y}<extra></extra>'
    )

    light_effect_trace = go.Scatter(
        x=sorted_sentence_lengths,
        y=[sentence_lengths.count(length) * 1.05 for length in sorted_sentence_lengths],
        mode='lines',
        line=dict(color='rgb(0, 255, 255)', width=5),
        hoverinfo='skip'
    )

    layout = go.Layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(title='Sentence Length', titlefont=dict(color='rgb(0, 255, 255)')),
        yaxis=dict(title='Count', titlefont=dict(color='rgb(0, 255, 255)')),
        font=dict(color='rgb(0, 255, 255)'),
        title='Sentence Length Distribution',
        title_font=dict(color='rgb(0, 255, 255)', size=20)
    )

    fig = go.Figure(data=[bar_trace, light_effect_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

# Function to generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(background_color='black', width=800, height=400, max_words=200, colormap='Blues').generate(text)
    fig = go.Figure(go.Image(z=np.dstack((wordcloud.to_array(), wordcloud.to_array(), wordcloud.to_array()))))
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        title='Word Cloud for Input Text',
        title_font=dict(color='rgb(0, 255, 255)', size=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to visualize word frequency
def visualize_word_frequency(text):
    words = text.split()
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    colorscale = [[0.0, 'rgb(255, 0, 0)'], [0.5, 'rgb(0, 255, 0)'], [1.0, 'rgb(0, 0, 255)']]  # Cyberpunk palette
    bar_trace = go.Bar(
        x=[word for word, count in sorted_word_counts[:20]],
        y=[count for word, count in sorted_word_counts[:20]],
        marker=dict(color=[count / sorted_word_counts[0][1] for word, count in sorted_word_counts[:20]],
                    colorscale=colorscale),
        hovertemplate='Word: %{x}<br>Count: %{y}<extra></extra>'
    )

    light_effect_trace = go.Scatter(
        x=[word for word, count in sorted_word_counts[:20]],
        y=[count * 1.05 for word, count in sorted_word_counts[:20]],
        mode='lines',
        line=dict(color='rgb(0, 255, 255)', width=5),
        hoverinfo='skip'
    )

    layout = go.Layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(title='Word', titlefont=dict(color='rgb(0, 255, 255)'), tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(title='Count', titlefont=dict(color='rgb(0, 255, 255)')),
        font=dict(color='rgb(0, 255, 255)'),
        title='Word Frequency',
        title_font=dict(color='rgb(0, 255, 255)', size=20)
    )

    fig = go.Figure(data=[bar_trace, light_effect_trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title('Text Visualization and Sentiment Analysis')

    # App information column
    with st.sidebar:
        st.markdown("""
        This app provides various visualizations for analyzing text data, including:

        - **Word Length Distribution**: A histogram showing the distribution of word lengths in the text.
        - **Sentence Length Distribution**: A histogram showing the distribution of sentence lengths in the text.
        - **Word Cloud**: A visual representation of the most frequent words in the text.
        - **Word Frequency**: A bar chart displaying the frequency of the top 20 most common words in the text.

        Additionally, the app includes a sentiment analysis feature that predicts the sentiment (Negative, Neutral, or Positive) of the input text using a pre-trained deep learning model.

        To get started, simply enter your text in the text area below and click the 'Predict' button. The app will generate the visualizations and provide the predicted sentiment for your input text.
        """)

    input_text = st.text_area("Enter text:")

    if st.button("Predict"):
        if input_text:
            if input_text.lower() != 'q':
                tokenizer = load_tokenizer()
                model, max_length = load_sentiment_model()
                predicted_sentiment = predict_sentiment(input_text, tokenizer, model, max_length)
                st.write("Predicted Sentiment:", predicted_sentiment)
                visualize_word_length_distribution(input_text)
                visualize_sentence_length_distribution(input_text)
                generate_word_cloud(input_text)
                visualize_word_frequency(input_text)
            else:
                st.write("Exiting...")

if __name__ == "__main__":
    main()
