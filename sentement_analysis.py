import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze the sentiment of a single text input using VADER sentiment analysis.

    Parameters:
        text (str): The text input to analyze.

    Returns:
        sentiment (str): Overall sentiment as 'Positive', 'Negative', or 'Neutral'.
        scores (dict): Dictionary of sentiment scores.
    """
    sentiment_scores = sid.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, sentiment_scores

def analyze_multiple_texts(text_data):
    """
    Analyze multiple texts and store the results in a DataFrame.

    Parameters:
        text_data (list of str): List of text data to analyze.

    Returns:
        df (pd.DataFrame): DataFrame containing text, sentiment, and sentiment scores.
    """

    sentiments = []
    scores_list = []
    for text in text_data:
        sentiment, scores = analyze_sentiment(text)
        sentiments.append(sentiment)
        scores_list.append(scores)\
        
    df = pd.DataFrame({
        'Text': text_data,
        'Sentiment': sentiments,
        'Negative Score': [s['neg'] for s in scores_list],
        'Neutral Score': [s['neu'] for s in scores_list],
        'Positive Score': [s['pos'] for s in scores_list],
        'Compound Score': [s['compound'] for s in scores_list]
    })
    
    return df

def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiments.

    Parameters:
        df (pd.DataFrame): DataFrame containing text, sentiment, and sentiment scores.
    """
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], figsize=(8, 5))
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.show()

text_data = []
print("Enter text data for sentiment analysis (leave blank and press Enter to finish):")
while True:
    text = input("Enter text: ")
    if text == "":
        break
    text_data.append(text)

if text_data:
    df = analyze_multiple_texts(text_data)
    print("\nSentiment Analysis Results:")
    print(df)
    plot_sentiment_distribution(df)
else:
    print("No text data was entered.")
