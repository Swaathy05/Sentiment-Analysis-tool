import pandas as pd
import speech_recognition as sr
import surprise as sur
from textblob import TextBlob
from surprise import KNNWithMeans
from surprise import Reader
import numpy as np
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random
import pygame
import re

df = pd.read_csv("my python folder/newnmdata.csv")

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def analyze_sentiment_from_dataset():
    print("Analyzing sentiment from dataset:")
    for index, row in df.iterrows():
        sentiment = analyze_sentiment(row['reviews.text'])
        print(f"Sentiment of row {index + 1}: {sentiment}")
        print("Text Data:")
        print(row['reviews.text'])
        print()

def audio_to_text_and_sentiment():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        print("Processing...")
        try:
            text = recognizer.recognize_google(audio_data)
            print("Text from audio:", text)
            sentiment = analyze_sentiment(text)
            print("Sentiment from audio input:", sentiment)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

def recommendation():
    print("Number of unique users in Raw data=", df["id"].nunique())
    no_of_rated_products_per_user = df.groupby(by='id')['reviews.rating'].count().sort_values(ascending=False)
    new_df = df.groupby("asins").filter(lambda x: x['reviews.rating'].count() >= 50)
    no_of_ratings_per_product = new_df.groupby(by='asins')['reviews.rating'].count().sort_values(ascending=False)
    ratings_mean_count = pd.DataFrame(new_df.groupby('asins')['reviews.rating'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('asins')['reviews.rating'].count())
    popular_products = pd.DataFrame(new_df.groupby('asins')['reviews.rating'].count())
    most_popular = popular_products.sort_values('reviews.rating', ascending=False)
    new_df.columns
    new_df = new_df[['id', 'asins', 'reviews.rating']]
    reader = Reader(rating_scale=(1, 5))
    data = sur.Dataset.load_from_df(new_df, reader)
    trainset, testset = train_test_split(data, test_size=0.3, random_state=10)
    algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)
    test_pred = algo.test(testset)
    print("Item-based Model : Test Set")
    new_df1 = new_df.head(2000)
    ratings_matrix = new_df1.pivot_table(values='reviews.rating', index='id', columns='asins', fill_value=0)
    X = ratings_matrix.T
    SVD = TruncatedSVD(n_components=8)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    X.index[2]
    i = "B01LW1MS9C"
    product_names = list(X.index)
    product_ID = product_names.index(i)
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    Recommend = list(X.index[correlation_product_ID < 0.65])
    print('Recommendations for customer id')
    print(Recommend[0:2])

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive', '😊'
    elif polarity == 0:
        return 'Neutral', '😐'
    else:
        return 'Negative', '😢'

sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

def update_sentiment_counts(row):
    sentiment, emoji = analyze_sentiment(row['reviews.text'])
    sentiment_counts[sentiment] += 1
    plt.clf()
    plt.subplot(2, 3, 1)
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['lightgreen', 'lightgray', 'lightcoral'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')
    positive_emoji = '😊'
    neutral_emoji = '😐'
    negative_emoji = '😢'
    for sentiment, count in sentiment_counts.items():
        if sentiment == 'Positive':
            emoji = positive_emoji
        elif sentiment == 'Neutral':
            emoji = neutral_emoji
        else:
            emoji = negative_emoji
        for _ in range(count):
            x_pos = random.uniform(-0.2, 0.2)
            y_pos = random.uniform(0, count)
            plt.text(sentiment, count, emoji, fontsize=20, ha='center', va='center')

    plt.subplot(2, 3, 2)
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['lightgreen', 'lightgray', 'lightcoral'])
    plt.title('Pie Chart')

    plt.subplot(2, 3, 3)
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['lightgreen', 'lightgray', 'lightcoral'], wedgeprops=dict(width=0.4))
    plt.title('Donut Chart')

    plt.subplot(2, 3, 4)
    sns.histplot(df['reviews.rating'], kde=True, color='skyblue')
    plt.title('Histogram')

    plt.subplot(2, 3, 5)
    sns.scatterplot(x='reviews.rating', y='reviews.numHelpful', data=df, hue='reviews.rating', palette='viridis')
    plt.title('Scatter Plot')

    plt.subplot(2, 3, 6, polar=True)
    sentiment_angles = {'Positive': 120, 'Neutral': 240, 'Negative': 0}
    sentiment_colors = {'Positive': 'lightgreen', 'Neutral': 'lightgray', 'Negative': 'lightcoral'}
    ax = plt.gca()
    for sentiment, count in sentiment_counts.items():
        ax.bar(np.radians(sentiment_angles[sentiment]), count, width=np.radians(40), color=sentiment_colors[sentiment], alpha=0.7, label=sentiment)

    plt.title('Polar Chart')
    plt.legend(loc='upper right')

def animate(i):
    if i < len(df):
        update_sentiment_counts(df.iloc[i])

def classify_text(text):
    sentiment, emoji = analyze_sentiment(text)
    sentiment_counts = df['reviews.text'].apply(lambda x: analyze_sentiment(x)[0]).value_counts(normalize=True)
    positive_percentage = sentiment_counts.get('Positive', 0) * 100
    neutral_percentage = sentiment_counts.get('Neutral', 0) * 100
    negative_percentage = sentiment_counts.get('Negative', 0) * 100
    pygame.init()
    screen_width = 600
    screen_height = 200
    emoji_size = 100
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Emoji Animation")
    positive_emoji = pygame.image.load("positive_emoji.png")
    neutral_emoji = pygame.image.load("neutral_emoji.png")
    negative_emoji = pygame.image.load("negative_emoji.png")
    positive_emoji = pygame.transform.scale(positive_emoji, (emoji_size, emoji_size))
    neutral_emoji = pygame.transform.scale(neutral_emoji, (emoji_size, emoji_size))
    negative_emoji = pygame.transform.scale(negative_emoji, (emoji_size, emoji_size))
    screen.blit(positive_emoji, (50, 50))
    screen.blit(neutral_emoji, (250, 50))
    screen.blit(negative_emoji, (450, 50))
    font = pygame.font.Font(None, 36)
    text_surface = font.render(f"{positive_percentage:.2f}%", True, (255, 255, 255))
    screen.blit(text_surface, (50, 170))
    text_surface = font.render(f"{neutral_percentage:.2f}%", True, (255, 255, 255))
    screen.blit(text_surface, (250, 170))
    text_surface = font.render(f"{negative_percentage:.2f}%", True, (255, 255, 255))
    screen.blit(text_surface, (450, 170))
    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

def analyze_sentiment_with_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub(r'', text)
    analysis = TextBlob(cleaned_text)
    polarity = analysis.sentiment.polarity
    emoji_sentiment = analyze_emoji_sentiment(text)
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def analyze_emoji_sentiment(text):
    if any(char in text for char in emoji_pattern.findall(text)):
        if any(char in text for char in [u"\U0001F600", u"\U0001F601", u"\U0001F602", u"\U0001F603",
                                         u"\U0001F604", u"\U0001F605", u"\U0001F606", u"\U0001F607",
                                         u"\U0001F609", u"\U0001F60A", u"\U0001F60B", u"\U0001F60E",
                                         u"\U0001F60D", u"\U0001F618", u"\U0001F617", u"\U0001F619",
                                         u"\U0001F61A", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B",
                                         u"\U0001F61E", u"\U0001F61F", u"\U0001F620", u"\U0001F621",
                                         u"\U0001F622", u"\U0001F623", u"\U0001F624", u"\U0001F625",
                                         u"\U0001F626", u"\U0001F627", u"\U0001F628", u"\U0001F629",
                                         u"\U0001F62A", u"\U0001F62B", u"\U0001F62C", u"\U0001F62D",
                                         u"\U0001F62E", u"\U0001F62F", u"\U0001F630", u"\U0001F631",
                                         u"\U0001F632", u"\U0001F633", u"\U0001F634", u"\U0001F635",
                                         u"\U0001F636", u"\U0001F637", u"\U0001F641", u"\U0001F642",
                                         u"\U0001F643", u"\U0001F644"]):
            return 'Positive'
        elif any(char in text for char in [u"\U0001F614", u"\U0001F615", u"\U0001F616", u"\U0001F61F",
                                           u"\U0001F624", u"\U0001F62A", u"\U0001F62D", u"\U0001F629",
                                           u"\U0001F622", u"\U0001F620", u"\U0001F621", u"\U0001F63F",
                                           u"\U0001F494", u"\U0001F62B", u"\U0001F622", u"\U0001F625",
                                           u"\U0001F62C", u"\U0001F628", u"\U0001F630", u"\U0001F613",
                                           u"\U0001F62F", u"\U0001F635", u"\U0001F631", u"\U0001F640",
                                           u"\U0001F63E", u"\U0001F63B", u"\U0001F629", u"\U0001F612",
                                           u"\U0001F61E", u"\U0001F623", u"\U0001F61F", u"\U0001F626",
                                           u"\U0001F627", u"\U0001F628", u"\U0001F630", u"\U0001F62B",
                                           u"\U0001F971", u"\U0001F624", u"\U0001F634", u"\U0001F64D",
                                           u"\U0001F615", u"\U0001F641", u"\U0001F612", u"\U0001F615",
                                           u"\U0001F614", u"\U0001F623", u"\U0001F613", u"\U0001F629",
                                           u"\U0001F62D", u"\U0001F62B", u"\U0001F622", u"\U0001F62C",
                                           u"\U0001F630", u"\U0001F631", u"\U0001F636", u"\U0001F610",
                                           u"\U0001F611", u"\U0001F623", u"\U0001F625", u"\U0001F629",
                                           u"\U0001F92E", u"\U0001F635", u"\U0001F62F", u"\U0001F641",
                                           u"\U0001F640", u"\U0001F63E", u"\U0001F971", u"\U0001F628",
                                           u"\U0001F625", u"\U0001F61E", u"\U0001F632", u"\U0001F633",
                                           u"\U0001F97A", u"\U0001F624", u"\U0001F62A", u"\U0001F637",
                                           u"\U0001F912", u"\U0001F915", u"\U0001F922", u"\U0001F92C",
                                           u"\U0001F92D", u"\U0001F92E", u"\U0001F92F", u"\U0001F925",
                                           u"\U0001F928", u"\U0001F637", u"\U0001F912", u"\U0001F915",
                                           u"\U0001F922", u"\U0001F92C", u"\U0001F92D", u"\U0001F92E",
                                           u"\U0001F92F", u"\U0001F925", u"\U0001F928"]):
            return 'Negative'
    return 'Neutral'

def menu():
    while True:
        print("\nMain menu")
        print("1. Analyze sentiment from dataset")
        print("2. Audio to text and sentiment analysis")
        print("3. Personalized Recommendation")
        print("4. Analyze sentiment with emojis")
        print("5. Innovative sentiment analysis")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            print("The sentiments of the reviews from Dataset: \n")
            analyze_sentiment_from_dataset()
            print()
        elif choice == '2':
            print("\nAnalyzing sentiment from live audio input: ")
            audio_to_text_and_sentiment()
            print()
        elif choice == '3':
            recommendation()
            print()
        elif choice == '4':
            text = input("Enter text with emojis: ")
            sentiment = analyze_sentiment_with_emoji(text)
            print("Sentiment with emojis:", sentiment)
            print()
        elif choice == '5':
            fig, ax = plt.subplots()
            ani = animation.FuncAnimation(fig, animate, interval=500, frames=len(df))
            plt.tight_layout()
            plt.show()
            print()
            classify_text("")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

menu()
