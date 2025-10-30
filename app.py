
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
from textblob import TextBlob
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import random

# Download NLTK resources (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis ",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTitle {
        color: #1DA1F2;
        font-size: 3em !important;
    }
    .stSidebar {
        background-color: #EFF3F4;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("Twitter Sentiment Analysis ")
st.markdown("Analyze the sentiment of tweets from our demo dataset")

# Generate sample Twitter data
def generate_sample_data(query, tweet_count=100):
    topics = {
        "covid": ["vaccine", "lockdown", "mask", "pandemic", "social distancing", "covid-19"],
        "politics": ["election", "democracy", "policy", "government", "debate", "vote"],
        "sports": ["game", "team", "player", "score", "championship", "coach"],
        "technology": ["smartphone", "computer", "app", "software", "innovation", "digital"],
        "climate": ["global warming", "environment", "sustainability", "green energy", "pollution"]
    }
    
    # Generate usernames
    usernames = [f"user_{i}" for i in range(1, 20)]
    
    # Get relevant keywords for the query
    related_words = []
    for key, words in topics.items():
        if key.lower() in query.lower():
            related_words.extend(words)
    
    if not related_words:
        # If no topic matches, use technology as default
        related_words = topics["technology"]
    
    # Sample tweets
    positive_templates = [
        "I really love how {} is developing!",
        "Great news about {} today!",
        "The latest {} update is impressive.",
        "I'm so happy with the progress on {}.",
        "Fantastic developments in {} recently!",
        "{} has been a positive change in my life.",
        "I'm optimistic about the future of {}."
    ]
    
    negative_templates = [
        "I'm concerned about the latest {} news.",
        "The situation with {} is getting worse.",
        "I don't like how {} is being handled.",
        "The recent {} changes are disappointing.",
        "I'm worried about where {} is heading.",
        "{} has been causing problems lately.",
        "I'm frustrated with the state of {}."
    ]
    
    neutral_templates = [
        "Have you heard the latest about {}?",
        "What's your opinion on {}?",
        "Any updates on {}?",
        "The discussion about {} continues.",
        "{} is in the news again today.",
        "There's talk about {} in my community.",
        "People are talking about {} online."
    ]
    
    tweets_list = []
    
    for _ in range(tweet_count):
        sentiment_choice = random.choices(["positive", "neutral", "negative"], weights=[0.4, 0.3, 0.3])[0]
        
        if sentiment_choice == "positive":
            template = random.choice(positive_templates)
        elif sentiment_choice == "negative":
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)
        
        keyword = random.choice(related_words)
        text = template.format(keyword)
        
        # Add random hashtags occasionally
        if random.random() > 0.7:
            hashtags = random.sample(related_words, k=random.randint(1, 2))
            text += " " + " ".join([f"#{tag.replace(' ', '')}" for tag in hashtags])
        
        # Create tweet entry
        tweet = {
            'text': text,
            'created_at': datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 7), 
                                                                      hours=random.randint(0, 23), 
                                                                      minutes=random.randint(0, 59)),
            'username': random.choice(usernames),
            'name': "Demo User",
            'retweet_count': random.randint(0, 1000),
            'favorite_count': random.randint(0, 2000),
        }
        
        # Clean and analyze the text
        tweet['clean_text'] = clean_tweet(tweet['text'])
        tweet['sentiment'] = sentiment_choice  # We already know the sentiment
        tweet['polarity'] = get_polarity(tweet['clean_text'])
        tweet['subjectivity'] = get_subjectivity(tweet['clean_text'])
        
        tweets_list.append(tweet)
    
    return pd.DataFrame(tweets_list)

# Sidebar for parameters
with st.sidebar:
    st.header("Search Parameters")
    search_type = st.radio("Search by:", ("Query/Hashtag", "Username"))
    
    if search_type == "Query/Hashtag":
        query = st.text_input("Enter demo query or topic:", "technology")
    else:
        username = st.text_input("Enter demo username:", "user_1")
        query = username  # We'll use this for data generation
    
    tweet_count = st.slider("Number of tweets to analyze:", 10, 500, 100)
    
    st.header("Filtering")
    language = st.selectbox("Select language:", ["en"], index=0)
    
    # Add date range selector
    st.subheader("Date Range")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)
    since_date = st.date_input("Since date:", start_date)
    until_date = st.date_input("Until date:", end_date)
    
    run_analysis = st.button("Run Analysis")

# Helper Functions
def clean_tweet(tweet):
    """
    Clean tweet text by removing links, special characters, etc.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    """
    Classify sentiment of a tweet using TextBlob
    """
    # Create TextBlob object
    analysis = TextBlob(clean_tweet(tweet))
    
    # Set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def get_polarity(text):
    """Get polarity score from text"""
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text):
    """Get subjectivity score from text"""
    return TextBlob(text).sentiment.subjectivity

def get_top_words(tweets_list, n=10):
    """Get most frequent words in tweets"""
    stop_words = set(stopwords.words('english'))
    
    # Join all tweets into a single string and tokenize
    all_text = ' '.join([tweet for tweet in tweets_list])
    words = word_tokenize(all_text.lower())
    
    # Remove stop words and non-alphabetic tokens
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Get frequency distribution
    from collections import Counter
    freq_dist = Counter(filtered_words)
    
    return freq_dist.most_common(n)

def generate_wordcloud(tweets_list):
    """Generate wordcloud from tweets"""
    stop_words = set(stopwords.words('english'))
    all_text = ' '.join([tweet for tweet in tweets_list])
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         stopwords=stop_words, max_words=100).generate(all_text)
    
    # Display the wordcloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Main function to analyze demo tweets
def analyze_twitter_data():
    # Generate sample data based on query
    with st.spinner('Generating demo tweet data...'):
        df = generate_sample_data(query, tweet_count)
        
        # Filter by date if needed
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        df = df[(df['date'] >= since_date) & (df['date'] <= until_date)]
    
    if df.empty:
        st.warning("No tweets found matching your criteria.")
        return
    
    # Display Results
    st.header("Analysis Results")
    
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tweets Analyzed", len(df))
    with col2:
        positive_pct = round((df['sentiment'] == 'positive').mean() * 100, 2)
        st.metric("Positive Sentiment", f"{positive_pct}%")
    with col3:
        negative_pct = round((df['sentiment'] == 'negative').mean() * 100, 2)
        st.metric("Negative Sentiment", f"{negative_pct}%")
    
    # Plot sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                color='Sentiment',
                color_discrete_map={'positive':'#4CAF50', 'neutral':'#FFC107', 'negative':'#F44336'},
                hole=0.4)
    fig.update_layout(title_text='Tweet Sentiment Breakdown')
    st.plotly_chart(fig)
    
    # Plot polarity and subjectivity
    st.subheader("Polarity and Subjectivity")
    fig = px.scatter(df, x='polarity', y='subjectivity', color='sentiment',
                   color_discrete_map={'positive':'#4CAF50', 'neutral':'#FFC107', 'negative':'#F44336'},
                   title='Polarity vs Subjectivity',
                   labels={'polarity': 'Polarity (-1 to 1)', 'subjectivity': 'Subjectivity (0 to 1)'})
    st.plotly_chart(fig)
    
    # Plot sentiment over time
    st.subheader("Sentiment Over Time")
    sentiment_by_date = df.groupby([pd.to_datetime(df['created_at']).dt.date, 'sentiment']).size().reset_index(name='count')
    
    fig = px.line(sentiment_by_date, x='created_at', y='count', color='sentiment',
                color_discrete_map={'positive':'#4CAF50', 'neutral':'#FFC107', 'negative':'#F44336'},
                title='Sentiment Trends Over Time')
    st.plotly_chart(fig)
    
    # Display top words
    st.subheader("Top Words Used")
    top_words = get_top_words(df['clean_text'].tolist())
    
    # Convert to DataFrame for display
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.table(top_words_df)
    
    with col2:
        fig = px.bar(top_words_df, x='Word', y='Frequency', 
                   title='Most Frequent Words',
                   color='Frequency', color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig)
    
    # Generate and display word cloud
    st.subheader("Word Cloud")
    wordcloud_fig = generate_wordcloud(df['clean_text'].tolist())
    st.pyplot(wordcloud_fig)
    
    # Display sample tweets
    st.subheader("Sample Tweets")
    
    # Create tabs for different sentiments
    tab1, tab2, tab3 = st.tabs(["Positive Tweets", "Neutral Tweets", "Negative Tweets"])
    
    with tab1:
        positive_df = df[df['sentiment'] == 'positive'].head(5)
        if not positive_df.empty:
            for _, row in positive_df.iterrows():
                st.write(f"**@{row['username']}**: {row['text']}")
                st.write(f"*Polarity: {row['polarity']:.2f}, Subjectivity: {row['subjectivity']:.2f}*")
                st.write("---")
        else:
            st.write("No positive tweets found.")
    
    with tab2:
        neutral_df = df[df['sentiment'] == 'neutral'].head(5)
        if not neutral_df.empty:
            for _, row in neutral_df.iterrows():
                st.write(f"**@{row['username']}**: {row['text']}")
                st.write(f"*Polarity: {row['polarity']:.2f}, Subjectivity: {row['subjectivity']:.2f}*")
                st.write("---")
        else:
            st.write("No neutral tweets found.")
    
    with tab3:
        negative_df = df[df['sentiment'] == 'negative'].head(5)
        if not negative_df.empty:
            for _, row in negative_df.iterrows():
                st.write(f"**@{row['username']}**: {row['text']}")
                st.write(f"*Polarity: {row['polarity']:.2f}, Subjectivity: {row['subjectivity']:.2f}*")
                st.write("---")
        else:
            st.write("No negative tweets found.")
    
    # Display raw data with option to expand
    with st.expander("View Raw Data"):
        st.dataframe(df)
        
        # Option to download the data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"twitter_sentiment_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )

# Display notice about demo mode
st.info("""
**DEMO MODE:** Twitter Sentiment Analysis is a Natural Language Processing (NLP) 
        technique used to determine the sentiment of tweets (Positive, Negative, or Neutral).
         This demo allows users to input a tweet and analyzes its sentiment using TextBlob.
""")

# Run the app
if 'run_analysis' in locals() and run_analysis:
    analyze_twitter_data()
else:
    # Display initial instructions
    st.info("Configure the search parameters in the sidebar and click 'Run Analysis' to begin.")
    
    # Show features of the app
    with st.expander("Features of this app"):
        st.markdown("""
        This Twitter Sentiment Analysis demo app provides the following features:
        
        - Search by simulated query/hashtag or username
        - Filter by date range
        - Sentiment analysis using TextBlob
        - Visualization of sentiment distribution
        - Polarity and subjectivity analysis
        - Sentiment trends over time
        - Word frequency analysis
        - Interactive word cloud
        - Sample tweets by sentiment category
        - Raw data export
        User Input: The user enters a tweet or any text.
        Sentiment Analysis: The tool processes the text and assigns a sentiment score.
                                                                       
            - Output: Based on the score, the sentiment is classified as:
            - Positive 😊 (Polarity > 0)
            - Neutral 😐 (Polarity = 0)
            - Negative 😡 (Polarity < 0)
        



        Note: This is a demonstration using simulated data. For real Twitter data analysis,
        you would need to use the Twitter API with proper credentials.
        """)
