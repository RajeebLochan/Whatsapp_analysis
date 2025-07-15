from urlextract import URLExtract
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud 
import pandas as pd
import emoji
import re
def fetch_stats(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    num_messages = df.shape[0]
    words = sum(df["user_message"].str.split().str.len())
    num_media_messages = df[df["user_message"] == "<Media omitted>"].shape[0]
    extractor = URLExtract()
    links = sum(df["user_message"].apply(lambda message: len(extractor.find_urls(message))))
    
    return num_messages, words, num_media_messages, links


def most_activity(df):
    
    user_activity = df['user'].value_counts().reset_index()
    user_activity = user_activity[user_activity['user'] != 'system']  # Exclude group notifications
    user_activity.columns = ['user', 'messages']
    return user_activity

def word_cloud(selected_user,df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    # Remove unwanted words 'null' and '<Media omitted>' from user_message column
    df['user_message'] = df['user_message'].replace(['null', '<Media omitted>'], '', regex=True)
    df = df[df['user_message'] != '']  # Remove empty messages
    all_words = ' '.join(df['user_message'].dropna().tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    return wc

# def common_words(selected_user, df):
#     if selected_user != "All Users":
#         df = df[df["user"] == selected_user]
    
#     # Remove unwanted words 'null' and '<Media omitted>' from user_message column
#     df['user_message'] = df['user_message'].replace(['null', '<Media omitted>'], '', regex=True)
#     df = df[df['user_message'] != '']  # Remove empty messages
    
#     all_words = ' '.join(df['user_message'].dropna().tolist())
#     word_list = all_words.split()
    
#     common_words = pd.Series(word_list).value_counts().reset_index()
#     common_words.columns = ['word', 'count']
    
#     return common_words.head(10)  # Return top 10 common words

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def common_words(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    # Clean message column
    df['user_message'] = df['user_message'].replace(['null', '<Media omitted>'], '', regex=True)
    df = df[df['user_message'] != '']  # Remove empty messages

    # Join all messages into one string
    all_words = ' '.join(df['user_message'].dropna().tolist())
    word_list = all_words.lower().split()  # Lowercase for uniformity

    # Remove stopwords
    filtered_words = [word for word in word_list if word not in stop_words and word.isalpha()]

    # Count most common words
    common_words = pd.Series(filtered_words).value_counts().reset_index()
    common_words.columns = ['word', 'count']

    return common_words.head(10)
    
def common_emojis(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    # Clean message column
    df['user_message'] = df['user_message'].replace(['null', '<Media omitted>'], '', regex=True)
    df = df[df['user_message'] != '']  # Remove empty messages

    # Extract emojis from messages
    emoji_list = []
    for message in df['user_message'].dropna():
        emoji_list.extend([char for char in message if emoji.is_emoji(char)])

    # Count most common emojis
    common_emojis = pd.Series(emoji_list).value_counts().reset_index()
    common_emojis.columns = ['emoji', 'count']

    return common_emojis.head(10)


# def chat_duration(df):
#     df['date'] = pd.to_datetime(df['date'])
#     start_date = df['date'].min()
#     end_date = df['date'].max()
#     duration = (end_date - start_date).days
#     return duration + "days"

# def message_per_day(df):
#     df['date'] = pd.to_datetime(df['date'])
#     messages_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='messages')
#     messages_per_day.columns = ['date', 'messages']
#     return messages_per_day

# def message_per_month(df):
#     df['date'] = pd.to_datetime(df['date'])
#     messages_per_month = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='messages')
#     messages_per_month.columns = ['month', 'messages']
#     return messages_per_month

# def message_per_year(df):
#     df['date'] = pd.to_datetime(df['date'])
#     messages_per_year = df.groupby(df['date'].dt.year).size().reset_index(name='messages')
#     messages_per_year.columns = ['year', 'messages']
#     return messages_per_year

def message_per_day_user(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    #remove system colums
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'duration': ['No data available']})
    df['date'] = pd.to_datetime(df['date'])
    messages_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='messages')
    messages_per_day.columns = ['date', 'messages']
    return messages_per_day

def message_per_month_user(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    #remove system colums
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'duration': ['No data available']})
    df['date'] = pd.to_datetime(df['date'])
    messages_per_month = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='messages')
    messages_per_month.columns = ['month', 'messages']
    return messages_per_month

def message_per_year_user(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    #remove system colums
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'duration': ['No data available']})
    df['date'] = pd.to_datetime(df['date'])
    messages_per_year = df.groupby(df['date'].dt.year).size().reset_index(name='messages')
    messages_per_year.columns = ['year', 'messages']
    return messages_per_year

def chat_duration(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    df['date'] = pd.to_datetime(df['date'])
    #remove system colums
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'duration': ['No data available']})   
    start_date = df['date'].min()
    end_date = df['date'].max()
    duration = (end_date - start_date).days
    return f"{duration} days"

def average_messages_per_day(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    #remove system colums
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'duration': ['No data available']})
    df['date'] = pd.to_datetime(df['date'])
    messages_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='messages')
    average_messages = messages_per_day['messages'].mean()
    return average_messages

def activity_time_of_day(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    # Remove system messages
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'time': ['No data available']})
    
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['date'].dt.hour
    activity_time = df.groupby('time').size().reset_index(name='messages')
    activity_time.columns = ['time', 'messages']
    
    return activity_time

def average_response_time(selected_user, df):
    if selected_user != "All Users":
        df = df[df["user"] == selected_user]
    
    # Remove system messages
    df = df[df['user'] != 'system']
    if df.empty: 
        return pd.DataFrame({'time': ['No data available']})
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    # Calculate response time
    df['response_time'] = df['date'].diff().dt.total_seconds() / 60  # Convert to minutes
    average_response = df['response_time'].median()
    
    return average_response 



def preprocess_text(text):
    """Preprocess text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Remove mentions, hashtags
    if text.strip().lower() in ['null', 'you deleted this message', 'this message was deleted']:
        return ""
    text = re.sub(r'[^\w\s.?!]', '', text)  # Optional: remove emojis/special chars
    return text.strip()


