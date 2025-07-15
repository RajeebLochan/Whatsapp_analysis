
import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from preprcosseing import preprocess
from helper import (
    fetch_stats, most_activity, word_cloud, common_words, common_emojis,
    message_per_day_user, message_per_month_user, message_per_year_user,
    chat_duration, average_messages_per_day, activity_time_of_day,
    average_response_time, preprocess_text
)

import os
from dotenv import load_dotenv
from together import Together

# === Initialize Database ===
def init_db():
    conn = sqlite3.connect("uploaded_files.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(filename, content):
    conn = sqlite3.connect("uploaded_files.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files WHERE filename = ?", (filename,))
    exists = cursor.fetchone()[0]
    if exists == 0:
        cursor.execute("INSERT INTO files (filename, content) VALUES (?, ?)", (filename, content))
        conn.commit()
    conn.close()

init_db()

# === Streamlit Config ===
# Page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit branding and menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === Initialize Session State for button clicks ===
if 'show_analysis_clicked' not in st.session_state:
    st.session_state.show_analysis_clicked = False
if 'current_selected_user' not in st.session_state:
    st.session_state.current_selected_user = None
if 'current_df' not in st.session_state:
    st.session_state.current_df = None


# === Sidebar ===
st.title("📂 Upload Chat File")
uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

if uploaded_file:
    data = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")
    save_to_db(uploaded_file.name, data)

    df = preprocess(data)
    st.dataframe(df) # Keeping this commented as in your original code

    user_list = df['user'].unique().tolist()
    user_list = sorted([u for u in user_list if u != "system"])
    user_list.insert(0, "All Users")

    # Use a key for selectbox to ensure its state is managed properly across reruns
    selected_user = st.selectbox("Select User", user_list, key="user_selection_main")

    # Define a callback function for the "Show Analysis" button
    def show_analysis_callback():
        st.session_state.show_analysis_clicked = True
        st.session_state.current_selected_user = selected_user # Store selected user
        st.session_state.current_df = df # Store the dataframe

    # The "Show Analysis" button
    # If the selected_user changes, we should reset the analysis display
    if st.session_state.current_selected_user != selected_user:
        st.session_state.show_analysis_clicked = False

    st.button("Show Analysis", on_click=show_analysis_callback)
    
    # Conditionally display the analysis section
    if st.session_state.show_analysis_clicked:
        st.title("📊 WhatsApp Chat Analysis")

        # Use the stored selected_user and df from session_state for consistency
        current_selected_user = st.session_state.current_selected_user
        current_df = st.session_state.current_df

        # === Stats Cards ===
        stats = fetch_stats(current_selected_user, current_df)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Messages", stats[0])
        col2.metric("Total Words", stats[1])
        col3.metric("Media Messages", stats[2])
        col4.metric("Links Shared", stats[3])
        # Ensure division by zero is handled if df['date'].nunique() is 0
        avg_msg_per_day = f"{current_df['user_message'].count() / current_df['date'].nunique():.2f}" if current_df['date'].nunique() > 0 else "N/A"
        col5.metric("Avg Messages/Day", avg_msg_per_day)
        col6.metric("Chat Duration", chat_duration(current_selected_user, current_df))

        # === Most Active Users ===
        st.subheader("🔥 Most Active Users")
        act = most_activity(current_df)
        col1, col2 = st.columns(2)
        col1.bar_chart(act.set_index("user")["messages"].head(5))
        col2.dataframe(act.head(5), use_container_width=True)

        # === Response Time ===
        st.subheader("⏳ Response Time")
        avg_resp = average_response_time(current_selected_user, current_df)
        st.info(f"Average response time: **{avg_resp:.2f} minutes**")

        # === Word Cloud ===
        st.subheader("☁️ Word Cloud")
        wc = word_cloud(current_selected_user, current_df)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # === Common Words/Emojis ===
        st.subheader("⏰ Activity Time of Day")
        st.line_chart(activity_time_of_day(current_selected_user, df).set_index("time")["messages"])

        st.subheader("🔠 Common Words")
        st.dataframe(common_words(current_selected_user, current_df), use_container_width=True)

        st.subheader("😊 Common Emojis")
        st.dataframe(common_emojis(current_selected_user, df), use_container_width=True)

        # === Trends ===
        st.subheader("📈 Daily Messages")
        st.line_chart(message_per_day_user(current_selected_user, df).set_index("date")["messages"])

        st.subheader("📆 Monthly Messages")
        st.area_chart(message_per_month_user(current_selected_user, df).set_index("month")["messages"])

        st.subheader("📊 Yearly Messages")
        st.bar_chart(message_per_year_user(current_selected_user, df).set_index("year")["messages"])

      
        # text level analysis with ML Models 
        st.subheader("🔍 Text Level Analysis")
        #remove the data where user is system
        df = df[df['user'] != 'system'].reset_index(drop=True)

        # Apply cleaning
        df['cleaned_message'] = df['user_message'].apply(preprocess_text)

        # -------------------------
        # 2. Load Sentiment Model
        # -------------------------
        @st.cache_resource
        def load_sentiment_model():
            model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        try:
            sentiment_task = load_sentiment_model()
        except Exception as e:
            st.error(f"❌ Failed to load sentiment model: {e}")
            st.stop()

        # -------------------------
        # Run Sentiment Analysis with Progress
        # -------------------------
        st.info("Analyzing sentiment... This may take a few seconds.")
        results = []
        progress_bar = st.progress(0)

        for i, message in enumerate(df['cleaned_message']):
            if message.strip():
                try:
                    sentiment = sentiment_task(message)[0]
                except:
                    sentiment = {'label': 'Error', 'score': None}
            else:
                sentiment = {'label': 'No Text', 'score': None}
            results.append(sentiment)
            progress_bar.progress((i + 1) / len(df))

        # -------------------------
        # Post-processing
        # -------------------------
        df_results = pd.DataFrame(results)
        df['sentiment_label'] = df_results['label']
        df['sentiment_score'] = df_results['score']

        # Filter out "No Text" rows
        df = df[df['sentiment_label'] != 'No Text']
        

        df['sentiment_score'] = df['sentiment_score'].astype(float).round(2)

        # -------------------------
        # Show Results
        # -------------------------
        st.subheader("💬 Sentiment Analysis Result")
        st.dataframe(df[['user', 'user_message', 'cleaned_message', 'sentiment_label', 'sentiment_score']], use_container_width=True)
        
        #remove error from sentiment_label
        df = df[df['sentiment_label'] != 'Error'].reset_index(drop=True)

        # Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts()
        st.bar_chart(sentiment_counts)

        #user wise sentimate analysis shows how many messages each user has sent with different sentiments by sentimate label
        st.subheader("👥 User-wise Sentiment Analysis")
        user_sentiment_counts = df.groupby(['user', 'sentiment_label']).size().unstack(fill_value=0)
        st.dataframe(user_sentiment_counts, use_container_width=True)
        st.bar_chart(user_sentiment_counts)
        # Sentiment Score Distribution
        st.subheader("📈 Sentiment Score Distribution")
        st.line_chart(df.groupby('date')['sentiment_score'].mean())
        st.info("Sentiment scores range from 0 (negative) to 1 (positive).")
        st.success("Sentiment analysis completed successfully!")

        # Apology frequency / gratitude frequency (“sorry”, “thanks”, “please”) based on user and sentiment and user_message
        st.subheader("🙏 Apology and Gratitude Frequency")
        apology_words = ["sorry", "apologies", "apologize"]
        gratitude_words = ["thanks", "thank you", "please"]

        # Count apologies in negative sentiment messages
        apology_counts = df[
            (df['sentiment_label'] == 'negative') &
            (df['user_message'].str.contains('|'.join(apology_words), case=False, na=False))
        ].shape[0]

        # Count gratitude in positive sentiment messages
        gratitude_counts = df[
            (df['sentiment_label'] == 'positive') &
            (df['user_message'].str.contains('|'.join(gratitude_words), case=False, na=False))
        ].shape[0]

        st.write(f"Apology Frequency: {apology_counts}")
        st.write(f"Gratitude Frequency: {gratitude_counts}")

        # Count apology and gratitude words by user (regardless of sentiment)
        apology_by_user = df[
            df['user_message'].str.contains('|'.join(apology_words), case=False, na=False)
        ].groupby('user').size().reset_index(name='apology_count')

        gratitude_by_user = df[
            df['user_message'].str.contains('|'.join(gratitude_words), case=False, na=False)
        ].groupby('user').size().reset_index(name='gratitude_count')

        st.subheader("👥 Apology and Gratitude by User")
        st.dataframe(apology_by_user, use_container_width=True)
        #draw bar chart for apology by user
        st.bar_chart(apology_by_user.set_index('user')['apology_count'])
        st.dataframe(gratitude_by_user, use_container_width=True)
        #draw bar chart for gratitude by user
        st.bar_chart(gratitude_by_user.set_index('user')['gratitude_count'])
        st.success("Apology and gratitude frequency analysis completed successfully!")

        # 3. Conversation Initiation Who starts most chats? Visualize who puts more effort into continuing the relationship
        st.subheader("💬 Conversation Initiation Analysis")
        # Count the number of messages sent by each user
        conversation_initiation = df.groupby('user')['user_message'].count().reset_index()
        conversation_initiation.columns = ['user', 'message_count']
        # Sort by message count
        conversation_initiation = conversation_initiation.sort_values(by='message_count', ascending=False)
        st.dataframe(conversation_initiation, use_container_width=True)
        # Draw bar chart for conversation initiation
        st.bar_chart(conversation_initiation.set_index('user')['message_count'])
        st.success("Conversation initiation analysis completed successfully!")
        # 4. User Engagement Over Time
        st.subheader("📅 User Engagement Over Time")
        # Count messages per user per day
        engagement_over_time = df.groupby(['date', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_over_time, use_container_width=True)
        # Draw line chart for user engagement over time
        st.line_chart(engagement_over_time)
        st.success("User engagement over time analysis completed successfully!")
        # 5. User Sentiment Over Time
        st.subheader("📈 User Sentiment Over Time")
        # Calculate average sentiment score per user per day
        sentiment_over_time = df.groupby(['date', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_over_time, use_container_width=True)
        # Draw line chart for user sentiment over time
        st.line_chart(sentiment_over_time)
        st.success("User sentiment over time analysis completed successfully!")
        # 6. User Activity Heatmap
        st.subheader("🌡️ User Activity Heatmap")
        # Create a pivot table for user activity
        activity_heatmap = df.pivot_table(index='date', columns='user', values='user_message', aggfunc='count', fill_value=0)
        st.dataframe(activity_heatmap, use_container_width=True)
        # Draw heatmap for user activity
        st.line_chart(activity_heatmap)
        st.success("User activity heatmap analysis completed successfully!")
        # 7. User Sentiment Heatmap
        st.subheader("🌡️ User Sentiment Heatmap")
        # Create a pivot table for user sentiment
        sentiment_heatmap = df.pivot_table(index='date', columns='user', values='sentiment_score', aggfunc='mean', fill_value=0)
        st.dataframe(sentiment_heatmap, use_container_width=True)
        # Draw heatmap for user sentiment
        st.line_chart(sentiment_heatmap)
        st.success("User sentiment heatmap analysis completed successfully!")
        # 8. User Engagement by Time of Day
        st.subheader("🕒 User Engagement by Time of Day")
        # Extract hour from date
        df['hour'] = df['date'].dt.hour
        # Count messages per user per hour
        engagement_by_hour = df.groupby(['hour', 'user']).size().unstack(fill_value=0) 
        st.dataframe(engagement_by_hour, use_container_width=True)
        # Draw line chart for user engagement by hour   
        st.line_chart(engagement_by_hour)
        st.bar_chart(engagement_by_hour)    
        st.success("User engagement by time of day analysis completed successfully!")    
        # 9. User Sentiment by Time of Day      
        st.subheader("🕒 User Sentiment by Time of Day")   
        # Calculate average sentiment score per user per hour
        sentiment_by_hour = df.groupby(['hour', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_hour, use_container_width=True)
        # Draw line chart for user sentiment by hour
        st.line_chart(sentiment_by_hour)
        st.bar_chart(sentiment_by_hour)
        st.success("User sentiment by time of day analysis completed successfully!")
        # 10. User Engagement by Day of Week
        st.subheader("📅 User Engagement by Day of Week")
        # Extract day of week from date
        df['day_of_week'] = df['date'].dt.day_name()
        # Count messages per user per day of week
        engagement_by_day = df.groupby(['day_of_week', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_day, use_container_width=True)   
        # Draw line chart for user engagement by day of week
        st.line_chart(engagement_by_day)
        st.bar_chart(engagement_by_day)
        st.success("User engagement by day of week analysis completed successfully!")
        # 11. User Sentiment by Day of Week
        st.subheader("📅 User Sentiment by Day of Week")
        # Calculate average sentiment score per user per day of week
        sentiment_by_day = df.groupby(['day_of_week', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_day, use_container_width=True)
        # Draw line chart for user sentiment by day of week
        st.line_chart(sentiment_by_day)
        st.bar_chart(sentiment_by_day)
        st.success("User sentiment by day of week analysis completed successfully!")
        # 12. User Engagement by Month
        st.subheader("📅 User Engagement by Month")
        # Extract month from date
        df['month'] = df['date'].dt.month_name()
        # Count messages per user per month
        engagement_by_month = df.groupby(['month', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_month, use_container_width=True)
        # Draw line chart for user engagement by month
        st.line_chart(engagement_by_month)
        st.bar_chart(engagement_by_month)
        st.success("User engagement by month analysis completed successfully!")
        # 13. User Sentiment by Month
        st.subheader("📅 User Sentiment by Month")
        # Calculate average sentiment score per user per month
        sentiment_by_month = df.groupby(['month', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_month, use_container_width=True)
        # Draw line chart for user sentiment by month
        st.line_chart(sentiment_by_month)
        st.bar_chart(sentiment_by_month)    
        st.success("User sentiment by month analysis completed successfully!")
        # 14. User Engagement by Year
        st.subheader("📅 User Engagement by Year")
        # Extract year from date
        df['year'] = df['date'].dt.year 
        # Count messages per user per year
        engagement_by_year = df.groupby(['year', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_year, use_container_width=True)  
        # Draw line chart for user engagement by year
        st.line_chart(engagement_by_year)
        st.bar_chart(engagement_by_year)
        st.success("User engagement by year analysis completed successfully!")
        # 15. User Sentiment by Year
        st.subheader("📅 User Sentiment by Year")
        # Calculate average sentiment score per user per year
        sentiment_by_year = df.groupby(['year', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_year, use_container_width=True)
        # Draw line chart for user sentiment by year
        st.line_chart(sentiment_by_year)
        st.bar_chart(sentiment_by_year)
        st.success("User sentiment by year analysis completed successfully!")
        # 16. User Engagement by Hour of Day
        st.subheader("🕒 User Engagement by Hour of Day")
        # Extract hour of day from date
        df['hour_of_day'] = df['date'].dt.hour
        # Count messages per user per hour of day
        engagement_by_hour_of_day = df.groupby(['hour_of_day', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_hour_of_day, use_container_width=True)
        # Draw line chart for user engagement by hour of day
        st.line_chart(engagement_by_hour_of_day)
        st.bar_chart(engagement_by_hour_of_day)
        st.success("User engagement by hour of day analysis completed successfully!")   
        # 17. User Sentiment by Hour of Day
        st.subheader("🕒 User Sentiment by Hour of Day") 
        # Calculate average sentiment score per user per hour of day   
        sentiment_by_hour_of_day = df.groupby(['hour_of_day', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_hour_of_day, use_container_width=True)
        # Draw line chart for user sentiment by hour of day
        st.line_chart(sentiment_by_hour_of_day)
        st.bar_chart(sentiment_by_hour_of_day)
        st.success("User sentiment by hour of day analysis completed successfully!")
        # 18. User Engagement by Day of Month
        st.subheader("📅 User Engagement by Day of Month")
        # Extract day of month from date
        df['day_of_month'] = df['date'].dt.day
        # Count messages per user per day of month
        engagement_by_day_of_month = df.groupby(['day_of_month', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_day_of_month, use_container_width=True)
        # Draw line chart for user engagement by day of month
        st.line_chart(engagement_by_day_of_month)
        st.bar_chart(engagement_by_day_of_month)
        st.success("User engagement by day of month analysis completed successfully!")
        # 19. User Sentiment by Day of Month
        st.subheader("📅 User Sentiment by Day of Month")
        # Calculate average sentiment score per user per day of month
        sentiment_by_day_of_month = df.groupby(['day_of_month', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_day_of_month, use_container_width=True)
        # Draw line chart for user sentiment by day of month
        st.line_chart(sentiment_by_day_of_month)
        st.bar_chart(sentiment_by_day_of_month)
        st.success("User sentiment by day of month analysis completed successfully!")
        # 20. User Engagement by Week of Year
        st.subheader("📅 User Engagement by Week of Year")
        # Extract week of year from date
        df['week_of_year'] = df['date'].dt.isocalendar().week
        # Count messages per user per week of year
        engagement_by_week_of_year = df.groupby(['week_of_year', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_week_of_year, use_container_width=True)
        # Draw line chart for user engagement by week of year
        st.line_chart(engagement_by_week_of_year)
        st.bar_chart(engagement_by_week_of_year)
        st.success("User engagement by week of year analysis completed successfully!")
        # 21. User Sentiment by Week of Year
        st.subheader("📅 User Sentiment by Week of Year")
        # Calculate average sentiment score per user per week of year
        sentiment_by_week_of_year = df.groupby(['week_of_year', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_week_of_year, use_container_width=True)
        # Draw line chart for user sentiment by week of year
        st.line_chart(sentiment_by_week_of_year)
        st.bar_chart(sentiment_by_week_of_year)
        st.success("User sentiment by week of year analysis completed successfully!")
        # 22. User Engagement by Quarter
        st.subheader("📅 User Engagement by Quarter")
        # Extract quarter from date
        df['quarter'] = df['date'].dt.quarter
        # Count messages per user per quarter
        engagement_by_quarter = df.groupby(['quarter', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_quarter, use_container_width=True)
        # Draw line chart for user engagement by quarter
        st.line_chart(engagement_by_quarter)
        st.bar_chart(engagement_by_quarter)
        st.success("User engagement by quarter analysis completed successfully!")
        # 23. User Sentiment by Quarter
        st.subheader("📅 User Sentiment by Quarter")
        # Calculate average sentiment score per user per quarter
        sentiment_by_quarter = df.groupby(['quarter', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_quarter, use_container_width=True)
        # Draw line chart for user sentiment by quarter
        st.line_chart(sentiment_by_quarter)
        st.bar_chart(sentiment_by_quarter)
        st.success("User sentiment by quarter analysis completed successfully!")
        # 24. User Engagement by Season
        st.subheader("🌸 User Engagement by Season")    
        # Define seasons based on month
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        # Apply the get_season function to create a new column
        df['season'] = df['date'].dt.month.apply(get_season)    
        # Count messages per user per season
        engagement_by_season = df.groupby(['season', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_season, use_container_width=True)    
        # Draw line chart for user engagement by season
        st.line_chart(engagement_by_season)
        st.bar_chart(engagement_by_season)
        st.success("User engagement by season analysis completed successfully!")
        # 25. User Sentiment by Season
        st.subheader("🌸 User Sentiment by Season")
        # Calculate average sentiment score per user per season
        sentiment_by_season = df.groupby(['season', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_season, use_container_width=True)
        # Draw line chart for user sentiment by season
        st.line_chart(sentiment_by_season)
        st.bar_chart(sentiment_by_season)
        st.success("User sentiment by season analysis completed successfully!")
        #Emotional Imbalance Meter More "sorry", "please", or emotional lines from one side? Plot imbalance: who gave more, who stayed silent more
        st.subheader("⚖️ Emotional Imbalance Meter")
        # Count emotional words in messages
        emotional_words = ["sorry", "please", "thank you", "love", "hate"]
        df['emotional_count'] = df['user_message'].apply(lambda x: sum(word in x.lower() for word in emotional_words))
        emotional_imbalance = df.groupby('user')['emotional_count'].sum().reset_index()
        emotional_imbalance = emotional_imbalance.sort_values(by='emotional_count', ascending=False)
        st.dataframe(emotional_imbalance, use_container_width=True)
        # Draw bar chart for emotional imbalance
        st.bar_chart(emotional_imbalance.set_index('user')['emotional_count'])
        st.success("Emotional imbalance analysis completed successfully!")
        # 26. User Engagement by Emotional Imbalance
        st.subheader("⚖️ User Engagement by Emotional Imbalance")
        # Count messages per user with emotional imbalance
        engagement_by_emotional_imbalance = df.groupby(['user', 'emotional_count']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_emotional_imbalance, use_container_width=True)
        # Draw line chart for user engagement by emotional imbalance
        st.line_chart(engagement_by_emotional_imbalance)
        st.bar_chart(engagement_by_emotional_imbalance)
        st.success("User engagement by emotional imbalance analysis completed successfully!")
        #Word Cloud + Topic Segmentation Top used terms per person Cluster messages into themes: study, love, anger, confusion
        st.subheader("☁️ Word Cloud + Topic Segmentation")
        # Generate word cloud for each user
        from wordcloud import WordCloud
        def generate_wordcloud(text):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            return wordcloud
        user_wordclouds = {}
        for user in df['user'].unique():
            user_text = ' '.join(df[df['user'] == user]['user_message'])
            user_wordclouds[user] = generate_wordcloud(user_text)
        # Display word clouds for each user
        for user, wordcloud in user_wordclouds.items():
            st.subheader(f"☁️ Word Cloud for {user}")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        st.success("Word cloud and topic segmentation analysis completed successfully!")

        # 27. User Engagement by Topic Segmentation
        st.subheader("📊 User Engagement by Topic Segmentation")
        # Count messages per user per topic
        topic_words = ["study", "love", "anger", "confusion"]
        df['topic'] = df['user_message'].apply(lambda x: next((word for word in topic_words if word in x.lower()), 'Other'))
        engagement_by_topic = df.groupby(['topic', 'user']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_topic, use_container_width=True)
        # Draw line chart for user engagement by topic
        st.line_chart(engagement_by_topic)
        st.bar_chart(engagement_by_topic)
        st.success("User engagement by topic segmentation analysis completed successfully!")
        # 28. User Sentiment by Topic Segmentation
        st.subheader("📊 User Sentiment by Topic Segmentation")
        # Calculate average sentiment score per user per topic
        sentiment_by_topic = df.groupby(['topic', 'user'])['sentiment_score'].mean().unstack(fill_value=0)
        st.dataframe(sentiment_by_topic, use_container_width=True)
        # Draw line chart for user sentiment by topic
        st.line_chart(sentiment_by_topic)
        st.bar_chart(sentiment_by_topic)
        st.success("User sentiment by topic segmentation analysis completed successfully!")
        # 29. User Engagement by Emotional Imbalance and Topic Segmentation
        st.subheader("⚖️ User Engagement by Emotional Imbalance and Topic Segmentation")
        # Count messages per user with emotional imbalance and topic segmentation
        # Bin emotional_count to reduce number of columns
        bins = [0, 1, 2, 3, 5, 10, float('inf')]
        labels = ['0', '1', '2', '3-4', '5-9', '10+']
        df['emotional_count_binned'] = pd.cut(df['emotional_count'], bins=bins, labels=labels, right=False)
        engagement_by_emotional_imbalance_topic = df.groupby(['user', 'topic', 'emotional_count_binned']).size().unstack(fill_value=0)
        st.dataframe(engagement_by_emotional_imbalance_topic, use_container_width=True)
        # Draw line chart for user engagement by emotional imbalance and topic segmentation
        # st.line_chart(engagement_by_emotional_imbalance_topic)
        # st.bar_chart(engagement_by_emotional_imbalance_topic)
        st.success("User engagement by emotional imbalance and topic segmentation analysis completed successfully!")
        # 30. Conversation Length Analysis
        import re
        df['message_length'] = df['user_message'].apply(lambda x: len(re.findall(r'\b\w+\b', x)))
        #Conversation length: avg length per session
        st.subheader("📏 Conversation Length Analysis")
        # Calculate average message length per user
        df['message_length'] = df['user_message'].apply(lambda x: len(x.split()))
        avg_length_per_session = df.groupby('user')['message_length'].mean().reset_index()  
        avg_length_per_session.columns = ['user', 'avg_message_length']
        st.dataframe(avg_length_per_session, use_container_width=True)
        # Draw bar chart for average message length per user
        st.bar_chart(avg_length_per_session.set_index('user')['avg_message_length'])
        st.success("Average message length per session analysis completed successfully!")

        # Silence detection: longest gaps in chat, and who breaks the silence
        st.subheader("🔕 Silence Detection Analysis")

        # Calculate time differences between messages
        df['time_diff'] = df['date'].diff().dt.total_seconds() / 60  # Convert to minutes

        # Find all silence gaps longer than 60 minutes
        silence_gaps = df[df['time_diff'] > 60].copy()
        silence_gaps['prev_user'] = df['user'].shift(1)
        silence_gaps['gap_start_time'] = df['date'].shift(1)
        silence_gaps['gap_end_time'] = df['date']

        # Who breaks the silence most often?
        breaker_counts = silence_gaps['user'].value_counts().reset_index()
        breaker_counts.columns = ['user', 'times_broke_silence']

        # Longest silence per user
        max_silence_gaps = silence_gaps.groupby('user')['time_diff'].max().reset_index()
        max_silence_gaps.columns = ['user', 'max_silence_gap']

        # Combine insights
        silence_insight = pd.merge(max_silence_gaps, breaker_counts, on='user', how='outer').fillna(0)
        silence_insight = silence_insight.sort_values(by='max_silence_gap', ascending=False)

        st.markdown("**Longest silence gaps (minutes) and who broke the silence most often:**")
        st.dataframe(silence_insight, use_container_width=True)

        import plotly.express as px

        fig1 = px.bar(
            silence_insight,
            x='user',
            y='max_silence_gap',
            color='times_broke_silence',
            title="Longest Silence Gap per User (minutes)",
            labels={'max_silence_gap': 'Max Silence Gap (min)', 'user': 'User'},
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            breaker_counts,
            x='user',
            y='times_broke_silence',
            title="Who Broke the Silence Most Often",
            labels={'times_broke_silence': 'Times Broke Silence', 'user': 'User'},
            color='times_broke_silence',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Show top 5 longest silence gaps with context
        top_gaps = silence_gaps.nlargest(5, 'time_diff')[['gap_start_time', 'gap_end_time', 'prev_user', 'user', 'time_diff']]
        top_gaps.columns = ['Silence Start', 'Silence End', 'Who Was Silent', 'Who Broke Silence', 'Gap (min)']
        st.markdown("**Top 5 longest silences and who broke them:**")
        st.dataframe(top_gaps, use_container_width=True)

        st.info(
            "Silence detection reveals not only who tends to leave the longest gaps, but also who most frequently resumes the conversation. "
            "This can indicate engagement patterns and relationship dynamics."
        )
        st.success("Silence detection analysis completed successfully!")


        #makeing summerzation of whatsapp chat using generative ai like llama4 using together api its create other person chat summerize
        st.subheader("🤖 WhatsApp Chat Summarization")
        st.info(
            "This feature uses generative AI to summarize the WhatsApp chat, providing a concise overview of the conversation."
        )
        # Placeholder for chat summarization logic
        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY")

        if not api_key:
            raise ValueError("Missing TOGETHER_API_KEY in environment or .env file")

        client = Together(api_key=api_key)
        
                
        prompt = f"""
        You are a character profiler. Based on the following WhatsApp messages from a person named unknown, describe their personality traits, communication style, emotions, values, and any unique behaviors or expressions. This profile should be suitable for an actor preparing to play this person.write a sort story abouth this person according to message you read one sided text
        Messages:
        {data} write big context and make a parsona of the person based on the chat messages. Include details about their personality, interests, and any notable characteristics that can help in understanding who they are.in data first chat is from the person you are making a character about, so make sure to include that in the summary.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes WhatsApp chats make buid Charecter about chat sat its which type of person is write and guess who is the person."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=messages,
            stream=True
        )

        # Display the AI-generated summary in Streamlit, streaming as tokens arrive
        summary_text = ""
        summary_placeholder = st.empty()
        for token in response:
            if hasattr(token, 'choices') and token.choices and hasattr(token.choices[0], 'delta'):
                content = token.choices[0].delta.content
                if content:
                    summary_text += content
                    summary_placeholder.markdown(f"**Summary:**\n\n{summary_text}")
        st.success("Summary complete.")




        



        
        



        
         
        


    