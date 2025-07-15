import pandas as pd
import re
def preprocess(data):
    """this function preprocesses the WhatsApp chat data and returns a DataFrame."""

    # Regex pattern to match messages
    pattern_for_message = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[APMapm]{2}\s-\s'

    # Split into messages
    messages = re.split(pattern_for_message, data)[2:]

    # Extract dates
    messages = [item.strip() for item in messages]
    
    # Extract users and message content
    users = []
    cleaned_messages = []

    for msg in messages:
        parts = msg.split(': ', 1)
        if len(parts) == 2:
            users.append(parts[0])
            cleaned_messages.append(parts[1])
        else:
            # system message or deleted message etc.
            users.append('system')
            cleaned_messages.append(parts[0])
    

    pattern_time = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APMapm]{2})'
    data = data.replace('\u202f', ' ').replace('\xa0', ' ')
    timestamps = re.findall(pattern_time, data)[1:]

    # Create DataFrame
    df = pd.DataFrame({'user':users,'user_message': cleaned_messages,'message_date': timestamps})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day_name'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Add period column that shows data capture between which 24 hour format
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    
    return df;