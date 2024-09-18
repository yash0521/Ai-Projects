from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji


def fetch_stats(selected_user, df):

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    # Number of message
    num_messages = df.shape[0]

    # Number of words
    words = []
    for message in df["message"]:
        words.extend(message.split())

    # Number of media omitetd
    num_media_messages = df[df["message"] == "<Media omitted>\n"].shape[0]

    # Number of links
    links = []
    extractor = URLExtract()

    for message in df["message"]:
        links.extend(extractor.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    # Number of messages sent by users
    most_busy_user = df["user"].value_counts().head(5)

    # Percentage of chat
    df = round((df["user"].value_counts() / df.shape[0])*100,
               2).reset_index().rename(columns={"index": "name", "user": "percentage"})
    return most_busy_user, df


def create_wordCloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # Remvoing Group notification
    temp = df[df["user"] != "group_notification"]

    # Removing Media omitted
    temp = temp[temp["message"] != "<Media omitted>\n"]

    # stopWords_file = open("./stop_hinglish.txt", "r")
    # stop_words = stopWords_file.read()

    wc = WordCloud(width=500, height=500, max_font_size=30,
                   background_color="white")
    df_wc = wc.generate(temp["message"].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remvoing Group notification
    temp = df[df["user"] != "group_notification"]

    # Removing Media omitted
    temp = temp[temp["message"] != "<Media omitted>\n"]

    # Removing stop words
    # Here chat could be in hindi, so we can't use default stop words removal from nltk
    # We create a file of common Hinglish words and use it

    stopWords_file = open("./stop_hinglish.txt", "r")
    stop_words = stopWords_file.read()

    words = []

    for message in temp["message"]:
        for word in message.lower().split():
            if (word not in stop_words):
                words.append(word)

    commonWords_df = pd.DataFrame(Counter(words).most_common(20))

    return commonWords_df


def emoji_helper(selected_user, df):
    emojis = []
    for message in df["message"]:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()[
        "message"].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline["month"][i] + "-" + str(timeline["year"][i]))
    timeline["time"] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby("separate_date").count()[
        "message"].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df["month"].value_counts()


def activiy_heat_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    activiy_heatmap = df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count").fillna(0)

    return activiy_heatmap
