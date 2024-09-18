import pandas as pd
import re


def proprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    message = re.split(pattern, data, flags=re.I)[1:]

    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': message, 'message_date': dates})

    df['message_date'] = pd.to_datetime(
        df['message_date'], format='%d/%m/%Y, %H:%M - ')

    df.rename(columns={"message_date": "date"}, inplace=True)

    user = []
    messages = []
    for message in df["user_message"]:
        entry = re.split("([\w\W]+?):\s", message)
        if entry[1:]:
            user.append(entry[1])
            messages.append(entry[2])
        else:
            user.append("group_notification")
            messages.append(entry[0])

    df["user"] = user
    df["message"] = messages
    df.drop(columns=["user_message"], inplace=True)

    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["day_name"] = df["date"].dt.day_name()
    df["separate_date"] = df["date"].dt.date
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    period = []
    for hour in df[["day_name", "hour"]]["hour"]:
        if hour == 23:
            period.append(str(hour)+"-"+str("00"))
        elif hour == 0:
            period.append(str("00")+"-"+str(hour+1))
        else:
            period.append(str(hour)+"-"+str(hour+1))

    df["period"] = period

    return df
