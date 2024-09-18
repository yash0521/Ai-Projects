import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessor
import helper

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    data = bytes_data.decode("utf-8")
    # st.text(data)

    df = preprocessor.proprocess(data)

    # st.dataframe(df)

    # Fetch Unique Users
    unique_users = df["user"].unique().tolist()
    unique_users.remove("group_notification")
    unique_users.sort()
    unique_users.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis of:", unique_users)

    if st.sidebar.button("Show Analysis"):

        # ''' Stats area for group and solo user '''

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(
            selected_user, df)

        st.title("Top Most Statstics ")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.info("Total Messages")
            st.write(num_messages)

        with col2:
            st.info("Total Words")
            st.write(words)

        with col3:
            st.info("Media Shared")
            st.write(num_media_messages)

        with col4:
            st.info("Links Shared")
            st.write(num_links)

        # ''' daily Timeline '''

        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)

        fig, ax = plt.subplots()
        ax.plot(daily_timeline["separate_date"], daily_timeline["message"])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ''' Monthly Timeline '''

        st.title("Monthly Timeline")
        monthly_timeline = helper.monthly_timeline(selected_user, df)

        fig, ax = plt.subplots()
        ax.plot(monthly_timeline["time"], monthly_timeline["message"])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ''' Activity mape'''
        st.title("Activity Name")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            st.header("Most busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="orange")
            plt.xticks(rotation=90)
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        activiy_heatmap = helper.activiy_heat_map(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(activiy_heatmap)
        st.pyplot(fig)

        # ''' Finding the busiest users area for group only '''

        if selected_user == "Overall":
            st.info("Busiest Users")
            col1, col2 = st.columns(2)

            most_busy_user, nuw_df = helper.most_busy_users(df)

            fig, ax = plt.subplots()

            with col1:
                ax.bar(most_busy_user.index, most_busy_user.values)
                plt.xticks(rotation=90)
                st.pyplot(fig)

            with col2:
                st.info("Percentage of messages sent")
                st.dataframe(nuw_df)

        # ''' WordCloud '''
        st.title("Word Cloud")
        df_wc = helper.create_wordCloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # ''' Most Common words '''
        st.title("Most Common Words")
        most_common_words = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_words[0], most_common_words[1])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ''' Most Common Emojis '''
        st.title("Most Common Emojis")
        most_common_emojis = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)
        with col1:
            st.info("Emojis")
            st.dataframe(most_common_emojis)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(most_common_emojis[1].head(),
                   labels=most_common_emojis[0].head(), autopct="%0.2f")
            st.pyplot(fig)
