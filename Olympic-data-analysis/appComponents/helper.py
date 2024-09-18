import numpy as np


def medal_telly(df):

    medal_telly = df.drop_duplicates(subset=['Team', "NOC", "Games", "Year",
                                             "City", "Sport", "Event", "Medal"])

    medal_telly = medal_telly.groupby("region").sum()[["Gold", "Silver", "Bronze"]].sort_values(
        by="Gold", ascending=False).reset_index()

    medal_telly["total"] = medal_telly["Bronze"] + \
        medal_telly["Silver"] + medal_telly["Gold"]

    return medal_telly


def county_year_list(df):
    year = df["Year"].unique().tolist()
    year.sort()
    year.insert(0, "Overall")

    county = np.unique(df["region"].dropna().values).tolist()
    county.sort()
    county.insert(0, "Overall")

    return year, county


def fetch_medal_tally(df, year, county):
    df = df.drop_duplicates(subset=['Team', "NOC", "Games", "Year",
                                    "City", "Sport", "Event", "Medal"])

    flag = 0
    if year == "Overall" and county == "Overall":
        temp_df = df
    elif year == "Overall" and county != "Overall":
        flag = 1
        temp_df = df[df["region"] == county]
    elif year != "Overall" and county == "Overall":
        temp_df = df[df["Year"] == int(year)]
    elif year != "Overall" and county != "Overall":
        temp_df = df[(df["Year"] == int(year)) & (df["region"] == county)]

    if flag == 1:
        temp_df = temp_df.groupby("Year").sum()[['Bronze', 'Silver', 'Gold']
                                                ].sort_values("Year", ascending=True).reset_index()
    else:
        temp_df = temp_df.groupby("region").sum()[['Bronze', 'Silver', 'Gold']
                                                  ].sort_values("Gold", ascending=False).reset_index()

    temp_df["total"] = temp_df["Bronze"] + \
        temp_df["Silver"] + temp_df["Gold"]

    return temp_df


def data_over_time(df, clm_name):
    nations_overTime = df.drop_duplicates(["Year", clm_name])[
        "Year"].value_counts().reset_index().sort_values("Year")

    return nations_overTime


def most_successfull_athelete(df, sport):
    df = df.dropna(subset=["Medal"])

    if sport != "Overall":
        df = df[df["Sport"] == sport]

    temp_df = df["Name"].value_counts().reset_index().merge(
        df, left_on="Name", right_on="Name", how="left")[["Name", "count", "Sport", "region"]].drop_duplicates().reset_index(drop=True).head(10)
    return temp_df


def yearwise_medal_tally(df, country):
    temp_df = df.dropna(subset=["Medal"]).drop_duplicates(subset=[
        "Team", "NOC", "Games", "Year", "City", "Sport", "Event", "Medal"])
    temp_df = temp_df[temp_df["region"] == country]
    temp_df = temp_df.groupby("Year").count()["Medal"].reset_index()

    return temp_df


def country_yearwise_medal_tally_overyear(df, county):
    temp_df = df.dropna(subset=["Medal"]).drop_duplicates(subset=[
        "Team", "NOC", "Games", "Year", "City", "Sport", "Event", "Medal"])
    temp_df = temp_df[temp_df["region"] == county]
    pivot_table = temp_df.pivot_table(
        index="Sport", columns="Year", aggfunc="count", values="Medal").fillna(0).astype(int)
    return pivot_table


def most_successfull_athelete_countrywise(df, county):
    df = df.dropna(subset=["Medal"])

    df = df[df["region"] == county]
    temp_df = df["Name"].value_counts().reset_index().merge(
        df, left_on="Name", right_on="Name", how="left")[["Name", "count", "Sport"]].drop_duplicates().head(10)
    return temp_df
