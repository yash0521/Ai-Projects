import pandas as pd


def preprocess(df, region_df):

    # Filtering only the Summer Olympics
    df = df[df['Season'] == 'Summer']

    # Mearging the dataframe with region_df
    df = df.merge(region_df, how='left', on='NOC')

    # Dropping duplicate values
    df.drop_duplicates(inplace=True)

    # One hot encoding on mdeal column
    df = pd.concat([df, pd.get_dummies(df['Medal']).astype(int)], axis=1)

    return df
