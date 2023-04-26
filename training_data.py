def data():
    # import necessary libraries
    import pandas as pd
    import numpy as np
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fake_news_path = os.path.join(current_dir,"Fake.csv")
    real_news_path = os.path.join(current_dir,"True.csv")
    
    #Importing relevant dataset
    df_fake = pd.read_csv(fake_news_path)
    df_real = pd.read_csv(real_news_path)

    #df_fake dataset modification
    df_fake["year"] = df_fake["date"].apply(lambda x: x[-4:])
    df_fake["fake?"] = 1

    #df_real dataset modification
    df_real["year"] = df_real["date"].apply(lambda x: x.strip()[-4:])
    df_real["fake?"] = 0

    # Combining and shuffling the datasets
    # no nulls in the data_set(tested in Jupyter Notebook)
    df = pd.concat([df_real, df_fake])
    # shuffling the dataset
    df = df.sample(frac = 1)
    df = df.reset_index()
    df = df.drop("index", axis=1)
    return df
