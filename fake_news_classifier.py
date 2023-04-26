def ML_model():
    import pickle
    # open relevant files
    f1 = open("MNB_model", "rb")
    f2 = open("tf_id_model", "rb")
    # storing pickle files as objects
    clf = pickle.load(f1)
    tf_id = pickle.load(f2)
    # Close the files
    f1.close()
    f2.close()
    return clf, tf_id

def classify(title, text):
    import numpy as np
    import pandas as pd
    clf, tf_id  = ML_model()
    dict = {"title": title, 
            "text": text}
    df_entry = pd.DataFrame(dict, index = [0])
    import data_processor
    df_to_tfid = data_processor.data_transform(df_entry)
    tf = tf_id.transform(df_to_tfid["text_final"])
    prediction = clf.predict(tf)
    return prediction

