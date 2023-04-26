from sklearn.feature_extraction.text import TfidfVectorizer
import training_data
import data_processor
#TF-IDF Vectorization

df = training_data.data()
df = data_processor.data_transform(df)

def tf_idf(df):
    #tfidf vectorization
    tfv = TfidfVectorizer(analyzer="word")
    tfv.fit(df['text_final'])
    return tfv

def fake_news_clfier(tfv, df):
    X = tfv.transform(df["text_final"])
    y = df['fake?']
    # splitting training and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    #Model Training
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train, y_train)
    #Model Evaluation on Testing Set
    from sklearn.metrics import f1_score, precision_score, recall_score
    # Accuracy: 0.948
    # Precision Score: 0.947
    # Recall Score: 0.953
    # F1 Score: 0.950
    return clf

def file_save():
    tfv_1 = tf_idf(df)
    clf = fake_news_clfier(tfv_1, df)
    import pickle
    # open a file, where you ant to store the data
    file1 = open('tf_id_model', 'wb')
    file2 = open('MNB_model', 'wb')

    # dump information to that file
    pickle.dump(tfv_1, file1)
    pickle.dump(clf, file2)
    # close the file
    file1.close()
    file2.close()
    return None

#run this command
file_save()