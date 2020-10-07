import pandas as pd
import string
from bayesian_classifier import BayesianClassifier


def get_stop_word(file_name: str):
    """
    str -> list

    :param file_name: file to read the words from
    :return: stop words list
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        words = f.read()
        words = words.split('\n')
    return words


def count_words(sentence):
    """
    str -> dict

    :param sentence: sentence to count words of
    :return: dictionary with a word as a key and count as
    a value
    """
    counts = {}
    sentence = sentence.split(' ')
    for word in sentence:
        counts[word] = sentence.count(word)
    return counts


def edit_tweet(tweet: str):
    """
    str -> str

    :param tweet: tweet to edit
    :return: edited tweet according to conditions
    """

    # getting list of stop words
    stop_words = get_stop_word('stop_words.txt')
    stop_words.append('user')

    # clearing from the punctuation
    for symbol in string.punctuation:
        tweet = tweet.replace(symbol, '')

    # clearing from the wrong encoding
    utf8_letters = string.ascii_letters + ' '

    for char in tweet:
        if char not in utf8_letters:
            tweet = tweet.replace(char, '')

    # clearing from the stop words
    tweet = tweet.split(' ')
    tweet = [word for word in tweet if word not in stop_words]

    return " ".join(tweet)


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """

    # reading file
    data = pd.read_csv(data_file, encoding='UTF-8')
    df = pd.DataFrame(data=data)

    # dropping redundant columns
    labels_df = df['label']

    df = df.drop(columns=['id', 'label', 'Unnamed: 0'])

    bag_of_words = []

    # clearing data
    for index in df.index:
        df.loc[index, 'tweet'] = edit_tweet(df.loc[index, 'tweet'])
        this = df.loc[index, 'tweet']
        bag_of_words.append(count_words(this))

    df['Count'] = bag_of_words

    # writing result into files
    labels_df.to_csv('labels.csv', encoding='utf-8')
    df.to_csv('result.csv', encoding='utf-8')

    return labels_df, df


if __name__ == "__main__":
    """
    train_X, train_y = process_data("your train data file")
    test_X, test_y = process_data("your test data file")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    classifier.predict_prob(test_X[0], test_y[0])

    print("model score: ", classifier.score(test_X, test_y))
    """