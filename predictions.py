import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences


def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords_and_stem(text):
    global ps
    text = [ps.stem(word.lower()) for word in text.split() if word.lower() not in set(stopwords.words("english"))]
    return " ".join(text)



def load_assets():
    global model, tokenizer, ps
    ps = PorterStemmer()
    model = keras.models.load_model('Models/FirstLSTM.h5')
    tokenizer = pickle.load(open('Models/FirstLSTM_tokenizer.pkl', 'rb'))


max_length = 20
load_assets()

text = input('Enter your input: ')
while text != 'stop':
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_punct(text)
    text = remove_stopwords_and_stem(text)

    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=max_length, padding="post", truncating="post")
    #print(text)
    result = model.predict(text)
    if result[0][0] >= 0.5:
        print('Prediction: Offensive')
    else:
        print('Prediction: Non-Offensive')

    text = input('Enter your input: ')
