import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


plt.style.use(style="seaborn")

data = pd.read_csv("SanjuDon.csv")


train = data

train = train.dropna().reset_index(drop=True)
print(len(train['text']))

print(train.head())


def lower_text(text):
    return text.lower()

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


train["text"] = train.text.map(lambda x: lower_text(x))
train["text"] = train.text.map(lambda x: remove_URL(x))
train["text"] = train.text.map(lambda x: remove_html(x))
train["text"] = train.text.map(lambda x: remove_emoji(x))
train["text"] = train.text.map(lambda x: remove_punct(x))

stop = set(stopwords.words("english"))
ps = PorterStemmer()

# def remove_stopwords(text):
#     text = [word.lower() for word in text.split() if word.lower() not in stop]
#
#     return " ".join(text)
#
#
# train["text"] = train["text"].map(remove_stopwords)


# def stopwords_removal_and_stemming(text):
#     text = [ps.stem(word) for word in text if word not in stop]
#     return " ".join(text)
#
#
# train["text"] = train["text"].map(stopwords_removal_and_stemming)

train['text'][194365] = ' '.join(train['text'][194365].split()[:-1])

for i in range(len(train['text'])):
    if i % 2000 == 0:
        print(i*100/len(train['text']))

    train['text'][i] = ' '.join([ps.stem(word) for word in train['text'][i].split() if word not in stop])




print(train.text)



def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count


text = train.text

counter = counter_word(text)

print(len(counter))

print(counter)

num_words = len(counter)

# Max number of words in a sequence
max_length = 20

train_size = int(train.shape[0] * 0.8)

train_sentences = train.text[:train_size]
train_labels = train.is_offensive[:train_size]

test_sentences = train.text[train_size:]
test_labels = train.is_offensive[train_size:]


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

print(word_index)

train_sequences = tokenizer.texts_to_sequences(train_sentences)


train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post"
)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding="post", truncating="post"
)

print(train.text[0])
print(train_sequences[0])

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode(train_sequences[0]))

print(f"Shape of train {train_padded.shape}")
print(f"Shape of test {test_padded.shape}")


model = Sequential()
model.add(LSTM(64, dropout=0.1))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model_with_embedding = Sequential()
model_with_embedding.add(Embedding(len(word_index), 32, input_length=max_length))
model_with_embedding.add(model)

model_with_embedding.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model_with_embedding.fit(
    train_padded, train_labels, epochs=6, batch_size=10000, validation_data=(test_padded, test_labels),
)

print("Without: ")
model.summary()
print("With: ")
model_with_embedding.summary()
model_with_embedding.save('Models/FourthLSTM.h5')
model.save('Models/FourthLSTM_no_embedding_model.h5')

#plotting results
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

tflite_model = tf.keras.models.load_model('Models/FourthLSTM_no_embedding_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
converter.post_training_quantize = True
tflite_buffer = converter.convert()
open('Models/FourthLSTM.tflite', 'wb').write(tflite_buffer)

embedding_matrix = model_with_embedding.layers[0].get_weights()[0]  # --- ( 1 )
print('Embedding Shape ~> {}'.format(embedding_matrix.shape))


word_index_2 = dict()
for word, index in word_index.items():
    word_index_2[index] = word
word_index = word_index_2
embedding_dict = dict()


for i in range(len(embedding_matrix) - 1):
    embedding_dict[word_index[i + 1]] = embedding_matrix[i + 1].tolist()


with open('Models/FourthLSTM_embedding.json', 'w') as file:
    json.dump(embedding_dict, file)

with open('Models/FourthLSTM_word_index.json', 'w') as file:
    json.dump(word_index, file)

with open('Models/FourthLSTM_tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)







