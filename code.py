import json
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt


with open('data/content.json') as content:
    data = json.load(content)

tags = []
inputs = []
responses={}
for intent in data['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])
data = pd.DataFrame({"inputs":inputs,"tags":tags})
print(data)


import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

#tokenize the data
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

#apply padding
from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)


#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

#input length
_InputShape = x_train.shape[1]
print(_InputShape)

#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : keras",vocabulary)

#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)


model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim = 240, output_dim = 64)) 
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20)))
model.add(keras.layers.Dense(45, activation = 'relu'))
model.add(keras.layers.Dense(35, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(x_train, y_train, epochs=200)

print("\n\n")

#chatting
import random
while True:
    texts_p = []
    prediction_input = input('You : ')

    #removing punctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],_InputShape)

    #getting output from model
    output = model.predict(prediction_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    print("Going Merry : ", random.choice(responses[response_tag]))
    if response_tag == "goodbye":
        break
