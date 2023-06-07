import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()


def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    return sequences


def encode_seq(seq, mapping):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict character
        predict_x = model.predict(encoded, verbose=0)
        classes_x = np.argmax(predict_x, axis=1)
        # reverse map integer of character
        out_char = ''
        for char, index in mapping.items():
            if index == classes_x:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


def main(): 
    with open('data.txt') as f:
        lines = f.readlines()
    data_text = '\n'.join(lines)
    
    # preprocess the text
    data_new = text_cleaner(data_text)
    # create sequences   
    sequences = create_seq(data_new)
    # create a character mapping index
    chars = sorted(list(set(data_new)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    # encode the sequences
    sequences = encode_seq(sequences, mapping)
    
    # vocabulary size
    vocab = len(mapping)
    sequences = np.array(sequences)
    # create X and y
    X, y = sequences[:,:-1], sequences[:,-1]
    # one hot encode y
    y = to_categorical(y, num_classes=vocab)
    # create train and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # define model
    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
#     print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_tr, y_tr, epochs=100, verbose=1, validation_data=(X_val, y_val))
    
    print('Please enter the phrase you want completed or "exit" to end the program.')
    string = input()
    while string != 'exit':
        print(generate_seq(model, mapping, 30, string.lower(), len(string)))
        
        print('Please enter another phrase or "exit".')
        string = input()
    
    
main()
    