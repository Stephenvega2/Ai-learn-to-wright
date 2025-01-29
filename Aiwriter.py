import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pyttsx3

# Sample dataset (for demonstration, use a larger dataset for better performance)
texts = [
    "Once upon a time, in a land far away, there lived a princess.",
    "She loved to explore the enchanted forest near her castle.",
    "One day, she found a magical flower that granted wishes.",
]

# Preprocessing the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Model Training
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_sequence_length-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Prepare data for training
X = padded_sequences[:, :-1]
y = tf.keras.utils.to_categorical(padded_sequences[:, -1], num_classes=vocab_size)

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Generate text using the trained model
def generate_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

# Main Loop
def main_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_text(user_input)
        print(f"AI: {response}")
        speak(response)

# Start the main loop
main_loop()
