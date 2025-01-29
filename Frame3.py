import requests
from bs4 import BeautifulSoup
import numpy as np
import cv2
import speech_recognition as sr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from gtts import gTTS
import os

# Sample dataset (for demonstration, use a larger dataset for better performance)
texts = [
    "Once upon a time, in a land far away, there lived a princess.",
    "She loved to explore the enchanted forest near her castle.",
    "One day, she found a magical flower that granted wishes.",
]

# Prepare the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.arange(len(texts))  # Dummy labels for demonstration

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Text-to-Speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

# Capture Visual Input
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return frame

# Capture Auditory Input
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print('Listening...')
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print(f'You said: {text}')
        return text
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
        return None
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except Exception as e:
        print(e)
        return None

# Scrape Wikipedia for Language Data
def scrape_wikipedia(topic):
    url = f"https://en.wikipedia.org/wiki/{topic}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([para.text for para in paragraphs[:5]])  # Get first 5 paragraphs
    return content

# Generate text using the trained model
def generate_text(seed_text):
    seed_vector = vectorizer.transform([seed_text])
    predicted_label = model.predict(seed_vector)[0]
    response = texts[predicted_label]
    return response

# Main Loop
def main_loop():
    while True:
        user_input = recognize_speech()
        if user_input:
            # Scrape Wikipedia for knowledge on communication
            wiki_content = scrape_wikipedia("Communication")
            print(f'Wikipedia content: {wiki_content}')
            speak(wiki_content)

            response = generate_text(user_input)
            print(f"AI: {response}")
            speak(response)

# Start the main loop
main_loop()
