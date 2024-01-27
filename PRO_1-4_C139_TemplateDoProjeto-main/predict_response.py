# Bibliotecas de pré-processamento de dados de texto
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np
import random
import tensorflow
from tensorflow.keras.models import load_model

# Download de recursos necessários para o NLTK
nltk.download('punkt')

# Carregue o modelo
model = load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))

# Palavras a serem ignoradas/omitidas ao enquadrar o conjunto de dados
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

def preprocess_user_input(user_input):
    bag = [0] * len(words)

    # Tokenize a entrada do usuário
    tokens = word_tokenize(user_input)

    # Converta a entrada do usuário em suas palavras-raiz: stemização
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]

    # Remova duplicatas e classifique a entrada do usuário
    unique_tokens = sorted(set(stemmed_tokens))

    # Codificação de dados de entrada: crie a BOW para user_input
    for token in unique_tokens:
        if token in words:
            bag[words.index(token)] = 1

    return np.array(bag)

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    inp = inp.reshape(1, len(words))

    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])

    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]

    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            responses = intent['responses']
            bot_response = random.choice(responses)
            return bot_response

print("Oi, eu sou a Estela, como posso ajudar?")

while True:
    # Obtenha a entrada do usuário
    user_input = input('Digite sua mensagem aqui: ')

    if user_input.lower() == 'sair':
        break

    response = bot_response(user_input)
    print("Resposta do Robô:", response)
