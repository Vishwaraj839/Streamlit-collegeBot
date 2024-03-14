import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model(r"C:\Users\MS\Downloads\VITchat\model.h5")
import json
import random
intents = json.loads(open(r"C:\Users\MS\Downloads\VITchat\intents.json").read())
words = pickle.load(open(r"C:\Users\MS\Downloads\VITchat\texts.pkl",'rb'))
classes = pickle.load(open(r"C:\Users\MS\Downloads\VITchat\labels.pkl",'rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])    
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
def format_response(response):
    # Add HTML markup for proper formatting
    formatted_response = f"<br>{response}<br>"
    return formatted_response


import streamlit as st



# Streamlit app code
def main():
    st.title("VITChat App")

    # Create a text input for user messages
    user_input = st.text_input("You: ", "")

    # Check if user has entered a message
    if user_input:
        # Get response from chatbot
        bot_response = chatbot_response(user_input)
        # Format the response
        formatted_bot_response = format_response(bot_response)
        # Display chatbot response with proper formatting
        st.markdown(formatted_bot_response)

if __name__ == "__main__":
    main()
# def main():
#     st.title("Chatbot App")

#     # Create a text input for user messages
#     user_input = st.text_input("You: ", "")

#     # Check if user has entered a message
#     if user_input:
#         # Get response from chatbot
#         bot_response = chatbot_response(user_input)
#         # Display chatbot response
#         st.text_area("Chatbot: ", value=bot_response, height=100)

# if __name__ == "__main__":
#     main()




# from flask import Flask, render_template, request

# app = Flask(__name__)
# app.static_folder = 'static'

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return chatbot_response(userText)


# if __name__ == "__main__":
#     app.run(debug=True)

