from logging import debug
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


app = Flask(__name__)
errors = []
output = ""


@app.route("/")
def home():
    return render_template("index.html", value=output)

@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        seed_text = request.form['word']
        next_words = int(request.form['no'])
        max_sequence_len = 52
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = load_model("next_word.h5", compile=False)
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
    global output
    output = seed_text.capitalize()
    return render_template("index.html", value=output)

if __name__ == "__main__":
    app.run(debug=True, port=5605)