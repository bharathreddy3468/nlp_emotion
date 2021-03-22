from flask import Flask, request, jsonify, render_template
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import joblib


wn = WordNetLemmatizer()
punctuations = ['!', '(', ')', '-', '[', ']', '{', '}', ';', ':', '<', '>', '.', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~', '\'', '"', "\\"]


def lem(sen):
    words = []
    for word in sen:
        if word not in [stopwords.words('english'), punctuations]:
            wn.lemmatize(word.lower())
            words.append(word)
    sen = ' '.join(words)
    return [sen]


img_folder = os.path.join('static', 'images')
app = Flask(__name__)
model = joblib.load('nlp_emotion.sav')
cv = joblib.load('vecfile.sav', 'r')
app.config['img_folder'] = img_folder

@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [request.form['message']]
    sen = lem(int_features)
    print(sen)
    sen_array = cv.transform(sen).toarray()
    prediction = model.predict(sen_array)[0]
    print(prediction)

    output_dict = {'happy': ['"you are so happy"', 'happy.jpg'],
                   'joy': ['"you are joyful"', 'joy.jpg'],
                   'love': ['"you are expressing your love"', 'love.png'],
                   'sadness': ['"you are so sad"', 'sad.png'],
                   'fear': ['"you are so scared"', 'scared.jpg'],
                   'surprise': ['"surprise!!!!!!!!!!!!!!!!!"', 'surprise.jpg']}
    full_img_path = os.path.join(app.config['img_folder'], output_dict[prediction][1])
    return render_template('output.html', prediction_text=output_dict[prediction][0], full_img_path=full_img_path)


if __name__ == "__main__":
    app.run(debug=True)
