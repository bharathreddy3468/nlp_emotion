# Emotion Detection with Deployment: 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Installation](#installation)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)
  * [Source](#Source)


## Demo
Link: [https://bharaths-emotion-detector.herokuapp.com/](https://bharaths-emotion-detector.herokuapp.com/)

[![](https://imgur.com/OYDAzwV.png)](https://bharaths-emotion-detector.herokuapp.com/)

[![](https://imgur.com/YtmWIi7.png)](https://bharaths-emotion-detector.herokuapp.com/)

[![](https://imgur.com/k2Bc9MF.png)](https://bharaths-emotion-detector.herokuapp.com/)


## Overview
This is a Flask app which helps in predicting the emotion of a message.

## Motivation
Learning Data science is not enough , showing what you have learned will make a difference. This pandemic helped me in learning data science and also this is my first end to end project.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── static 
│   ├── css
│   ├── images

├── template
│   ├── home.html
    ├── output.html
├── Procfile
├── README.md
├── emotion_classifier.py
├── nlp_emotion(2).ipynb
├── nlp_emotion.ipynb
├── nlp_emotion.sav
├── nltk.txt
├── requirements.txt
├── vecfile.sav
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/bharathreddy3468/nlp_emotion/issues) here by including your search query and the expected result

## Future Scope

* Use multiple Algorithms
* Optimize Flask app.py
* Front-End 

## Source
* data is taken from kaggle 
* got the knowledge of doing from krish naik youtube channel
