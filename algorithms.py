import json
import numpy as np
import pandas as pd
import os

import vectorizer

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as logr

def vectorize_capandhash(data): #how many caption words and how many hashtags
    listdata = []
    for rec in data:
        vecrec = vectorizer.convert_caption_to_vector(rec)
        listdata.append(vecrec)

    return np.asarray(listdata)

def vectorize_author(data): #is author verified or not
    listauth = []
    for rec in data:
        author = rec["authorMeta"]
        vecauth = vectorizer.convert_author_to_vector(author)
        listauth.append(vecauth)

    return np.asarray(listauth)

def vectorize_audio(data): #is music original or not
    listaudio = []
    for rec in data:
        audio = rec["musicMeta"]
        vecaud = vectorizer.convert_audio_to_vector(audio)
        listaudio.append(vecaud)

    return np.asarray(listaudio)

def vectorize_video(data): #time of video
    listvid = []
    for rec in data:
        vecvid = vectorizer.convert_video_metadata_to_vector(data)
        listvid.append(vecvid)

    return np.asarray(listvid)

def vectorize_commsandshares(data):
    listcomandshare = []
    for rec in data:
        veccs = vectorizer.convert_ground_truth_to_vector(rec).numpy()
        listcomandshare.append(veccs)

    return np.asarray(listcomandshare)

def lin_regression(X,Y):
    linear_model.fit(X,Y)
    lin_coef = linear_model.coef_[0]
    r2 = linear_model.score(X, Y)
    # linear_model.predict

def multi_regression(X,Y):
    linear_model.fit(X,Y)
    lin_coef = linear_model.coef_[0]
    r2 = linear_model.score(X, Y)
    # linear_model.predict
    #X[['a', 'b']]
    #for i in range(4):
    #print('Coefficient of '+ cols[i] + ' is ' + str(round(linear_model.coef_[i],2)))

def logistic_regression(X,Y):
    logistic_model.fit(X,Y)
    r2 = logistic_model.score(X, Y)
    #logistic_model.predict


linear_model = lr()
logistic_model = logr()

file = open("trending.json")
data = json.load(file)
file.close()
print(data)

# CAPTION AND HASHTAGS
capsandhashs = vectorize_capandhash(data)
captions = capsandhashs[:, [0]]
hashtags = capsandhashs[:, [1]]

# AUTHOR
authors = vectorize_author(data)
authverified = authors[:, [2]]

# AUDIO
audios = vectorize_audio(data)
audioorg = audios[:, [2]]

# VIDEO
videos = vectorize_video(data)

# SHARES AND COMMENTS
sharesandcomms = vectorize_commsandshares(data)
shares = sharesandcomms[:, [2]]
comments = sharesandcomms[:, [3]]

#SINGLE LINEAR REGRESSION

#MULTI LINEAR REGRESSION

#LOGISTIC REGRESSION
