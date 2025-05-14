Emotion Detector

A modular Python pipeline for training, evaluating, and batch-predicting emotions in free-form text, using the GoEmotions dataset.


How to use
Be sure to follow the steps in order.

1. Train your models

This will load GoEmotions, preprocess, vectorize, train two classifiers, and write:
results/logistic_regression.joblib
results/linear_svc.joblib
results/metrics.json

RUN train.py

2. Evaluate on the held-out split

Generates confusion-matrix PNGs for the top few emotions, saved under results/:

RUN eval.py

3. Batch-predict new entries

important!!!
Place your text file at User/1.txt (one sentence per line). Then simply run:

RUN Predict.py
By default this uses results/logistic_regression.joblib and User/1.txt.

