import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib

# Read model data from files
current_dir = os.path.dirname(__file__)
all_vectors_word_index_dict = joblib.load(os.path.join(current_dir, "all_vectors_word_index_dict.pkl"))
all_vectors_transformed_array = joblib.load(os.path.join(current_dir, "all_vectors_transformed_array.pkl"))
knn_classifiers = joblib.load(os.path.join(current_dir, "knn_classifiers.pkl"))

# Predict whether word "word" is classified as True or False by the kNN
# classifier with criterion "criterion". If "word" does not exist in the
# dataset, return None.
def predict_word(criterion, word):
    if word not in all_vectors_word_index_dict:
        return None
    transformed_word_vector = all_vectors_transformed_array[all_vectors_word_index_dict[word]]
    word_prediction = knn_classifiers[criterion].predict(transformed_word_vector[0:3].reshape(1,-1))
    return word_prediction