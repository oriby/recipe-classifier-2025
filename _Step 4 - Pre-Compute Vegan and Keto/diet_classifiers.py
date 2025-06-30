import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

import re
#import apply_test
#import knn_food_classifier
import diet_classifiers_dependencies.apply_test as apply_test
import diet_classifiers_dependencies.knn_food_classifier as knn_food_classifier


# Classify each ingredient, as it appears in the ingredient list,
# as keto or not keto.
def is_ingredient_keto(ingredient: str) -> bool:
    ingredient_class = None
    # Isolate the ingredient name using the CRF model.
    ingredient_name = apply_test.ingredient_name_from_sentence(ingredient)
    # If CRF does not find any ingredient name within the top 3 hits, then
    # return False.
    if ingredient_name is None:
        return False
    # If there is a multiple-word phrase...
    if ' ' in ingredient_name:
        # First, try to find the entire phrase. (for example, "peanut butter"
        # becomes "peanut-butter" which is included in the dataset)
        words_in_phrase = ingredient_name.split()
        hyphenated_phrase = '-'.join(words_in_phrase)
        ingredient_class = knn_food_classifier.predict_word("keto", ingredient_name)
        # If the entire phrase is not found...
        if ingredient_class is None:
            first_word = words_in_phrase[0]
            last_word = words_in_phrase[-1]
            # Say it's keto if the last word is keto OR the first word is a type of meat
            ingredient_class = knn_food_classifier.predict_word("keto", last_word) or knn_food_classifier.predict_word("meat", first_word)
    # If it's just one word, lookup the one word.
    else:
        ingredient_class = knn_food_classifier.predict_word("keto", ingredient_name)
    # If the word (or the component words) are not found in the vector
    # database, then return False.
    if ingredient_class is None:
        return False
    return ingredient_class

# Classify each ingredient, as it appears in the ingredient list,
# as vegan or not vegan.
def is_ingredient_vegan(ingredient: str) -> bool:
    ingredient_class = None
    # Isolate the ingredient name using the CRF model.
    ingredient_name = apply_test.ingredient_name_from_sentence(ingredient)
    # If CRF does not find any ingredient name within the top 3 hits, then
    # return False.
    if ingredient_name is None:
        return False
    # If there is a multiple-word phrase...
    if ' ' in ingredient_name:
        # First, try to find the entire phrase. (for example, "peanut butter"
        # becomes "peanut-butter" which is included in the dataset)
        words_in_phrase = ingredient_name.split()
        hyphenated_phrase = '-'.join(words_in_phrase)
        ingredient_class = knn_food_classifier.predict_word("vegan", ingredient_name)
        # If the entire phrase is not found...
        if ingredient_class is None:
            first_word = words_in_phrase[0]
            last_word = words_in_phrase[-1]
            # Say it's vegan if the last word is vegan AND the first word is not a type of meat
            ingredient_class = knn_food_classifier.predict_word("vegan", last_word) and not knn_food_classifier.predict_word("meat", first_word)
    # If it's just one word, lookup the one word.
    else:
        ingredient_class = knn_food_classifier.predict_word("vegan", ingredient_name)
    # If the word (or the component words) are not found in the vector
    # database, then return False.
    if ingredient_class is None:
        return False
    return ingredient_class

def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))

def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))