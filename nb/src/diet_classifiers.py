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

# Parse the list string contained in the ground truth table.
def parse_list_string(s):
    s = s.strip()[1:-1]
    return re.findall(r"'(.*?)'", s)

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

# I modified the "main" function so that "is_keto" and "is_vegan" receive
# a parsed list of ingredients, rather than an unprocessed string that is
# supposed to represent the list.
# For error analysis purposes, there are optional print statements showing
# each recipe's predicted vegan or keto status.
def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    try:
        start_time = time()
        #ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(
            lambda recipe: is_keto(parse_list_string(recipe)))
        #ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
        #    is_vegan)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            lambda recipe: is_vegan(parse_list_string(recipe)))
        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    #print("Predictions:")
    #print(list(ground_truth['keto_pred']))
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    #print("Predictions:")
    #print(list(ground_truth['vegan_pred']))
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))
