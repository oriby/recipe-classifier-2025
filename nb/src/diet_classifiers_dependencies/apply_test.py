import subprocess
import re
import os

import sklearn_crfsuite
import joblib

# Number of CRF results to search for in case the most likely CRF result
# does not mark any words as "NAME" (i.e., belonging to the ingredient name). 
n_top_options = 3

# Set the working directory to be the current folder when running crf_test.
current_dir = os.path.dirname(__file__)

# The location of the CRF model file.
#model_file_location = "./diet_classifiers_dependencies/ingredient_phrase_tagger_model_file"
#model_file_location = "ingredient_phrase_tagger_model_file"
crf = joblib.load(os.path.join(current_dir, "crf_model2.pkl"))

# "clumpFractions" and "tokenize" are copied and modified functions from the
# "utils" package provided by the New York Times researchers. This is so that
# the ingredient strings are split the exact same way, so the model can be
# applied correctly.

def clumpFractions(s):
    return re.sub(r'(\d+)\s+(\d)/(\d)', r'\1$\2/\3', s)

def tokenize(s):
    american_units = ['cup', 'tablespoon', 'teaspoon', 'pound', 'ounce', 'quart', 'pint']
    for unit in american_units:
        s = s.replace(unit + '/', unit + ' ')
        s = s.replace(unit + 's/', unit + 's ')
    return list(filter(None, re.findall(r'\d*\$\d+(?:/\d+)?|\d+(?:/\d+)?|\w+(?:[-/]\w+)*|[.,()$]', clumpFractions(s))))

def test_input_from_tokens(tokens):
    input_line_list = []
    for itoken in range(len(tokens)):
        paren_status = 'YesPAREN' if str(tokens[itoken]) in ['(', ')'] else 'NoPAREN'
        feature_list = [ str(tokens[itoken]), 'I'+str(itoken+1), 'L4', 'NoCAP', paren_status  ]
        input_line_list.append(dict(zip(   [str(x) for x in range(5)], feature_list   )))
    return [input_line_list]

def ingredient_name_from_sentence(sentence):
    # Identify tokens (i.e., words, but commas and parentheses get their own tokens)
    tokens = tokenize(sentence)
    #tokens = list(utils.tokenize(sentence))
    # Include features, format as needed for crf.predict
    test_input = test_input_from_tokens(tokens)
    # Apply CRF model to predict token categories
    test_output = crf.predict(test_input)
    # List of booleans: which tokens are names
    which_are_names = list(map(lambda x: x[-4:]=="NAME", test_output.flatten().tolist()))
    # True in which_are_names - need to check if a name was found.
    # List of tokens which are marked as names
    name_tokens = [tokens[itoken] for itoken in range(len(tokens)) if which_are_names[itoken]]
    # Join them with a space, then return.
    return ' '.join(name_tokens)
