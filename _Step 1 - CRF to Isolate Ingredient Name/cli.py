import re
import decimal
import optparse
import pandas as pd

import ingredient_phrase_tagger.training.utils as utils
#import utils

#import importlib
#importlib.reload(utils)
import sys
sys.dont_write_bytecode = True


def generate_data(data_path, count, offset):
    """
    Generates training data in the CRF++ format for the ingredient
    tagging task
    """
    df = pd.read_csv(data_path)
    df = df.fillna("")

    start = int(offset)
    end = int(offset) + int(count)

    df_slice = df.iloc[start: end]

    X = []
    y = []

    for index, row in df_slice.iterrows():
        X_sentence = []
        y_sentence = []
        
        try:
            # extract the display name
            display_input = utils.cleanUnicodeFractions(row["input"])
            tokens = utils.tokenize(display_input)
            del(row["input"])

            rowData = addPrefixes([(t, matchUp(t, row)) for t in tokens])

            for i, (token, tags) in enumerate(rowData):
                features = utils.getFeatures(token, i+1, list(tokens))
                # IAT
                token_with_features = [token] + features
                token_feature_dict = dict(zip(range(len(token_with_features)), token_with_features))
                token_feature_dict = dict(zip([str(x) for x in range(len(token_with_features))], token_with_features))
                #print("Feature dictionary:", token_feature_dict)
                chosen_label = bestTag(tags)
                #print("Label:", bestTag(tags))
                #print(utils.joinLine([token] + features + [bestTag(tags)]))
                X_sentence.append(token_feature_dict)
                y_sentence.append(chosen_label)

            X.append(X_sentence)
            y.append(y_sentence)

        # ToDo: deal with this
        except UnicodeDecodeError:
            pass

        #print()
    return X, y

def parseNumbers(s):
    """
    Parses a string that represents a number into a decimal data type so that
    we can match the quantity field in the db with the quantity that appears
    in the display name. Rounds the result to 2 places.
    """
    ss = utils.unclump(s)

    m3 = re.match('^\d+$', ss)
    if m3 is not None:
        return decimal.Decimal(round(float(ss), 2))

    m1 = re.match(r'(\d+)\s+(\d)/(\d)', ss)
    if m1 is not None:
        num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
        return decimal.Decimal(str(round(num,2)))

    m2 = re.match(r'^(\d)/(\d)$', ss)
    if m2 is not None:
        num = float(m2.group(1)) / float(m2.group(2))
        return decimal.Decimal(str(round(num,2)))

    return None


def matchUp(token, ingredientRow):
    """
    Returns our best guess of the match between the tags and the
    words from the display text.

    This problem is difficult for the following reasons:
        * not all the words in the display name have associated tags
        * the quantity field is stored as a number, but it appears
          as a string in the display name
        * the comment is often a compilation of different comments in
          the display name

    """
    ret = []

    # strip parens from the token, since they often appear in the
    # display_name, but are removed from the comment.
    token = utils.normalizeToken(token)
    decimalToken = parseNumbers(token)

    for key, val in ingredientRow.items():
        if isinstance(val, str):

            for n, vt in enumerate(utils.tokenize(val)):
                if utils.normalizeToken(vt) == token:
                    ret.append(key.upper())

        elif decimalToken is not None:
            try:
                if val == decimalToken:
                    ret.append(key.upper())
            except:
                pass

    return ret


def addPrefixes(data):
    """
    We use BIO tagging/chunking to differentiate between tags
    at the start of a tag sequence and those in the middle. This
    is a common technique in entity recognition.

    Reference: http://www.kdd.cis.ksu.edu/Courses/Spring-2013/CIS798/Handouts/04-ramshaw95text.pdf
    """
    prevTags = None
    newData = []

    for n, (token, tags) in enumerate(data):

        newTags = []

        for t in tags:
            p = "B" if ((prevTags is None) or (t not in prevTags)) else "I"
            newTags.append("%s-%s" % (p, t))

        newData.append((token, newTags))
        prevTags = tags

    return newData


def bestTag(tags):

    if len(tags) == 1:
        return tags[0]

    # if there are multiple tags, pick the first which isn't COMMENT
    else:
        for t in tags:
            if (t != "B-COMMENT") and (t != "I-COMMENT"):
                return t

    # we have no idea what to guess
    return "OTHER"
