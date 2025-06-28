# Recipe Classifier
Author: Ori Ben Yossef ([LinkedIn](https://www.linkedin.com/in/ori-ben-yossef/))

Date: June 28, 2025

This machine-learning-based program classifies recipes as vegan or not vegan, and keto-friendly or not keto-friendly, based on a list of ingredients. I programmed this classifier as part of an online challenge posed by the [Argmax](https://argmaxml.com/) as part of their hiring process.

## Summary

Let's suppose you're making dinner for your vegan best friend and want to look up some tasty vegan recipes on a recipe website. If the recipes are not marked as vegan, you may have to examine each recipe individually to determine whether it's something your friend can eat. This can take a long time, or worse, push you to check a different website instead.

What if machine learning could help? To explore this question, I programed a machine learning model that can classify a recipe as vegan or not vegan, and as keto-friendly or not keto-friendly, so that you don't have to.

To classify each recipe, I took the following approach:

-	Analyze each ingredient separately. A recipe is classified as vegan if all of its ingredients are vegan. Similarly, a recipe is classified as keto-friendly if all of its ingredients are keto-friendly.
-	For each ingredient, isolate the ingredient name. For example, if the ingredient is "1 clove garlic, minced," we want to isolate the word "garlic." This makes it easier for the computer to compare and classify each ingredient. It's simpler to compare "garlic" against "butter" than, say, "1 clove garlic, minced" against "4 tablespoons butter," which contain lots of extra noise.
-	Once we have isolated the ingredient name, we compare it against a training set of other ingredient names and see what its nearest neighbors are. This approach uses a database of vectors corresponding to words. The vector positions are designed such that words with smaller distances are closer together in meaning. If an ingredient name is closer to words marked as vegan (or not vegan), we classify the ingredient as vegan. Similarly with keto. We are assuming that ingredient names closer in meaning to vegan names are more likely to be vegan, and so on, which may or may not be true in practice.

![Schematic showing this model's approach: isolate ingredient name, compare against reference words.](./Schematic%20of%20Model%20Approach.png)

## Data and Methods

To isolate the ingredient name, I trained a conditional random field (CRF) model that labels each word in the ingredient as a name, quantity, unit, comment, or other. Fortunately, New York Times analysts solved this exact problem and made their [data and code publicly available](https://github.com/nytimes/ingredient-phrase-tagger), so I used their data and modified their implementation. You can follow my implementation in the folder "_Step 1 - CRF to Isolate Ingredient Name."

To create the vector space of words against which to compare ingredient names, I came up with about 75 words representing a broad variety of ingredient types. I marked each one as vegan or not vegan, keto or not keto, and meat or not meat (I will explain later why I did this). I used the "wiki-news-300d-1M-subword" dataset [provided by FastText](https://fasttext.cc/docs/en/english-vectors.html), which places about 1,000,000 word vectors in a 300-dimensional space. Notably, such high-dimensional spaces might suffer from a "curse of dimensionality," which occurs when vector distances become less meaningful in higher-dimensional spaces. To avoid this effect and speed up my computations, I used principal component analysis (PCA) to project the vectors onto a 3-dimensional space that best captures the differences between the ~75 words I chose. You can follow my implementation in the folder "_Step 2 - PCA to Transform Word Vectors." (I removed the data file from the GitHub repository because it exceeded GitHub's file size limit.)

To evaluate whether an ingredient name is closer in meaning to vegan or non-vegan ingredient names, I used the k nearest neighbors (KNN) algorithm. I found that the algorithm performed well when looking at the 3 nearest neighbors and weighting them by distance from the target ingredient name. You can follow my implementation in the folder "_Step 3 - kNN to Classify Ingredients."

Notably, some ingredient names were uncommon phrases, such as "chicken breast," which were not contained in the vector dataset. When this occurred, I analyzed the words separately. I noticed that in vegan ingredients, the first word tended to be inconclusive, and the second word tended to be a vegan ingredient name itself (as in "red pepper" or "kalamata olives"). In contrast, in meat-based ingredients, the first word tended to by a type of meat, and the second word tended to be a type of cut (as in "chicken breast" or "chicken thigh"). So, in my set of ~75 words, I added a "meat" criterion in addition to the "vegan" and "keto" criteria. I chose to classify a multiple-word ingredient name as vegan if the last word was classified as vegan and the first word was not classified as a type of meat. I found that my model performed well that way. I used similar thinking to design my keto ingredient name classifier. You can follow this logic in "diet_classifiers.py."

## Conclusion and Discussion

I found that my program classified foods as keto or not keto with 81% accuracy, and vegan or not vegan with 93% accuracy, on a ground-truth dataset of 75 recipes provided by Argmax. (The original dataset had 100 recipes, but I removed 25 duplicate instances of one recipe.)

To me, this suggests that we, as humans, distinguish between vegan and non-vegan foods more readily than we distinguish between keto and non-keto foods. Indeed, I could immediately label my selected words as vegan or non-vegan because all it takes in knowing where an ingredient comes from. However, I am not an expert on keto diets, so I had to look up a guide, categorize my choices using that guide, and modify my categorizations when they led to incorrect results in the ground-truth dataset.

The vector dataset may be computer-based, but it is trained on human-written data. Since humans distinguish between vegan and non-vegan foods more often than keto and non-keto foods, this bias likely manifests in how we write about food online. This, in turn, reflects in how the vectors are positioned relative to each other, with clearer distinctions between vegan and non-vegan foods than between keto and non-keto foods. As a result, if we want to classify words as keto or non-keto, using their meanings and associations may not be an optimal approach. 

![Scatterplot showing the distributions of reference words marked as vegan and non-vegan in the first two principal components, along with the areas predicted to be vegan or non-vegan.](./2D%20Vegan%20Classifier.png)

Vector positions of training words marked as vegan (blue) and non-vegan (red), projected onto the first two principal components. For the most part, they occupy different parts of the space. This allows us to accurately categorize foods as vegan or non-vegan using their vector positions.

![Scatterplot showing the distributions of reference words marked as keto and non-keto in the first two principal components, along with the areas predicted to be keto or non-keto.](./2D%20Vegan%20Classifier.png)

Vector positions of training words marked as keto (blue) and non-keto (red), projected onto the first two principal components. Unlike the vegan classifier, there is lots of overlap between the areas occupied by keto and non-keto word vectors. This leads to mixup, confusion, and lower classifier accuracy. This can be addressed by choosing more meaningful features that more closely relate to a food's nutrition information.

If I were to complete this challenge again, I would try to find a dataset that more directly reflects an ingredient's nutrition information. Then, I would use features from that dataset to more meaningfully position my ~75 training words as vectors. That way, keto and non-keto foods can have a clearer separation in the vector space, and the KNN algorithm can perform better.

## Credits and Acknowledgements

This challenge taught me many important lessons and techniques. I studied how Docker works, I encountered some useful data science libraries and datasets, and I learned the hard way not to rely on outdated analysis software. I discovered some useful tools and tricks, such as the joblib library and its dump and load features, that I will carry with me as I continue working on data science projects. I designed the machine learning classifier, but Argmax staff wrote the surrounding code and designed the graphical user interface. I am grateful to Argmax for the valuable learning opportunity, which helped me expand my capabilities as a data scientist.
