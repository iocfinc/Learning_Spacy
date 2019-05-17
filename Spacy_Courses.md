# Sentiment Analysis

To tackle the Sentiment Analysis problem I am going back to the roots. First up is **What is Sentiment Analysis?** Also called *Opinion Mining* in Natural Language Processing (NLP), Sentiment analysis is the field where we try to identify and extract **opinions** within text. Together with the *opinion* the system also extracts *Polarity*, *Subject* and *Opinion Holder* from the statement.

The applications of sentiment analysis include automatic curation of reviews and texts of opinions from vast sources (forums, review sites, blogs, social media posts) and getting the general public's opinion for a product/service. Commercially, the applications potentially include marketing analysis, public reactions (to a news or a product), product review, promotion scoring (advertisement), product feedback and customer service.

Now that we have an idea of sentiment analysis we read up on **What an opinion is on the context of NLP?** So text information can be categorized, broadly, into *facts* and *opinions*. Facts are the **objective** expression about something. Opinions, on the other hand, are the **subjective** expressions describing the user's sentiment or input towards a certain topic or subject.

Like many NLP problems, we can model sentiment analysis as a classification problem where two sub-problems have to be resolved:

* Classification of the input text as *subjective* or *objective*, (**subjectivity classification**) and
* Classification of the input as an expression of *positive*, *negative* or *neutral* opinion (**polarity classification**).

Now that we know what opinions are we can move on to defining the different kinds of opinions. Opinions can be *direct* or *comparative*. Direct opinions talk directly about the subject and its feature. Comparative opinions would compare the feature of one subject to another. Below we have sample opinions on price for a subject.

* Comparative: "Model A's price is lower than Model B"
* Direct: "Model A is cheap"

Opinions can also be *explicit* or *implicit*. If it is an explicit opinion, the opinion is given directly to a certain feature of the subject. In implicit, the opinion provide implies to the feature of the subject. Below we have opinions on the quality of Model A.

* Explicit: "Model A's quality is poor."
* Implicit: "My Model A broke in 2 minutes."

We can now move on to the different possible scope of sentiment analysis. Sentiment analysis can be applied at a *Document Level* where the entire document or paragraph is considered. It can be applied in the *sentence level* where we try to obtain the sentiment of individual sentences. Finally we can apply it to a *sub-sentence level* where we can choose to figure out the sentiment of a phrase or a word.

## Different Sentiment Analysis Types

There could be different types of sentiment analysis model depending on what information we are trying to get. Our model could be focused on determining polarity (positive or negative). We can choose to detect emotions or feelings (angry, happy, sad). We can create a model that can detect intention (interested or not interested).

Our model can vary in *granularity* where we can choose to create a model that is binomial or we can choose to create a multi-class model. An example of this would be (positive or negative) and (Very Positive, Positive, Neutral, Negative, Very Negative). We can also add another layer of classification for our model, for example for Negative text we can choose to find the associated feeling the opinion is conveying (sadness, frustration, anger).

There is also something called *Aspect-based sentiment analysis* where we try to identify the particular subject the opinion is referring to. For example we take the sentence: "The workmanship of Model A is terrible". In the sample sentence, the subject of the opinion is still Model A but the sentiment it is referring to is the workmanship which is an aspect of Model A.

We can also find *Intent Analysis* models. These attempt to figure out if the person will commit to an action based on the text. This is somewhat a specialized concept because it might need certain contextual knowledge of the subject. An example of this would be predicting if the customer is about to churn based on his sentence. "The quality of service is terrible. Your customer support kept me on hold for an hour and the issue was not resolved." Based on the sentence, it is probable that the likelihood of that customer to file a complaint or find another provider is high. For a human this is possibly easy given the context but this could be very specialized for a machine.

## Different Sentiment Analysis Algorithms

There are many possible algorithms and methods on implementation of sentiment analysis. From the source they provided three simple classification. First is **Rule-based**, **Automatic**, **Hybrid**.

The simplest to define would be **Hybrid**, its just the mix of **Rule-based** and **Automatic**. By combining the two we are hopefully creating a better resultant model.

For rule-based approach this is going to be programmed in a way where there are rules on how the text are to be classified. These would usually be in script form of if-else statements looking for features in the text and determining the opinion formed by the text. The inputs for these rules could be stemming, tokenization, part of speech tagging and parsing. Another possible reference of rules would be lexicons, which are basically list of words or expressions.

A good example of rule-based model would be going over the words of a sentence then counting the number of positive and negative words. From the count we can check the *general sentiment* of the sentence. Obviously, there is a very great downside to these methods. The method we just made a sample of would not check for context. It would not be able to know how the word or the phrase was used. For example: "pretty bad movie" would turn out neutral since we have 1 positive (pretty) and 1 negative (bad). In the phrase, pretty was used an adverb adding a degree to the adjective bad.

We can now look at the **Automatic approach** for sentiment analysis. This would rely on machine-learning models where the task is described as a classification problem where the input is the text and the output is the category.

![Flow of automatic approach](https://monkeylearn.com/static/img/automatic-approaches.png)

The automatic approach usually follows the flowchart above. The training model is first created by using feature extraction on the input which would then be used for the machine learning algorithm. Once the model is trained sufficiently it can be used for prediction of inputs. In terms of *feature extraction* the machine learning model will be using the numerical representation of the text, usually as a vector. Often these vector represents the frequency of words in a predefined dictionary(e.g. a lexicon of polarized words). This process is called feature extraction or text vectorization. The classical approach has been bag-of-words (BOW) or ngrams for frequency. There are also new methods to extract features where the feature is based on word embeddings (or word vectors). These resolves words of the same meaning to have closer almost similar representation.

For the actual classification normally we use a statistical model like *Naive Bayes*, *Logistic Regression*, *Support Vector Machines* or *Neural Networks*.

## Metrics and Evaluation of Sentiment Analysis Models

Now that we have an idea of how a model is constructed, we can talk about how the model would be scored. One of the most common way to obtain the performance metric and evaluation of the classifier is to use *cross-validation*. In cross-validation we split or data in to folds. Usually the folds are *training* (~75% of the entire set) and *testing* (~25% or remainder of set). The model is trained on the training fold and the resultant model is tested on the testing fold to get the metrics. This is done multiple times to improve the model.

### Precision, Recall and Accuracy

The most standard metrics for any classifier performance would include *Precision*, *Recall*, *Accuracy*.

* Precision - out of all the predicted text classified into the category, how many were actually in the category. I am thinking: (Correct A - Incorrect A)/(Total A). High precision would mean our correct A is high and Incorrect A is low.
* Recall - out of all the correctly predicted text classified into the category, how many were actually in the category. Thinking of: (Correct A)/(Total A).
* Accuracy - out of the entire corpus, how many were correctly classified. (Total Correct)/(Total Size). High accuracy means that the entire model is trained to predict correctly each individual classification.



## Creating an Actual Model

Right now I want to showcase an app project that I can create using NLP by using Spacy with Sense2Vec. Right now I just want to be able to do a sentiment analyzer app. Then I can build the front end website.

Right now I am encountering an error with the vector model for reddit dataset.  I am unable to unpack the tar file.

## Learning Spacy

Okay, I have to admit that it was really frustrating to get blocked by a tar file issue on Sense2Vec model. I don't want to work on it for now. It just left me completely pissed. So right now I am going over the free lessons in Spacy by the creator of Spacy herself, Ines.

I am starting out late again. I voted earlier today and it was quick. Hopefully there would be changes that would come out of the choices the Filipino made today.

So far I have gone through the different languages models in Spacy. Right now Spacy can support 45 languages and several models. It should be noted that obviously, not all the supported language have models created for them. Only the common models have

Then we went on to *Documents*, *Spans* and *Tokens*. They are usually the different make up of the input text file. So *Document* would be the entire sentence or the paragraph that is fed to the Spacy method created. *Spans* are *slices* of the document based on its individual *token* components. To call a span, Spacy uses the same notation as a list slice but this time for Doc, so that would mean `doc[start of span : end of span+1]`. Do note that similar to Python slicing this is also an index zero operation. Finally we have the *Token* which is the most basic form that the Spacy model can create. *Tokens* are basically the individual word, character, number or punctuation detected by Spacy. All of the tokens detected by Spacy can have different attributes related to it For example, `index`, `is_alpha`, `pos`, `like_num`, `text`. Most of the documentation for the list of attributes are in the API documentation.

Now its 2:00AM and I am on the Statistical Models in Spacy. So first of, what is a statistical model? Basically its the trained model loaded by Spacy that will allow it to provide predictions *in context* like *Part of Speech* tags and *Syntactic dependencies* map. These models have been pre-trained on large data sets of labeled examples. In the context of transfer learning, its the VGG-19 or GloVe or Bert models that we can use to allow us to have a pre-trained network which we can readily use to predict attributes. Similar to other models in Computer Vision like VGG, we can also train the pre-existing model in Spacy to better suit our use case for fine-tuning the results.

There are readily available models in Spacy's library that we can use for our use case. They differ in their sizes, the vocabulary they have and the training corpus they were trained on. What is included in majority of these models would be the *Binary weights*, *Vocabulary* and *Meta Information*. Here is how I see it, the Binary Weights are similar to the pre-trained weights we get from training our neural network for a task. Its like loading a checkpoint file which already has a pre-trained model. In addition to this the models already have the Vocabulary, which is dependent on the size of the model loaded. So the vocabulary is basically the reference of the model to identify POS and several other attributes. The vocabulary comes from the training of the model on the data set. Finally, we have the Meta Information which is basically the information needed by Spacy to allow us to use the model. Think of this as the hyperparameters of the model. The pipeline and the language which the model was trained on is included on the Meta Information so its like the number of hidden layers and learning rate information that we normally get in a Neural Network.

To get a list of available models in Spacy we can refer to the Language Support documentation. To download a model, we simply have to call `spacy download <model_name>`. This is usually done with Python so `python -m spacy download en_core_web_sm` is a sample command that can be used. In this example we are going to be downloading the English model that has a small vocabulary. To load a model we simply have to use the method `spacy.load('<model_name>`. So the full command upon initialization would be:

```python
import spacy

nlp = spacy.load('en_core_web_sm')
```
Now that we have a model loaded, we can start with some predictions. First up would be the Part-of-speech tags. For POS we can call the attribute by using `token.pos_`. A quick note from Ines is that attributes in spacy that return strings usually end with underscores while those that do not end with underscores return an ID. On the code below we are going to process the doc which is "She ate the pizza". The input text is fed to the nlp model we have loaded before which is `en_core_web_sm`. One the input text is passed we can already get the attributes we can by iterating over the tokens in the doc we have defined. Notice that token and doc are of the same object that's why python knows what we refer to when we say for tokens in doc. in this case we want to print out the Part-of-speech tag for the model so we call on the `token.pos_` attribute to be printed.

```python
# NOTE: This is building up on the model loading we did previously.
# process a sample text: `She ate the pizza"
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # Print the token and the predicted part-of-speech tag
    print(token.text, token.pos_)
```
So the results in running the code above is pasted below. We see that the model predicts She as a PRON (pronoun), ate as a VERB, the as a DET (determiner) and pizza as a NOUN. So far this are all correct. 

```text
She PRON
ate VERB
the DET
pizza NOUN
```

But Spacy can do so much more that just basic POS tagging. It can also predict Syntactic dependencies, it can detect if the word/token is the subject or object of the sentence/doc. It can also provide a prediction on which part or word in the sentence a token is referring to. To call on the dependency tag we use `token.dep_`, then we can use `.head` attribute to return the syntactic head token or basically the parent token that the current word is attached to.

```python
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
```
which the output would be the one below. Breaking it down we see that the dependencies are printed. She is classified as the subject while ate is a ROOT, pizza is considered the object. The head attribute also shows the parent word for the token. For example, you see that She is linked to ate and pizza is linked to ate and finally the is a determinant pointed towards pizza. So far its correct.

```text
She PRON nsubj ate
ate VERB ROOT ate
the DET det pizza
pizza NOUN dobj ate
```

To break down the labels and to have uniformity, Spacy has a table of labels for the resulting dependency. For the example we have above, nsubj is the *nominal subject*, dobj is referring to the *direct object*.

Spacy can also detect *Named Entities* which are real world objects that are assigned a name like a person, a counter or an organization. Using `doc.ents` property allows us to access the named entities prediction by the model. This will return an iterator of span objects which we can then build on with the `text` and `label_` attributes. In the example below we see that Spacy detects `Apple` as an organization, `U.K.` as a Geopolitical entity and `$1 billion` as Money.

```python
#Process a sample text
doc = nlp(u"Apple is looking at buying U.K. startup for $1 billion*)

# Go over the detected entities in the document
for ent in doc.ents:
    # Print the entity and its labels
    print(ent.text, ent.label_)
```

The result of the code above would be seen below. To better understand the definitions of common tags and labels we can use `spacy.explain('<tag/label>')`.

```text
Apple ORG
U.K. GPE
$1 billion MONEY
```

Its now 12:05 AM, so its May 15. Its now another day. I was having some issues submitting my Request for Clearance through Service Now. The attachment tool seems to be broken. I will try again later today. For now I will be completing the Spacy lesson.

I am already in **Rule Based Matching** in Spacy. Of note for this functionality is the fact that it has a better function than just plain regular expressions (regex). In Spacy we are allowed to match `Doc` objects and not just strings, this means that we can match tokens based on their context like if its a verb or a noun and not just all the words that appear as the same string. An example of this would be "duck" the verb and "duck" the noun. If this was a plain regular expression then the vocabulary would not be able to distinguish how the word was used. On top of this, Spacy's rule based matching is flexible enough to allow not just search for text but for lexical attributes as well.  We can also use rules based on the model's prediction.

One step further of rule based matching is *Match Patterns*. Match patterns in Spacy are list of dictionaries describing a token. The keys would be the name of the token attribute and the value would be the expected value of the token text. An example of this would be: `[{'TEXT': 'iPhone'}, {'TEXT' : 'X'}]`. In this example, we are looking for two tokens that are text with values 'iPhone' and 'X'. Aside from token attributes, Spacy allows us to use lexical  attributes for example: `[{'lower': 'iphone'}, {'lower' : 'x'}]`. This example would look for two tokens whose lowercase values would match 'iphone' and 'x' which would mean it can catch misspelled words that indicate an iPhone like 'IPHONE' or Iphone'. Finally, we see an example of a match pattern using token attributes. We can use token attributes to find expressions in the document matching the pattern. For example: `[{'LEMMA': 'buy'}, {'POS': 'NOUN'}]`. In this example we are looking for any expression that has a LEMMA `buy` plus a NOUN. So an example match for this would be: `buying flowers` or `bought milk`.

Now that we know what a Spacy Matcher can do, we can proceed with an example of invoking the matcher via code.

```python
import spacy
# Load the Matcher
from spacy.matcher import Matcher

# Load the model we are going to use
nlp = spacy.load('en_core_web_sm')

# Initialize the Matcher by sharing the vocab from the model
matcher = Matcher(nlp.vocab)

# Define the pattern and add it to the Matcher
pattern =[{'TEXT' : 'iPhone'}. {'TEXT' : 'X'}]
matcher.add('IPHONE_PATTERN', None, pattern)

# Define the sample text
doc = nlp('New iPhone X release date leaked')

# Call the matcher
matches = matcher(doc)
```
The code for a matcher is defined above. Here is the breakdown of what is needed for the matcher. First is that we need to import matcher from Spacy. Next we initialize the matcher with a shared vocabulary from the loaded model (in this case `en_core_web_sm`). We then create a match pattern list and add that list to the matcher via `matcher.add`. The arguments for `matcher.add` would be the unique ID for the matcher (`IPHONE_PATTERN`), the optional callback which is not used on this case and the match pattern which we defined earlier as `pattern`. To match a pattern we just need to call matcher to any doc. Its important to note that we first need to make the text file as a doc otherwise there would be no matches.

Once we call a matcher to a document we actually get a return of three values in a tuple. They are the match_id, start and end. The `match_id` is the unique hash value for the pattern, the `start` is the start index of the span of text that matched and `end` is the end index of the matched span. So we can actually iterate over the matches with the code below:

```python
# Calling the doc to a text
doc = nlp('New iPhone X release date leaked')
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
```
The code above should produce the result `iPhone X` as the output since there is only one match. Its important to note here that the `matched_span` here is a span so it does have the same attributes as `.text`

Another example with a more complex lexical match pattern is given below:

```python
pattern = [
    {'IS_DIGIT' : TRUE},
    {'LOWER' : 'fifa'},
    {'LOWER' : 'world'},
    {'LOWER' : 'cup'},
    {'IS_PUNCT' : True}
]

doc = nlp('2018 Fifa World Cup: France won!)
```
The code above would result to `2018 Fifa World Cup:`. The pattern is broken down into five consecutive tokens where the first token would be a digit, three tokens that are case insensitive with values `fifa`, `world` and `cup`, and finally a punctuation token.

Some more examples:

```python
pattern = [
    {'LEMMA' : 'love', 'POS': VERB},
    {'POS' : 'NOUN'}
]

doc = nlp("I loved dogs but now I love cats")
```
This code would be looking for a lemma that is love which is a verb which is followed by a noun. As we can see, we can match patterns where there is more than one dictionary value for a token. In this case we see a combination of a LEMMA and VERB pattern for the first token. This means that any word that is a LEMMA of love and acts as a verb in the context is considered. Do note that this is building on the prediction of POS and other attributes that was loaded from the model. The result of running the code would be two matches: `loved dogs` and `love cats`. In this case both `loved` and `love` are verbs and not nouns so they are considered and they are followed by nouns.

A unique key-value pair for pattern matcher would be the operator-quantifiers pair. Operators and quantifiers let us define how often the token should be matched. Operators are added via the `OP` key. In the code below the quantity `?` makes the operator optional which in this case is an article (determiner) so it will match a token lemma of `buy`, an optional article and a noun token.

```python
pattern = [
    {'LEMMA' : 'buy'},
    {'POS' : 'DET', 'OP' : '?'}, # An optional token meaning it will try to match 0 or 1 times
    {'POS' : 'NOUN'},
]

doc = nlp('I bought a smartphone now I am buying apps')
```
This code would result in two  matches. First is `bought a smartphone` with `bought` a lemma of buy, `a` as an optional determiner and `smartphone` as the noun. The next result is `buying apps` with `buying` as a lemma of buy and `apps` as the noun with the optional determiner waived. To complete the operators `OP` pair we have the list of possible values below. There are only 4 possible values for the Operator.

```text
{'OP' : '!'} - A **!** value Negates the token so it is matched 0 times. Makes the current pair inactive in the pattern.
{'OP' : '?'} - A **?** value makes the token optional so it is matched either 0 or 1 times.
{'OP' : '+'} - A **+** value means that the token is matched 1 or more times. Should at least be one time but could be more
{'OP' : '*'} -  A __*__ value means that the token is matched 0 or more times. Can be optional but could be more.
```

Going over the exercises here are some additional use cases.

```python
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [{"TEXT": 'iOS'}, {"IS_DIGIT": True}]
```

```text
# OUTPUT
Total matches found: 3
Match found: iOS 7
Match found: iOS 11
Match found: iOS 10
```

```python
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# Write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": 'download'}, {"POS": 'PROPN'}]

```

```text
# OUTPUT
Total matches found: 3
Match found: downloaded Fortnite
Match found: downloading Minecraft
Match found: download Winzip

```

```python
doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write a pattern for adjective plus one or two nouns
pattern = [{"POS": 'ADJ'}, {"POS": 'NOUN'}, {"POS": 'NOUN', "OP": '?'}]
```

Notice how the pattern was structured.  In this case its important to know how many times the nouns are expected to appear. We could not use {POS: NOUN} and {POS: 'NOUN', OP:'*'} for this one. This was  my initial answer but if you study the pattern desired we need to have at least one with the 2nd token optional so using * value would mean that we are allowing at least one but it could be two or three or more which makes the logic incorrect. So we see here that the combination of the 4 possible values are very powerful and could also be very incorrect at times especially if we coded it incorrectly.

```text
Total matches found: 4
Match found: beautiful design
Match found: smart search
Match found: automatic labels
Match found: optional voice responses
```

For now this is the end of the **Introduction to Spacy*. I am ending the night on this one. I will resume tomorrow. Up next is **Large-Scale data analysis with Spacy**. I just checked and there are 4 chapters for the tutorial. After **Large-scale data analysis with Spacy** we would be taking on **Processing Pipelines** and finally the last chapter is **Training a Neural Network Model**.

## Large Scale Data Analysis with Spacy (15 modules)

May 15, 2019

I am now going over Lesson 2 of the Course in Spacy. I have completed individual tokens and attributes in lesson 1, lesson 2 is more of using the pipeline in Spacy to create models for large scale data. First off would be Data Structures and we begin with the shared vocabulary in Spacy. `Vocab` stores data that is shared across multiple documents. The `Vocab` would include words but also label schemes and tags and entities. Think of vocab as a an actual Dictionary used by Spacy which contains the words it knows as well as common terms. The vocab strings are actually saved as a hash ID to save memory. This would mean that if the word is appearing more than once we do not have to save it every time. To store a string to vocabulary we can use `word_hash = nlp.vocab.strings[<word str>]`. We have to note that what this creates would be a bi-directional lookup table of string-hash ID pairs. Internally, Spacy is only using hash IDs in its transactions. While string-hash ID pairing is bidirectional it is not reversible, meaning that if a word is not in a vocabulary there is no way for us to retrieve its string.

To get the string or hash value from the vocab we are going to use `nlp.vocab.strings[<String or Hash ID>]'. It is also possible for the hash to be reveled via the doc instead of the nlp. Example below.

```python
doc = nlp('I love coffee')
print('Hash Value: '. nlp.vocab.strings['coffee'])
print('String Value: ', nlp.vocab.strings[3197928453018144401])
```

Note that this is just an example so we already know the Hash value for the string 'coffee' beforehand. The code above should result in:

```text
Hash value: 3197928453018144401
String value: coffee
```

Also we can use `doc` to expose the vocab and strings. See example below.

```python
doc = nlp('I love coffee')
print('Hash Value: '. doc.vocab.strings['coffee'])
```

The result for the code above should be `Hash value: 3197928453018144401`. We are just after the example of using doc as an alternative to using nlp.

After vocab we move on to `lexeme`. *Lexemes* are objects which are entries in the vocabulary. Lexemes are context-independent, meaning they do not have attributes like POS, Dependencies or Entities tag. We could get a lexem by looking up a string or hash ID from the vocabulary. An example would be:

```python
doc = nlp("I love coffee")
lexeme = nlp.vocab['coffee']

# Print out lexical attributes
print(lexeme.text, lexeme.orth, lexeme.is_alpha)

# Output
# coffee 3197928453018144401 True
```

The code above is calling for the entry in the vocab for 'coffee'. Once we call a lexeme we can ask for its attributes like the `text` which is the string for the lexeme, `orth` which is the hash ID and `is_alpha` which checks the string if its numeric. To better explain how a lexme works in the context of Spacy we have the image below. To note where the lexeme stands and to understand why it is context-independent we should look at the location of a lexeme based on the document and the vocab. A `doc` will have multiple token inputs. The in-context attribute like POS and dependencies happen in doc level. We know that Spacy only works with Hash internally so for it to know the word it has to consider it's hash ID, this is where the lexeme layer is. As we can see, the pairings of the hash to text happens in the vocab and for the vocab to know which string is displayed in the document it has to know its hash value with the lexeme. So generally, the vocab only looks at the hash value and the context based on the doc but the context is not inherited by the lexeme, only the hash so that it can do conversions on the lower level.

![Lexeme in Doc](https://course.spacy.io/vocab_stringstore.png)

Here is a sample exercise from the course.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I have a cat")

# Look up the hash for the word "cat"
cat_hash = nlp.vocab.strings['cat']
print(cat_hash)

# Look up the cat_hash to get the string
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)
```

We would want to take a look at the hash ID and string value for `cat`. Notice that we can use the variable for cat_hash instead of the actual numeric hash. The result for this would be:

```text
5439657043933447811
cat
```

Do note that we *CAN* use the doc to call out the hash and the string. But to pass the unit test in the code we need to use nlp. Interchanging nlp or doc should provide the same result. So below is another exercise regarding the checking of 

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("David Bowie is a PERSON")

# Look up the hash for the string label "PERSON"
person_hash = nlp.vocab.strings['PERSON']
print(person_hash)

# Look up the person_hash to get the string
person_string = nlp.vocab.strings[person_hash]
print(person_string)
```

```text
5439657043933447811
cat
```

Now that we have covered the vocabulary we can move to the most important data structure in Spacy: **Doc**. Doc is a central data structure in Spacy. It is automatically created when we pass a text through the nlp, for example: `nlp('This is a text')`. We can also create a *Doc* manually. To create it manually we have to import the `Doc` class from the `spacy.token`. In the example below we have created a `Doc` class for `doc`. The `Doc` class will take three arguments: `vocab` which is the shared vocab, `words` for the word list and `spaces` which is a list that indicate if the corresponding word is followed by a space.

```python
# Create an NLP Object
from spacy.lang.en import English
nlp = English()

# Import the Doc class
from spacy.token import Doc

# Words and Spaces that is used to create the Doc
words = ['Hello', 'world', '!']
spaces = [True, False, False]

# Create the Doc manually
doc = Doc[nlp.vocab, words, spaces]
```

Within the `Doc` would be individual tokens. A slice of this `Doc`, composed of token/s, is called a `span`. A span takes, at least, three arguments: the `doc` it is referring to, the `start` index in the specific `doc` and the `end` index of the `doc` where the span ends. As with any Python slice, the `end` index is exclusive.

![Spacy Courses SPAN](https://course.spacy.io/span_indices.png)

To create the span we need to import the `span` class from `spacy.tokens`, the same way we did for `Doc`. After importing the `span` class we can instatiate it by providing the `doc`, `start` and `end` of the span. To create a span with a label we would need to lookup the string in the string store. From there we can add it as a `label` argument. We can also add a span to the `entities` list. The `doc.ents` is writable so we can add our created spans with label to the list by overwriting the `doc.ents` list.

```python
# Import Span and Doc
from spacy.tokens import Doc, Span

# Words and Spaces that is used to create the Doc
words = ['Hello', 'world', '!']
spaces = [True, False, False]

# Create the Doc manually
doc = Doc[nlp.vocab, words, spaces]

# Create the Span manually
span = Span[doc, 0, 2]

# Create a Span with label
span_with_label = Span[doc, 0, 2, label = 'GREETING']

# Add Span with labe to entities
doc.ents = [span_with_label]
```

Here are some best practices for `span` and `doc`. First is that `doc` and `span` are very powerful and hold references and relationships of words and sentences. It is good practice to covert the result to strings as late as possible. Doing it early might lose the relationships between the tokens. If manipulating `doc` or `span` it is better to use *token attributes* when available, for example `token.i` for token index. Also, do not forget to pass in the shared vocabulary. Now we can proceed with some exercises on `doc` and `span`. In the first example we are going to manually create a `doc` class.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "spaCy is cool!"
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
```

To create the `doc` object we first need to import `Doc` class from `spacy.tokens`. Once that is imported we have to define the `words` and `spaces` that would be in the `doc`. To instantiate the `doc` we have to call in `Doc` and provide the three arguments which are `nlp.vocab` for the shared vocab, `words` for the words in the doc, and `spaces` to indicate if there is a space following the word.

```text
# Output
spaCy is cool!
```

For the second exercices we have to provide the value of the spaces for the expression `Go, get started!`.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Go, get started!"
words = ["Go", ",", "get", "started", "!"]
spaces = [False, True, True, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
```

Referring to the desired text we can break down the desired value for the spaces. For the word `Go` there is space since it is followed by `,`. For `,` we need to add a space. We also need to add spaces for `get`. `started` does not need a space since it will have `!` succeeding it and `!` will not need a space since its going to be the last token. The result for the code above would be:

```text
# Output
Go, get started!
```

For the third exercise we are going to define the `words` and `spaces` list based on the desired text which is `Oh, really?!`. Breaking it down we have the code:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Import the Doc class
from spacy.tokens import Doc

# Desired text: "Oh, really?!"
words = ['Oh', ',', 'really', '?', '!']
spaces = [False, True, False, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)
```

Take note that we have to be careful to include all the punctuations when we are going to define the `words` for the `doc`. It goes without saying that we need the `len` of `words` to match `spaces`. The output for the code above would be:

```text
# Output
Oh, really?!
```

Now we move on to creating Docs, Spans and Entities. First up we do a basic creation of a `doc`, `span` and add the span to `entities`.

```python
from spacy.lang.en import English

nlp = English()

# Import the Doc and Span classes
from spacy.tokens import Doc, Span

words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]

# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words, spaces)
print(doc.text)

# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label='PERSON')
print(span.text, span.label_)

# Add the span to the doc's entities
doc.ents = [span]

# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])
```

Something to take note of here woudl be that the `end` argument for `span` is *exclusive*. In this case, even though there are only 4 words and a max index of 3 we have to enter the `end` argument as 4 so that we get the desired span. Also, we cannot `-1` as the end attribute. If it was declared similar to a slice it might be possible `doc[start:-1]` but if it is written in argument `-1` is not allowed.

```text
I like David Bowie
David Bowie PERSON
[('David Bowie', 'PERSON')]
✔ Perfect! Creating spaCy's objects manually and modifying the entities
will come in handy later when you're writing your own information extraction
pipelines.
```

The text block above is the result after running the code block. Notice the note that was left. Spacy objects are going to be handy when creating information extraction pipelines. Note also that we get the tuple for the `entities` and it has attributes `ent.text` and `ent.label_` which follow the same guideline that attributes which are expected to be string values have an underscore succeeding it. Now we go to code debugging as part of learning best practices. For the first exercise we are considering the following code:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Berlin is a nice city")

# Get all tokens and part-of-speech tags
token_texts = [token.text for token in doc]
pos_tags = [token.pos_ for token in doc]

for index, pos in enumerate(pos_tags):
    # Check if the current token is a proper noun
    if pos == "PROPN":
        # Check if the next token is a verb
        if pos_tags[index + 1] == "VERB":
            result = token_texts[index]
            print("Found proper noun before a verb:", result)
```

For this case `result` used the actual strings list via `token.text`. Remember that we would want to use token attributes when manipulating and only call out the string value at the last possible moment. The modified code below is a better implementation of the same logic as the code above. In this case, instead of getting a list of `text` and `pos_` we iterate over the `tokens` in the `doc`. We check the `pos_` of the current token if its a `PROPN` and we also check the `pos_` of the next token if its a `VERB` to get the `text` value of our `result`.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Berlin is a nice city")

# Get all tokens and part-of-speech tags

for token in doc:
    # Check if the current token is a proper noun
    if token.pos_ == "PROPN":
        # Check if the next token is a verb
        if doc[token.i + 1].pos_ == "VERB":
            result = token.text
            print("Found proper noun before a verb:", result)
```

Now we know how to manually create `docs`, `spans` and `ents`. We also know how to create simple rules for finding a specific token/span using the best practices for `span` and `doc`. We can now proceed with **Word Vectors** and **Semantic Similarity**. In this module we will be learning how to predict the similarity between docs, spans or tokens. We will also learn about the use of word vectors and taking advantage of them in our NLP application.

Its now 4:00 PM, I am resuming the topic on *Word Vectors* and *semantic similarity* in Spacy. One of Spacy's builtin feature is that it can predict how similar two objects are. For example we want to check the similarity between `doc A` and `doc B`, spacy can provide us a value for their similarity that is between 0 and 1. This is done with the help of word vectors. In Spacy, *word vectors* are packaged together with the model. It is important to note that word vecotrs are only available on *medium* and *large* sized models. So we cannot do a similarity check between objects when we load a small model. We can call on the `similarity` function via `.similarity`. We can check for similarity between `Doc`, `Span` and `Token` objects. This would mean we can use `Doc.similarity`, `Span.similarity` and `Token.similarity`.

```python
# Load a spacy model with word token
nlp = spacy.load('en_core_web_md')

# Define the two documents to compare.
doc1 = nlp("I like fastfood")
doc2 = nlp("I like pizza")

print(doc1.similarity(doc2))

# Output
0.8627204117787385
```

The code above is a breakdown of the steps we need to do so that we can get a doc similarity. We first need to load a model that has a word vector, in this case we used `en_core_web_md` as our model. Then we pass the text we want to compare to `nlp` so that it becomes a spacy object. To actually get the value for the similarity between the text we simply have to use `doc1.similarity(doc2)`. In this case we are getting the similarity of `doc2` with `doc1` as our reference. It should be bi-directional since this is a shared vocab and word vector but it was not implicitly stated in the notes so further test would be needed here. Do note also that the score for similarity was pretty high at **86%**. Based on the sample text they are almost identical except for the last part. What we do know is that `pizza` is close to a possible defenition of `fastfood` so it scores faily high in the similarity.

Below is another example of using `similiarity`. In this case we are going to compare similarity between tokens. See the code below. Again its the same thing, we want similarity. The only difference here is that we are just going to compare individual tokens from the same text.

```python
# Define the text
doc = nlp["I like pizza and pasta"]
# Compare the two tokens from the doc
token1 = doc[2] # Pizza
token2 = doc[4] # Pasta

print(token1.similarity(token2))

# Output
0.7369546
```

Next we have an example of comparing a document to a token and vice versa. The example code is seen below. As expected the similarity is quite low. This is to be expected since the token object would realatively be disimilar to the doc object.

```python
# Define the doc and token
doc = nlp('I like pizza')
token = nlp('soap')[0] # [0] being used to define it as a span

print(doc.similarity(token))

# Output
0.32531983166759537
```

Then we have the final example for this one where we compare a span to the entire doc. The sample text are in the context of fast food and from the results it looks like the model can figure out that pizza, pasta and Macdonalds or burgers have a high relationship degree in this context.

```python
# Compare Span to Doc
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("Macdonalds sells burgers")

print(span.similarity(doc))

# Output
0.619909235817623
```

In Spacy, similarity is determined by using word vectors. Word vectors are mutli-dimmensional (but compact) representations of words. The one spacy uses is similar to how **Word2Vec** is generated where vectors are trained on raw text. Vectors can be added to an initial statistical model. By default Spacy's similarity is using **cosine similarity** although it can be adjusted depending on the use. In *Word2Vec* the vector is a representative of the word in the vocabulary and is confined to provide the value for individual words. As we have seen, Spacy can create a vector for `Doc` and `Span` objects aside from the `Token` object. The value of the vector for `Doc` and `Span` are, by default, the average of their token vectors. We can observe this by checking the value of shorter phrases with relevant words. It is usually higher compared to long words that have a lot of irrelevant words. To better visualize what these word vectors look like we have an example.

```python
# Load a Spacy model that has word vectors (Medium in this case)
nlp = spacy.load('en_core_web_md')

# Create the sample Doc object
doc = nlp("I have a banana")

# Print the word vector via token.vector attribute
print(doc[3].vector)
```

The code above is the basic step by step flow of how we can check the word vector values on Spacy. First, we have to load a model that has word vectors on it so any medium or large sized model will do. Then we simply have to call on the `.vector` attribute for any `token`, `doc` or `span` that we would like to know the word vector of. The output of the word vector for `banana` or `doc[3].vector` is seen below. Do note that the word vector is a 300-dimmensional vector representation of a single word. If you are familiar already with NLP then you understand that this is already a compact-sized representation of a word.

```text
 [2.02280000e-01,  -7.66180009e-02,   3.70319992e-01,
  3.28450017e-02,  -4.19569999e-01,   7.20689967e-02,
 -3.74760002e-01,   5.74599989e-02,  -1.24009997e-02,
  5.29489994e-01,  -5.23800015e-01,  -1.97710007e-01,
 -3.41470003e-01,   5.33169985e-01,  -2.53309999e-02,
  1.73800007e-01,   1.67720005e-01,   8.39839995e-01,
  5.51070012e-02,   1.05470002e-01,   3.78719985e-01,
  2.42750004e-01,   1.47449998e-02,   5.59509993e-01,
  1.25210002e-01,  -6.75960004e-01,   3.58420014e-01,
 -4.00279984e-02,   9.59490016e-02,  -5.06900012e-01,
 -8.53179991e-02,   1.79800004e-01,   3.38669986e-01,
  ...
```

Predicting similarity can be useful for many types of applications. Some examples would be: recommendation system which checks the similarity between something you are reading and recommend a possible recommendation that would be similar to what you currently have or duplicate flagging/spam monitoring where messages labeled as Spam will get detected by checking the similarity prediction of the messages to a sample spam message.

Objectively, there is no one defenition of similarity. It will depend on how we are going to structure the model and how we use similarity in that context. For example we can frame similarity on checking for recommendations based on the similarity of the description of the product. We can guage sentiment by checking the comment and checking its similarity to known positive or negative comments. We can check recurring issues on a PC by checking the similarity betweent the log entries and mapping out the occurance to check for patterns. __"Similarity"__ becomes contextual in this case. To prove our point, check out the example provided below:

```python
nlp = spacy.load('en_core_web_md')

doc1 = nlp('I like cats')
doc2 = nlp('I hate cats')

print(doc1.similarity(doc2))

# Output would be
0.9501447503553421
```

As we can see above, the default spacy model scored the similarity between `doc1` and `doc2` as high with a value of __95.01%__. It makes sense that its high because it is referring to a sentiment about cats. At the same time it can be considered that the value should not be high since they express sentiment for cat that is actually polar opposites.

Now that we have discussed on the vector values and similarity for word vectors we can now reiforce our learnig via some exercises. First exercise would be to load the word vector of a token from a document. For this example we want to know the vector representation of the word `bananas` in the doc `Two bananas in pyjamas`.

```python
import spacy

# Load the en_core_web_md model
nlp = spacy.load('en_core_web_md')

# Process a text
doc = nlp("Two bananas in pyjamas")

# Get the vector for the token "bananas"
bananas_vector = doc[1].vector
print(bananas_vector)
```

The resulting vector would be seen below. Note that it is a 300-dimmensional vector, the dimmensions for the array is 50x6.

```text
[-2.2009e-01 -3.0322e-02 -7.9859e-02 -4.6279e-01 -3.8600e-01  3.6962e-01
 -7.7178e-01 -1.1529e-01  3.3601e-02  5.6573e-01 -2.4001e-01  4.1833e-01
  1.5049e-01  3.5621e-01 -2.1508e-01 -4.2743e-01  8.1400e-02  3.3916e-01
  2.1637e-01  1.4792e-01  4.5811e-01  2.0966e-01 -3.5706e-01  2.3800e-01
  2.7971e-02 -8.4538e-01  4.1917e-01 -3.9181e-01  4.0434e-04 -1.0662e+00
  1.4591e-01  1.4643e-03  5.1277e-01  2.6072e-01  8.3785e-02  3.0340e-01
  1.8579e-01  5.9999e-02 -4.0270e-01  5.0888e-01 -1.1358e-01 -2.8854e-01
 -2.7068e-01  1.1017e-02 -2.2217e-01  6.9076e-01  3.6459e-02  3.0394e-01
  5.6989e-02  2.2733e-01 -9.9473e-02  1.5165e-01  1.3540e-01 -2.4965e-01
  9.8078e-01 -8.0492e-01  1.9326e-01  3.1128e-01  5.5390e-02 -4.2423e-01
 -1.4082e-02  1.2708e-01  1.8868e-01  5.9777e-02 -2.2215e-01 -8.3950e-01
  9.1987e-02  1.0180e-01 -3.1299e-01  5.5083e-01 -3.0717e-01  4.4201e-01
  1.2666e-01  3.7643e-01  3.2333e-01  9.5673e-02  2.5083e-01 -6.4049e-02
  4.2143e-01 -1.9375e-01  3.8026e-01  7.0883e-03 -2.0371e-01  1.5402e-01
 -3.7877e-03 -2.9396e-01  9.6518e-01  2.0068e-01 -5.6572e-01 -2.2581e-01
  3.2251e-01 -3.4634e-01  2.7064e-01 -2.0687e-01 -4.7229e-01  3.1704e-01
 -3.4665e-01 -2.5188e-01 -1.1201e-01 -3.3937e-01  3.1518e-01 -3.2221e-01
 -2.4530e-01 -7.1571e-02 -4.3971e-01 -1.2070e+00  3.3365e-01 -5.8208e-02
  8.0899e-01  4.2335e-01  3.8678e-01 -6.0797e-01 -7.3760e-01 -2.0547e-01
 -1.7499e-01 -3.7842e-03  2.1930e-01 -5.2486e-02  3.4869e-01  4.3852e-01
 -3.4471e-01  2.8910e-01  7.2554e-02 -4.8625e-01 -3.8390e-01 -4.4760e-01
  4.3278e-01 -2.7128e-03 -9.0067e-01 -3.0819e-02 -3.8630e-01 -8.0798e-02
 -1.6243e-01  2.8830e-01 -2.6349e-01  1.7628e-01  3.5958e-01  5.7672e-01
 -5.4624e-01  3.8555e-02 -2.0182e+00  3.2916e-01  3.4672e-01  1.5398e-01
 -4.3446e-01 -4.1428e-02 -6.9588e-02  5.1513e-01 -1.3489e-01 -5.7239e-02
  4.9241e-01  1.8643e-01  3.8596e-01 -3.7329e-02 -5.4216e-01 -1.8152e-01
  4.3110e-01 -4.6967e-01  6.6801e-02  5.0323e-01 -2.4059e-01  3.6742e-01
  2.9300e-01 -8.7883e-02 -4.7940e-01 -4.3431e-02 -2.6137e-01 -6.2658e-01
  1.1446e-01  2.7682e-01  3.4800e-01  5.0018e-01  1.4269e-01 -3.3545e-01
 -3.9712e-01 -3.3121e-01 -3.4434e-01 -4.1627e-01 -3.5707e-03 -6.2350e-01
  3.7794e-01 -1.6765e-01 -4.1954e-01 -3.3134e-01  3.1232e-01 -3.9494e-01
 -4.6921e-03 -4.8884e-01 -2.2059e-02 -2.6174e-01  1.7937e-01  3.6628e-01
  5.8971e-02 -3.5991e-01 -4.4393e-01 -1.1890e-01  3.3487e-01  3.6505e-02
 -3.2788e-01  3.3425e-01 -5.6361e-01 -1.1190e-01  5.3770e-01  2.0311e-01
  1.5110e-01  1.0623e-02  3.3401e-01  4.6084e-01  5.6293e-01 -7.5432e-02
  5.4813e-01  1.9395e-01 -2.6265e-01 -3.1699e-01 -8.1778e-01  5.8169e-02
 -5.7866e-02 -1.1781e-01 -5.8742e-02 -1.4092e-01 -9.9394e-01 -9.4532e-02
  2.3503e-01 -4.9027e-01  8.5832e-01  1.1540e-01 -1.5049e-01  1.9065e-01
 -2.6705e-01  2.5326e-01 -6.7579e-01 -1.0633e-02 -5.5158e-02 -3.1004e-01
 -5.8036e-02 -1.7200e-01  1.3298e-01 -3.2899e-01 -7.5481e-02  2.9425e-02
 -3.2949e-01 -1.8691e-01 -9.5323e-01 -3.5468e-01 -3.3162e-01  5.6441e-02
  2.1790e-02  1.7182e-01 -4.4267e-01  6.9765e-01 -2.6876e-01  1.1659e-01
 -1.6584e-01  3.8296e-01  2.9109e-01  3.6318e-01  3.6961e-01  1.6305e-01
  1.8152e-01  2.2453e-01  3.9866e-02 -3.7607e-02 -3.6089e-01  7.0818e-02
 -2.1509e-01  3.6551e-01 -5.1603e-01 -5.8102e-03 -4.8320e-01 -2.5068e-01
 -5.2062e-02 -2.0828e-01  2.9060e-01  2.2084e-02 -6.8123e-01  4.2063e-01
  9.5973e-02  8.1720e-01 -1.5241e-01  6.2994e-01  2.6449e-01 -1.3516e-01
  3.2450e-01  3.0503e-01  1.2357e-01  1.5107e-01  2.8327e-01 -3.3838e-01
  4.6106e-02 -1.2361e-01  1.4516e-01 -2.7947e-02  2.6231e-02 -5.9591e-01
 -4.4183e-01  7.8440e-01 -3.4375e-02 -1.3928e+00  3.5248e-01  6.5220e-01]
```

Now that we have loaded the vector values of a token, we will go further and check the similarity of `docs`, `spans` and `tokens` which make use of the word vector under the hood. In the first example we want to print out the similarity between 2 documents.

```python
import spacy

nlp = spacy.load("en_core_web_md")

doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")

# Get the similarity of doc1 and doc2
similarity = doc1.similarity(doc2)
print(similarity)

# Output
0.8789265574516525
```

As we can see the resulting similarity between doc1 `It's a warm summer day` and doc2 `It's sunny outside` is fairly high. For our next example we would be checking for the similarity between two tokens.

```python
import spacy

nlp = spacy.load("en_core_web_md")

doc = nlp("TV and books")
token1, token2 = doc[0], doc[2]

# Get the similarity of the tokens "TV" and "books"
similarity = token1.similarity(token2)
print(similarity)

# Output
0.22325331
```

In the example above we are checking for the similarity between two tokens from the same document. `token1` which corresponds to `TV` and `token2` which corresponds to `books`. The resulting similarity is 0.22325331 which could make sense since TV would be quite disimilar to books. For the final exercise in word vector and similarity we want to compare `spans` from the same `document`.

```python
import spacy

nlp = spacy.load("en_core_web_md")

doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")

# Create spans for "great restaurant" and "really nice bar"
span1 = doc[3:5]
span2 = doc[12:15]

# Get the similarity of the spans
similarity = span2.similarity(span1)
print(similarity)

# Output
0.75173926
```

The predicted similarity between `span1` and `span2` is 75.17% which is quite high. On a final note, we have to understand that similarities would not *always* be conclusive. If we are going to be developing NLP applications that leverage semantic similarity we would want to consider training vectors on our own data or modify the similarity algorithm. This would be similar to __transfer learning__ where we just use the loaded model as reference and train it further on our own dataset.

Now we move on to a combination of statistical models with rule-based systems. This would be one of the strongest tool we can learn in our NLP toolbox and it should be exciting to implement it on Spacy. First off we need to differentiate statistical models and rule-based models and get an idea on their uses. First is statistical models. Statistical models are useful for applications where our model would benefit from generalizing from a small example of data. An example would be detecting a product or peron's name. Instead of creating a look-up table for reference of all available names we can leverage the ability of the model to detect a span as an entity. This is the same functionality that allows Spacy to create POS tags and dependencies. Obviously it cannot maintain a large vocab just for subjects or nouns so it makes use of a statsitical model. To create a statistical model we could be using a combination of entity recognizer, dependency parser and/or part-of-spech tagger.

Now we can talk about rule-based system. Generally, rule-based system works if it is easier for us to maintain a list of values which we want our model to detect. For example would be, list of countries or list of drugs or list of our product. Rule-based systems can be considered "specific" when compared to statistical models which are more "generic". To achieve a rule-based system in Spacy we can use a custom tokenization rules and the matcher and pattern matcher functionalities.

Just a quick recap on rule-based matcher. Our `Matcher` is imported from `spacy.matcher`. We load the shared vocab to the `Matcher`. We then define the `pattern` we want to detect, which are basically a list with dictionary entries of key-value pairs. We then call on `matcher` to the `doc`.

```python
# Initialize with the shared vocab
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Patterns are lists of dictionaries describing the tokens
pattern = [{'LEMMA': 'love', 'POS': 'VERB'}, {'LOWER': 'cats'}]
matcher.add('LOVE_CATS', None, pattern)

# Operators can specify how often a token should be matched
pattern = [{'TEXT': 'very', 'OP': '+'}, {'TEXT': 'happy'}]

# Calling matcher on doc returns list of (match_id, start, end) tuples
doc = nlp("I love cats and I'm very very happy")
matches = matcher(doc)
```

Now that we have reviewed on `Matcher` we can dive a little deeper. In the example below we want to find matches for `golden retriever`. So we added a rule in our dog with a tag `DOG`, a callback of `None` and a pattern looking for a lower  `golden` and a text with lower `retriever`. The `matcher` results as explained before have tuples that indicate the `match_id`, the `start` and `end` of the span where the match was detected. We can see in the `for-loop` section of the code below that we can check for the `span.root` which decides the category of the `span`. In thi example the span is `golden retriever` and the root would be `retriever` which is a dog breed. We can have the `head` of the `span.root` which is the synctatic "parent" of the span which in this case is the verb `have`. We can also look at the previous token from the span simply by defining the index before the span. In this case the previous token was `a` and its `pos_` tag is a `DET`

```python
matcher = Matcher(nlp.vocab)
matcher.add('DOG', None, [{'LOWER': 'golden'}, {'LOWER': 'retriever'}])
doc = nlp("I have a Golden Retriever")

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print('Matched span:', span.text)
    # Get the span's root token and root head token
    print('Root token:', span.root.text)
    print('Root head token:', span.root.head.text)
    # Get the previous token and its POS tag
    print('Previous token:', doc[start - 1].text, doc[start - 1].pos_)
```

```text
# Results
Matched span: Golden Retriever
Root token: Retriever
Root head token: have
Previous token: a DET
```

We are now going to learn about `PhraseMatcher`. This is a tool that will find the sequences of words in the entire data. It is similar to keyword search or regular expressions but it allows you access to the tokens in context as they are matched. It is more efficient and faster than `Matcher` and is great for finding a large list of words. The `PhraseMatcher` is almost the same as a `Matcher` with slight variations on how it is called. Like the `Matcher` the `PhraseMatcher` is also imported from `spacy.matcher` class. It will also require a shared vocabulary similar to `Matcher`. One of the main difference in `PhraseMatcher` is that instead of a `list` of dictionary values we pass in a `Doc` object for the pattern. Once we have added the `pattern` to the matcher we can pass our `Doc` text to the matcher. The same with the `Matcher` we also get the `match_id`, `start` and `end` of the matched span. A code example for `PhraseMatcher` is given below.

```python
# Import Phrase Matcher
from spacy.matcher import PhraseMatcher

# Load the shared dictionary

matcher = PhraseMatcher(nlp.vocab)

# Define the pattern and add it to the matcher
pattern = nlp('Golden Rertriever')
matcher.add('DOG', None, pattern)

# Create a sample text
doc = nlp("I have a Golden Retriever")

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Get the matched span
    span = doc[start:end]
    print('Matched Span: ', span.text

# Output
Matched span: Golden Retriever
```

Now that we have a review of `Matcher` and an introduction to `PhraseMatcher` we can do some exercises. Consider the code block below, why would the pattern not result a match for `Silicon Valley` in the `doc`?

```python
pattern = [{'LOWER': 'silicon'}, {'TEXT': ' '}, {'LOWER': 'valley'}]

doc = nlp("Can Silicon Valley workers rein in big tech from within?")
```

The reason for the lack of match is that the tokenizer does not create tokens for whitespaces. There would be no token to match `{'TEXT' : ' '}`.

The next debugging exercise would be pretty challenging. The scenario is that the patterns below will contain mistakes and not match as expected. We have to fix them so that proper behaviour is acheived.

```python
# ORIGINAL and incorrect code
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)

# Create the match patterns
pattern1 = [{"LOWER": "Amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
pattern2 = [{"LOWER": "ad-free"}, {"POS": "NOUN"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", None, pattern1)
matcher.add("PATTERN2", None, pattern2)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Print pattern string name and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)
```

This is a more challenging debug and it does not help that Docker takes a while to spoool up. Only to provide the incorrect results. So I have to refer to `HINTS` for this code. First up would be to check the actual breakdown of the `doc` file.

```python
for token in doc:
    print(token.i , token.text)
```

The code above is going to print line by line the token contents of the entire `doc` and show the `index` and `text`. What I was aiming to do in this step was check the breakdown of the span `ad-free veiwing`.

```text
26 :
27 ad
28 -
29 free
30 viewing
31 .
```

As it turns out, Spacy is considering `ad`, `-`, `free` and `viewing` as individual tokens. This makes the pattern in `pattern2` invalid since the pattern is trying to find `ad-free` as one single token. To resolve this I created the pattern for pattern2 to be `pattern2 = [{"LOWER": "ad"},{"IS_PUNCT": True},{"LOWER":'free'}, {"POS":"NOUN"}]`. This way the matcher will look for a span with a pattern consisting of 4 tokens instead of the 2-token pattern provided originally. Do note that I used `{"IS_PUNCT": True}` for the `-` token. You can also use `{"LOWER": '-'}`  or `{"TEXT": '-'}`. I tried using the `LOWER` key and it did provide the same results *BUT* using `TEXT` key provided an undesirable effect. It did find the match but printing it only got me `ad` instead of the entire span `ad-free viewing` which is the objective. I am not sure if it was due to my versioning since I was using my Google Colab notebook but the author stated that using `TEXT` is a valid solution and based on the results when running the docker program with the course it did provide the desired results so it might be something to do with the versioning in Colab.

The only issue with `pattern1` that I can see is the use of `{"LOWER":'Amazon'}`. I think that forcing the token to `lower` and then matching it to `Amazon` which has an upper-case will never result in a match. My solution for `pattern1` would be `pattern1 = [{"LOWER": 'amazon'}, {"IS_TITLE":True, "POS":"PROPN"}]`. This way  So my solution for the debugging code exercise is:

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)

# Create the match patterns
pattern1 = [{"LOWER": 'amazon'}, {"IS_TITLE":True, "POS":"PROPN"}]
pattern2 = [{"LOWER": "ad"},{"TEXT":'-'},{"LOWER":'free'}, {"POS":"NOUN"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", None, pattern1)
matcher.add("PATTERN2", None, pattern2)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Print pattern string name and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)
```

This should result in:

```text
PATTERN1 Amazon Prime
PATTERN2 ad-free viewing
PATTERN1 Amazon Prime
PATTERN2 ad-free viewing
PATTERN2 ad-free viewing
PATTERN2 ad-free viewing
```

In any case, the main point of exercise is to show how challenging it could be to get the actual desired results for a pattern match. There could be multiple combinations of patterns we can use to get a result but that also provides possiblities that other results could be fetched. This is where `PhraseMatcher` comes in. So to see the power of `PhraseMatcher` we look at the following exercise:

```python
import json
from spacy.lang.en import English

with open("exercises/countries.json") as f:
    COUNTRIES = json.loads(f.read())

nlp = English()
doc = nlp("Czech Republic may help Slovakia protect its airspace")

# Import the PhraseMatcher and initialize it
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

# Create pattern Doc objects and add them to the matcher
# This is the faster version of: [nlp(country) for country in COUNTRIES]
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", None, *patterns)

# Call the matcher on the test document and print the result
matches = matcher(doc)
print([doc[start:end] for match_id, start, end in matches])
```

In the code above we load a list of countries in a `json` file. This has been provided by the author so that we don't have to create a list. We focus more on the creation of a `PhraseMatcher` which is pretty straightforward. We first import `PhraseMatcher` from `spacy.matcher`. We do not have to code the pattern since it has been provided but take a look at the note from the author. `patterns = list(nlp.pipe(COUNTRIES))` created the list of countries that are `nlp` objects via pipe. The patterns are then added to the matcher and all that is left is for us to run the `doc` file to the `matcher`. This is a more efficient approach that regular matching and it is more powerfull than using `re` regular expressions.

```python
# OUTPUT
[Czech Republic, Slovakia]
```

On the final exercise for the lesson we have to create `span` matches and label them with `GPE`. We add the matched span to the `doc.ents`. We are already provided the list of `COUNTRIES` and the `TEXT` we would be scanning. There are a few things we have to accomplish in the exercise that was left out. First is that we have to load the text file as a `Doc` object. To do this we just had to call `doc=nlp(TEXT)`. This would automatically link up the `nlp` to the `country_text.txt` file that was loaded. Next up is we have to define our `Span` for the matches. To solve this we have to define our span, `span = Span(doc, start, end, label = "GPE")`. Recall that to define a `span` we need to point it to the `doc` that we have, provide the `start` and `end` index and for this case we have to provide it with a label `label ="GPE"`. Next up would be to *UPDATE* the `doc.ents` entries. Prior to this we have been simply overwriting the `doc.ents` entries but for this one we need to `append` the `span` to the list of `doc.ents`. To do this we have `doc.ents = list(doc.ents) + [span]`. What happens is that the original `doc.ents` is transformed into a list and then we append our latest `span` match to that list creating a new and updated `doc.ents`. Next task would be to get the root head token for the `span` which is basically the token that was being referred to by the span match. To do this we just have to define the attribute and assign it to the variable: `span_head_root = span.root.head`. For this example it resolves the root word which provides the category for the entire span.

```python
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import json

with open("exercises/countries.json") as f:
    COUNTRIES = json.loads(f.read())
with open("exercises/country_text.txt") as f:
    TEXT = f.read()

nlp = English()
matcher = PhraseMatcher(nlp.vocab)
patterns = list(nlp.pipe(COUNTRIES))
matcher.add("COUNTRY", None, *patterns)

# Create a doc and find matches in it
doc = nlp(TEXT)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Create a Span with the label for "GPE"
    span = Span(doc, start, end, label="GPE")

    # Overwrite the doc.ents and add the span
    doc.ents = list(doc.ents) + [span]

    # Get the span's root head token
    span_root_head = ____.____.____
    # Print the text of the span root's head token and the span text
    print(span_root_head.____, "-->", span.text)

# Print the entities in the document
print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])
```

The result for the code block above would be seen below. We see the `span.root.head` text value and its corresponding span. For the most part, single token spans have the same `span.root.head` values. The only difference would be the multiple token spans like `Sierra Leone` which has a root of `Sierra` or `South Africa` which has a root head of `South`. This is a bit odd, having `South` as the `root.head` or the parent for `South Africa` although I might have mixed up the defenitions. Additionally, we have a printout of the results for the `doc.ents` entries where our new `GPE` labeled entities are now added.

```python
# OUTPUT
Namibia --> Namibia
South --> South Africa
Cambodia --> Cambodia
Kuwait --> Kuwait
Somalia --> Somalia
Haiti --> Haiti
Mozambique --> Mozambique
Somalia --> Somalia
Rwanda --> Rwanda
Singapore --> Singapore
Sierra --> Sierra Leone
Afghanistan --> Afghanistan
Iraq --> Iraq
Sudan --> Sudan
Congo --> Congo
Haiti --> Haiti
[('Namibia', 'GPE'), ('South Africa', 'GPE'), ('Cambodia', 'GPE'), ('Kuwait', 'GPE'), ('Somalia', 'GPE'), ('Haiti', 'GPE'), ('Mozambique', 'GPE'), ('Somalia', 'GPE'), ('Rwanda', 'GPE'), ('Singapore', 'GPE'), ('Sierra Leone', 'GPE'), ('Afghanistan', 'GPE'), ('Iraq', 'GPE'), ('Sudan', 'GPE'), ('Congo', 'GPE'), ('Haiti', 'GPE')]
```

With this we end Lesson 2 and we now proceed with Lesson 3: Processing Pipelines. We will be discussing what a Spacy processing pipeline is, what goes on under the hood when we create a pipeline and process a text, how to write our own pipeline components and them to the pipeline and finally, how to use custom attributes to add our own metadata to documents, spans and tokens. There would be 16 chapters again. For now I'll be pausing the lesson and resume working on this tomorrow.

Resuming Chapter 3 of the course. We are going to be learning what is acutally happening under the nlp object. First is that the `tokenizer` is applied to convert the string of text into a `Doc` object. After passing the `tokenizer` it goes through a series of pipeline components. Inside this pipeline would be the `tagger`, the `parser` and the entity recognizer. Basically, every attribute of a token is retrieved via the pipeline. Once all the `token` in the `Doc` it is returned along with the attributes.

![Spacy Processing Pipelines](https://course.spacy.io/pipeline.png)

By default, Spacy comes with a built-in pipeline. First is the `tagger` which is the Part-of-speech tagger. The tagger creates `Token.tag`. Next is the `parser` which is the dependency parser. The parser takes care of `Token.dep`, `Token.head`, `Doc.sents`, `Doc.noun_chunks`. It is also responsible for detecting sentences and noun chunks. Next is `ner` which is the named entity recognizer. The named entity recognizer takes care of `Doc.ents`, `Token.ent_iob`, `Token.ent_type`. The entity recognizer also taskes care of the entity type attribues which would indicate if the token is part of an entity. Finally, we have `txtcat` which is the text classifier. It creates `Doc.cats` and sets category label to apply to the whole text. While the text classifier is powerful it does not come with the text categories. The categories would have to be trained first when we create it.

The `pipeline` details are contained in the `meta.json` file. The file defines the order of the pipeline. All models created and loaded into spacy includes several files and a `meta.json` file. The meta file contains the properties like language and pipeline. It is refered to by Spacy when instatiating the components. The built-in components that make predictions would also need binary data (weights). This data is included in the model package. It is included when loading the model.

The names of the components in the pipeline are in `nlp.pipe_names`. To see the list of components and component function as a tuple we just can call `nlp.pipeline`. The component functions are the function that is applied to the `Doc` while passing the component in the pipeline.

Now that we have defined what the pipeline is and the default components are in it we can proceed with exercices. We start of by a simple task of loading a model and checking the components that comes with it. In this case we will be checking pipeline components of `en_core_web_sm` model.

```python
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Print the names of the pipeline components
print(nlp.pipe_names)

# Print the full pipeline of (name, component) tuples
print(nlp.pipeline)
```

Below is the output for the exercise. As we can see the contents pre-packaged with the small model in spacy contains only three components. We have `tagger`, `parser` and `ner`.

```python
# Output

['tagger', 'parser', 'ner']
[('tagger', <spacy.pipeline.pipes.Tagger object at 0x7f033212ab00>), ('parser', <spacy.pipeline.pipes.DependencyParser object at 0x7f031c1a0ca8>), ('ner', <spacy.pipeline.pipes.EntityRecognizer object at 0x7f031c1a0d08>)]
```

We can now move on to adding custom pipeline components. This allows us to add custom functions or processing to the tokens in the `nlp` pipeline. For example we can modify the `Doc` and remove `html` tags or add more data to the `Doc`. In the image below we get an idea of what the pipeline is. The pipeline encompases the processes that the `Text` goes through in the `nlp` to create the `Doc`. Once we have added our customized function to the pipeline it will automatically get included in the processing of a text once we pass it through `nlp` again.

The pipeline is basically a set of components that takes a doc, modifies it and outputs it as a `doc` object. The pipeline components are individual functions that can be called and processes the input into the defined function of the component and passes it along to the next component in the pipeline.

![Spacy Pipelie-Custom](https://course.spacy.io/pipeline.png)

Now that we know what the pipeline components are we can now move on to creating a custom one. To create a custom component we have to first define it as a function and add the new custom component via `nlp.add_pipe`. `nlp.add_pipe` takes at least one argument which is the component function. We have a boiler plate code below.

```python
# Create the custom component function
def custom_component(doc): # We need to pass the doc as input to the custom_component
    # Do the custom processing in this part
    return doc

# Add the custom_component to the pipeline
nlp.add_pipe(custom_component)
```

Additional arguments for the `nlp.add_pipe` would be the positioning of the custom component to be added. We can choose to position our component at the `first`, `last` (default value), `before` or `after` a specific component.

```python
nlp.add_pipe(component, last=True) # Default setting, component is appended to the pipeline
nlp.add_pipe(component, first=True) # Sets the component to the head of the pipeline
nlp.add_pipe(component, before='<existing component name>') # Places the component before the existing component
nlp.add_pipe(component, after='<existing component name>') # Places the component after the existing component
```

Let us try writing an example code for a custom component to the pipeline.

```python
# Load the small english model
nlp = spacy.load('en_core_web_sm')

# Create the custom component. Its doing to print the length of the doc
def custom_component(doc):
    # print the doc's length
    print('Doc Length', len(doc))
    # return the doc object
    return doc

# Add the custom component to the pipeline. New component is to be the first component in the pipeline
nlp.pipe_add(custom_component, first=True)

# Print the pipeline component names (Verification)
print('Pipeline', nlp.pipe_names)
```

We will not go over the entire code above, just the portions regarding the component. First would be the creation of the `custom_component`. It is important that the input of the component is the `doc` and it returns the `doc` after processing. This way we ensure that the component will pass the `doc` once its done processing it. Once we have defined the process that happens to the doc inside the component we can add that component to the pipeline. In this case the `custom_component` is required to be the first value (just for the example). The resulting `pipeline` would be seen below. Notice that the `custom_component` is at the head of the pipeline and is followed by the components that come together with the `en_core_web_sm` model.

```text
Pipeline: ['custom_component', 'tagger', 'parser', 'ner']
```

The next example is applying the new pipeline to the text using the `nlp` object. Normally, the default components will "quietly" process the `doc` file. In this case the `custom_component` is designed to print out the `len` of the doc so we expect to get an output after passing the doc to the nlp.

```python
# Create nlp object with en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# Define the custom component
def custom_component(doc):
    print('Doc length:', len(doc))
    return doc

# Add the new component to the pipeline
nlp.add_pipe(custom_component, first = True)

# Process a sample text
doc = nlp('Hello World!')

#Output
Doc length: 3
```

Now that we know how to create a custom component and add it to the pipeline we can discuss the use cases for custom components. One possible use case would be creating custom values based on tokens and their attributes which is a form of feature engineering. In this use case we can add another attribute to the doc for example if its a rare word or not. Another possible use case would be to locating named entities based on a dictionary. What custom components cannot do is being used to add additional language support. Prior to the custom component being created the language of the model should already be defined.

We now move to exercises on Single components. For this example we will be completing a component function for the `doc`'s length. We will add that new component to the pipeline. Finally, we will test out the new pipeline on a sample text "This is a sentence.".

```python
import spacy

# Define the custom component
def length_component(doc):
    # Get the doc's length
    doc_length = len(doc)
    print("This document is {} tokens long.".format(doc_length))
    # Return the doc
    return doc


# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Add the component first in the pipeline and print the pipe names
nlp.add_pipe(length_component, first=True)
print(nlp.pipe_names)

# Process a text
doc = nlp('This is a sentence.')
```

The exercise above is straight forward since we have been doing that example before. The result after running the code above would be:

```text
['length_component', 'tagger', 'parser', 'ner']
This document is 5 tokens long.
```

We now move on to more complex components. In this exercise we will need to create a custom component that will use `PhraseMatcher` to find animal names in the document. It will add matched spans to the `doc.ents`. Add the new component to the pipeline taking a position after the `ner` component. Finally it should process the text and provide the resulting entity text and labels for the entries in `doc.ents`.

```python
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal_patterns:", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", None, *animal_patterns)

# Define the custom component
def animal_component(doc):
    # Apply the matcher to the doc
    matches = matcher(doc)
    # Create a Span for each match and assign the label 'ANIMAL'
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    # Overwrite the doc.ents with the matched spans
    doc.ents = spans
    return doc
# Add the component to the pipeline after the 'ner' component
nlp.add_pipe(animal_component, after='ner')
print(nlp.pipe_names)

# Process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])
```

The code above is the solution for the desired output. We are recalling `Span` and `PhraseMatcher` for this exercise on top of practicing more complex components. We first created the `pattern` that will be used by our `PhraseMatcher`. We add `PhraseMatcher` to our matcher object. We then define our custom component `animal_component`. It simply matches the doc to the `PhraseMatcher`. We add the custom component to our pipeline making sure to set it after `ner`. We then test out the pipeline to a sample text. The result of the code above is seen below. It would output the `animal_patterns` list, the components of the pipeline and finally the matches in the `doc.ents`.

```python
# Output
animal_patterns: [Golden Retriever, cat, turtle, Rattus norvegicus]
['tagger', 'parser', 'ner', 'animal_component']
[('cat', 'ANIMAL'), ('Golden Retriever', 'ANIMAL')]
```

We now go to Extension attributes. We will learn how to add custom attributes to the Doc, Token and Span objects to store custom data. Custom attributes would allow us to add meta data to Docs, Tokens and Spans. The new attributes can be added once and it is computed dynamically.

```python
# To add a custom metadata via ._
doc._.title = 'My document'
token._.is_color = True
span._.has_color = False

# Registering the new properties to global

# Import Global Classes
from spacy.tokens import Doc, Token, Span

# Set extentsions via the set_extension method
Doc.set_extension('title', default = None)
Token.set_extension('is_color', default = False)
Span.set_extension('has_color', default = False)
```

Once the extensions are added we need to define the default values for them. We can however overwrite tem. There are three types of extensions: attribute extensions, property extensions and method extensions.

```python
from spacy.token import Token

# Set an extension with default value
Token.set_extension('is_color', default = False)

# New doc to be processed
doc = nlp('The sky is blue')

# Overwriting an extenstion attribute value
doc[3]._.is_color = True # We change the value of attribute `is_color` to True for blue.
```

Property extensions are similar to properties in Python. They can define a getter function and an optional setter. An example of a getter function is seen below. The `getter` is going to check if the token is in the list of `colors` we have defined. Once we have defined the `getter` we can set is as an extension to the Token. We can now check the value of the attribute by checking `doc[index]._.is_color`. Do note that the `getter` is only called when you retrieve the attribute value, so the attribute is not stored while the attribute is not called. In this case, the `getter` will only check the attribute when we called `doc[3]._.is_color`.

```python
from spacy.tokens import Token

# Define getter function
def get_is_color(token):
    colors = ['red','yellow','blue']
    return token.text in colors

# Set the extension on the tokken with a getter
Token.set_extension('is_color', getter = get_is_color)

doc = nlp('The sky is blue')
print('{} - {}'.format(doc[3]._.is_color,doc[3].text))

# Output
True - blue
```

Now we move on to `Span` extensions. One important note for `Span` extensions is that they should almost always use a `getter`. Without the `getter` we would need to update the every possible span manually. See the example below for a `Span` extension

```python
from spacy.tokens import Span

# Define getter function
def get_has_color(span):
    colors = ['red','yellow','blue']
    return any(token.text in colors for token in span) # Will check the tokens in the span if it is in the list of colors and will return True otherwise False

# Set the extension
Span.set_extension('has_color', getter = get_has_color)

# Sample results
doc = nlp('The sky is blue')
print('{} - {}'.format(doc[1:4]._.has_color, doc[1:4].text))
print('{} - {}'.format(doc[0:2]._.has_color, doc[0:2].text))
```

The output of the code above is in the text block below. The `getter` function for the `Span.extension` will be checking for the tokens in the defined span and check if there is a match in the colors list.

```text
True - sky is blue
False - The sky
```

Now we discuss the last extension which is the Method extension. This type of extension makes the extension attribute a callable method. It is assigned to a function that becomes available as an object method. This will they let us pass arguments to the extension function.

```python
from spacy.tokens import Doc

# Define the method with arguments
def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc

# Set extension on the Doc object with the method defined
Doc.set_extension('has_token', method = has_token)

doc = nlp('The sky is blue.')
print('{} - {}'.format(doc._.has_token('blue'), `blue`))
print('{} - {}'.format(doc._.has_token('cloud'), `cloud`))
```

As we can see, method extension allows as to call methods to a text which in turn allows as to pass an argument. In this example, we create an extension that allows us to check if the token argument we have is in the tokens making up the Doc. The result for the code block above would be:

```text
True - blue
False - cloud
```

Now that we have discussed the three main types of extensions, property, attribute and method, we can proceed with the exercises. For the first exercise we want to practice setting up some extension attributes.

```python
from spacy.lang.en import English
from spacy.tokens import Token

nlp = English()

# Register the Token extension attribute 'is_country' with the default value False
Token.set_extension('is_country', default=False)

# Process the text and set the is_country attribute to True for the token "Spain"
doc = nlp("I live in Spain.")
doc[3]._.is_country = True

# Print the token text and the is_country attribute for all tokens
print([(token.text, token._.is_country) for token in doc])

# Output
[('I', False), ('live', False), ('in', False), ('Spain', True), ('.', False)]
```

So a breakdown of the code above. First is that we set an attribute extension for `is_country` which has a default value of `False`. We then pass on the sample text we have to the nlp pipeline and we manually change the `is_country` attribute for `Spain` to `True`. Up next we have an exercise for creating a reversed extension and print the values for each.

```python
from spacy.lang.en import English
from spacy.tokens import Token

nlp = English()

# Define the getter function that takes a token and returns its reversed text
def get_reversed(token):
    return token.text[::-1]


# Register the Token property extension 'reversed' with the getter get_reversed
Token.set_extension('reversed', getter=get_reversed)

# Process the text and print the reversed attribute for each token
doc = nlp("All generalizations are false, including this one.")
for token in doc:
    print("reversed:", token._.reversed)
```

In the next example we are going to deal with more complex attributes with getters and method extensions. The objective in the exercise is to check if the tokens inside the doc has a number in it.

```python
from spacy.lang.en import English
from spacy.tokens import Doc

nlp = English()

# Define the getter function
def get_has_number(doc):
    # Return if any of the tokens in the doc return True for token.like_num
    return any(token.like_num for token in doc)


# Register the Doc property extension 'has_number' with the getter get_has_number
Doc.set_extension('has_number', getter=get_has_number)

# Process the text and check the custom has_number attribute
doc = nlp("The museum closed for five years in 2012.")
print("has_number:", doc._.has_number)

# Output
has_number: True
```

The next exercise is about wrapping a span of text into an HTML tag. We then apply this method extension to a span in the document. We see below that the method will wrap the span with an HTML tag formatted with the tag name we input, in this case `strong`.

```python
from spacy.lang.en import English
from spacy.tokens import Span

nlp = English()

# Define the method
def to_html(span, tag):
    # Wrap the span text in a HTML tag and return it
    return "<{tag}>{text}</{tag}>".format(tag=tag, text=span.text)


# Register the Span property extension 'to_html' with the method to_html
Span.set_extension('to_html', method=to_html)

# Process the text and call the to_html method on the span with the tag name 'strong'
doc = nlp("Hello world, this is a sentence.")
span = doc[0:2]
print(span._.to_html('strong'))

# Output
<strong>Hello world</strong>
```

Up next we combine custom extenstion attributes with the model's predictions and create an attribute getter that would return a Wikipedia search URL if the span is an entity like a person, organization or location.

```python
import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")


def get_wikipedia_url(span):
    # Get a Wikipedia URL if the span has one of the labels
    if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
        entity_text = span.text.replace(" ", "_")
        return "https://en.wikipedia.org/w/index.php?search=" + entity_text


# Set the Span extension wikipedia_url using get getter get_wikipedia_url
Span.set_extension('wikipedia_url', getter=get_wikipedia_url)

doc = nlp(
    "In over fifty years from his very first recordings right through to his "
    "last album, David Bowie was at the vanguard of contemporary culture."
)
for ent in doc.ents:
    # Print the text and Wikipedia URL of the entity
    print(ent.text, ent._.wikipedia_url)

# Output
over fifty years None
first None
David Bowie https://en.wikipedia.org/w/index.php?search=David_Bowie
```

The exercise above was a challenging one and I had to look at the hints. It turns out I was using the wrong extension name that's why I was not passing the unit test. I named the extension `get_wikipedia_url` instead of the instruction `wikipedia_url`. Quite silly really. In any case the breakdown of the code is the fun part. First up is the defenition of the `getter` function for the attribute. The code has been mostly filled in but the idea behind it is that we want to check the label of the *entities* detected in the doc and see if it is recognized as one of the labels on the list. We had to use `.label_` and not `.text` for this one since we want to check the label of the entity and not the text of the entity. The `entity_text` variable would be creating a string of all tokens in the span and replace all the whitespace with `_` making it ready for use in the URL to Wikipedia. As we can see from the output there were three entities detected in the doc but only one of had the label that matched our list. In the example its `David Bowie`. Also, the fun part is that the link is actually working.

Now we move to the final exercise of the extensions module. In this one we are combining an extension with a pipeline component and a `PhraseMatcher`. We are now slowly creating more complex pipelines. In the exercise below we would be looking at the text and check for countries based on the `PhraseMatcher`. We will update our `doc.ents` based on the mathces of the `PhraseMatcher`. Once have defined the `countries_component` we then add it to the `nlp.pipeline`. We also create a `getter` function for an extension we would be creating that will retrieve the capitals of the `span` that was detected as countries. We use the getter to set a Span extension named `capital`. The goal of the code would be to check for the `PhraseMatcher` hits from the sample text, tag them under the `GPE` (geopolitical entity label) for `doc.ents` and retrieve the `capital` of the matched `span` by using the attribute extension that can lookup to the json file with the list of corresponding capitals.

```python
import json
from spacy.lang.en import English
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher

with open("exercises/countries.json") as f:
    COUNTRIES = json.loads(f.read())

with open("exercises/capitals.json") as f:
    CAPITALS = json.loads(f.read())

nlp = English()
matcher = PhraseMatcher(nlp.vocab)
matcher.add("COUNTRY", None, *list(nlp.pipe(COUNTRIES)))


def countries_component(doc):
    # Create an entity Span with the label 'GPE' for all matches
    matches = matcher(doc)
    doc.ents = [Span(doc, start, end, label='GPE') for match_id, start, end in matches]
    return doc


# Add the component to the pipeline
nlp.add_pipe(countries_component)
print(nlp.pipe_names)

# Getter that looks up the span text in the dictionary of country capitals
get_capital = lambda span: CAPITALS.get(span.text)

# Register the Span extension attribute 'capital' with the getter get_capital
Span.set_extension('capital', getter=get_capital)

# Process the text and print the entity text, label and capital attributes
doc = nlp("Czech Republic may help Slovakia protect its airspace")
print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])
```

The result of the code block above should be a printout of the current pipeline components and the tuple of `ent` detected on the text. The tuple would contain the `ent.text` which is the entity detected as a country, an `ent.label_` attribute which should reflect as `GPE` since the detected entities should be countries and finally the `ent._.capital` attribute result which is the capital for the entity/country basing on the look up table.

```python
# OUTPUT
['countries_component']
[('Czech Republic', 'GPE', 'Prague'), ('Slovakia', 'GPE', 'Bratislava')]
```