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