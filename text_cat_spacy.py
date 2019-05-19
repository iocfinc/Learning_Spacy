from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

@plac.annotations(
    model = ('Model name. Defaults to blank "en" model.', 'option', 'm', str),
    output_dir = ('Optional output directory.','option', 'o', Path),
    n_texts = ('Number of texts to train from', 'option', 't', int),
    n_iter = ('Number of iterations during training.','option', 'n', int),
    init_tok2vec = ('Initialize pretrained tok2vec weights','option', 't2v', Path)
)

def main(model = None, output_dir = None, n_iter = 20, n_texts = 2000, init_tok2vec=None):
    if output_dir is not None: # Check that there is an output dir
        output_dir = Path(output_dir) # Point the output_dir to the path.
        if not output_dir.exists(): # Create the new path for output_dir if it does not exists
            output_dir.mkdir()
    else:
        print('output_dir argument was empty.')
        
    if model is not None:
        nlp = spacy.load(model) # Load the existing model if defined
        print('Model "%s" has been loaded.' % model)
    else:
        nlp = spacy.blank('en') # Default to a blank 'en' model if model was not given
        print('Blank "en" model has been created.')
    
    # Adding the text classifier component to the pipeline
    if "textcat" not in nlp.pipe_names: # Checks that the textcat pipeline already exists
        textcat = nlp.create_pipe(
            "textcat",
            config = {
                "exclusive_classes": True,
                "architecture": 'simple_cnn'
            }
        )
        nlp.add_pipe(textcat, last = True) # Add the textcat component to the end of the pipeline
        
        # Initialize the labels for the text classifier.
        # In this case, since its a sentiment analyzer we need it to do Positive or Negative labels
        textcat.add_label('POSITIVE')
        textcat.add_label('NEGATIVE')

        # Load the IMDB Dataset via Thinc
        print('Loading IMDB data via thinc...')
        (train_texts, train_cats), (dev_texts,dev_cats) = load_data()
        # NOTE: dev_texts and dev_cat are the test set.
        train_texts = train_texts[:n_texts] # limit the number of text files to train on [n_texts]
        train_cats = train_cats[:n_texts] # limit the number of categories to train on [n_text]

        print(
            "Using {} examples ({} training, {} evaluation)".format(
                n_texts, len(train_texts), len(dev_texts)
            )
        )
        # Create the training data set
        # The format is (train_texts, [{'categories': train_cats}]) so we just loop ever them
        train_data = list(zip(train_texts,[{'cats':cats} for cats in train_cats]))

        # Get the names of the pipeline components which we would need to disable during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe !="textcat"]
        with nlp.disable_pipes(*other_pipes): # Disable the rest of the components except textcat
            optimizer = nlp.begin_training()
            if init_tok2vec is not None: # Load the pretrained init_tok2vec weights if available
                with init_tok2vec.open('rb') as file_:
                    textcat.model.tok2vec.from_bytest(file_.read())
            print("Training model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}\t".format(
                "ITER","LOSS", "P", "R", "F"))
            batch_sizes = compounding(4.0, 32.0, 1.001)
            for i in range(n_iter):
                losses = {}
                # Preprocessing of data
                # Random shuffle
                random.shuffle(train_data)
                batches = minibatch(train_data, size = batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch) # Unpack the text and cat for the batches
                    nlp.update(texts, annotations, sgd = optimizer, drop = 0.2, losses = losses)
                with textcat.model.use_params(optimizer.averages):
                    # Evaluate the current model score with the dev data we had
                    scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                print(
                    "{0:0.1f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}".format(
                        i,
                        losses['textcat'],
                        scores['textcat_p'],
                        scores['textcat_r'],
                        scores['textcat_f']
                        )
                )
        # Test the newly trained model
        test_text = 'This movie sucked so bad.'
        test_text2 = 'This movie is great and heart warming.'
        
        doc = nlp(test_text)
        print("Sample text {}\nPredicted Sentiment: {}".format(test_text,doc.cats))
        
        if output_dir is not None:
            # Save the new model
            with nlp.use_params(optimizer.averages):
                nlp.to_disk(output_dir)
            print("Saved model to: ", output_dir)
            # Testing the saved model
            print("Model Loading from: ", output_dir)
            nlp2 = spacy.load(output_dir)
            doc2 = nlp2(test_text2)
            print("Sample text {}\nPredicted Sentiment: {}".format(test_text,doc2.cats))
def load_data(limit = 0, split=0.8):
    """Data loader from IMDB dataset"""
    # Split part of the train data for evaluation
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    split = int(len(train_data)*split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0 # True Positives
    fp = 1e-8 # False Positives
    fn = 1e-8 # False Negatives
    tn = 0.0 # True Negatives
    for i, doc in enumerate(textcat.pipe(docs)): # Go over the entire list of docs for evaluation
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0 # Predicted Positive True Positive: TRUE POSITIVE
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0 # Predicted Positive True Negative: FALSE POSITIVE
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1.0 # Predicted Negative True Negative: TRUE NEGATIVE
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1.0 # Predicted Negative True Positive: FALSE NEGATIVE
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall)/(precision + recall)
    return {"textcat_p":precision, "textcat_r":recall,"textcat_f":f_score}

if __name__ == "__main__":
    plac.call(main)