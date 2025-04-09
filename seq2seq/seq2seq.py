from convokit import Corpus
from keras._tf_keras.keras.preprocessing.text import Tokenizer

corpus_name = "/Users/shubhang/.convokit/saved-corpora/movie-corpus"
corpus = Corpus(filename=corpus_name) 

pairs = []
for convo in corpus.iter_conversations():
    utts = list(convo.iter_utterances())
    for i in range(len(utts) - 1):
        input_text = utts[i].text
        target_text = utts[i+1].text
        target_text = "<start> " + target_text + " <end>"
        pairs.append((input_text, target_text))

input_text = [inp for inp, _ in pairs]
target_text = [out for _, out in pairs]


input_tokenizer = Tokenizer(input_text)
target_tokenizer = Tokenizer(target_text)


