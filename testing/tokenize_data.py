import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np

tokenizer = tf_text.UnicodeCharTokenizer()
x = tokenizer.tokenize(["Hello World"])
print(x.to_list())
string = tokenizer.detokenize(x)
print(string)