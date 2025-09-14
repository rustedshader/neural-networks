from convokit import Corpus
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, Embedding, Dense
import pickle
import os
import tensorflow as tf
import keras

home = os.environ["HOME"]
corpus_name = f"{home}/.convokit/saved-corpora/movie-corpus"
corpus = Corpus(filename=corpus_name)

pairs = []
for convo in corpus.iter_conversations():
    utts = list(convo.iter_utterances())
    for i in range(len(utts) - 1):
        input_text = utts[i].text
        target_text = utts[i + 1].text
        target_text = "<start> " + target_text + " <end>"
        pairs.append((input_text, target_text))

input_text = [inp for inp, _ in pairs]
target_text = [out for _, out in pairs]


input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()


x = input_tokenizer.fit_on_texts(input_text)
y = target_tokenizer.fit_on_texts(target_text)


input_sequences = input_tokenizer.texts_to_sequences(input_text)
target_sequences = target_tokenizer.texts_to_sequences(target_text)

max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_sequences)

encoder_input_data = pad_sequences(
    input_sequences, maxlen=max_encoder_seq_length, padding="post"
)
decoder_input_data = pad_sequences(
    target_sequences, maxlen=max_decoder_seq_length, padding="post"
)

decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

embedding_dim = 256
latent_dim = 512
vocab_input_size = len(input_tokenizer.word_index) + 1
vocab_target_size = len(target_tokenizer.word_index) + 1


# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html


encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(
    input_dim=vocab_input_size, output_dim=embedding_dim, mask_zero=True
)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb = Embedding(
    input_dim=vocab_target_size, output_dim=embedding_dim, mask_zero=True
)(decoder_inputs)
decoder_lstm = LSTM(
    latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_target_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3,clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

batch_size = 64
epochs = 1
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data[..., np.newaxis],
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_inference = Input(shape=(1,), name="decoder_inputs_inference")
dec_emb_layer = Embedding(
    input_dim=vocab_target_size, output_dim=embedding_dim, mask_zero=True
)
dec_emb = dec_emb_layer(decoder_inputs)  
dec_emb_inference = dec_emb_layer(decoder_inputs_inference)

decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm(
    dec_emb_inference, initial_state=decoder_states_inputs
)

decoder_outputs_inference = decoder_dense(decoder_outputs_inference)

decoder_model = Model(
    [decoder_inputs_inference] + decoder_states_inputs,
    [decoder_outputs_inference] + [state_h_inference, state_c_inference],
)

save_dir = "chatbot_model_files"
os.makedirs(save_dir, exist_ok=True)  

encoder_model.save(os.path.join(save_dir, "encoder_model.keras"))
decoder_model.save(os.path.join(save_dir, "decoder_model.keras"))
print("Inference models saved.")

preprocessing_data = {
    "input_tokenizer": input_tokenizer,
    "target_tokenizer": target_tokenizer,
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
}

with open(os.path.join(save_dir, "preprocessing_data.pkl"), "wb") as f:
    pickle.dump(preprocessing_data, f)
