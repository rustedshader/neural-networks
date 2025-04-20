from convokit import Corpus
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, Embedding, Dense
import pickle
import os

# Building a seq2seq model for chatbot
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


# tokenizer.fit_on_texts is used to give priority list of words based on how frequently it appears in sentences.
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

# Hyperparameters.
embedding_dim = 256
latent_dim = 512
vocab_input_size = len(input_tokenizer.word_index) + 1
vocab_target_size = len(target_tokenizer.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(
    input_dim=vocab_input_size, output_dim=embedding_dim, mask_zero=True
)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
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

# Define the training model that takes encoder and decoder inputs and outputs the decoder predictions.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
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
dec_emb = dec_emb_layer(decoder_inputs)  # For training
dec_emb_inference = dec_emb_layer(decoder_inputs_inference)

decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm(
    dec_emb_inference, initial_state=decoder_states_inputs
)

decoder_outputs_inference = decoder_dense(decoder_outputs_inference)

decoder_model = Model(
    [decoder_inputs_inference] + decoder_states_inputs,
    [decoder_outputs_inference] + [state_h_inference, state_c_inference],
)


def generate_response(input_text):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding="post")

    states_value = encoder_model.predict(input_seq)

    start_token_id = target_tokenizer.word_index["<start>"]
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token_id

    response_ids = []
    stop_condition = False
    max_response_length = max_decoder_seq_length

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        if sampled_token_index != 0:
            response_ids.append(sampled_token_index)

        end_token_id = target_tokenizer.word_index["<end>"]
        if (
            sampled_token_index == end_token_id
            or len(response_ids) >= max_response_length
        ):
            stop_condition = True

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    response_text = " ".join(
        target_tokenizer.index_word.get(idx, "")
        for idx in response_ids
        if idx != target_tokenizer.word_index.get("<end>", -1)
    )
    return response_text


print(generate_response("Hello, how are you?"))

save_dir = "chatbot_model_files"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# 1. Save the inference models
encoder_model.save(os.path.join(save_dir, "encoder_model.keras"))
decoder_model.save(os.path.join(save_dir, "decoder_model.keras"))
print("Inference models saved.")

# 2. Save the tokenizers and max lengths using pickle
preprocessing_data = {
    "input_tokenizer": input_tokenizer,
    "target_tokenizer": target_tokenizer,
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
}

with open(os.path.join(save_dir, "preprocessing_data.pkl"), "wb") as f:
    pickle.dump(preprocessing_data, f)
