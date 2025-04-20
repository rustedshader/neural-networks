import os
import pickle
import numpy as np
# Make sure to use the same Keras import style as when you saved
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

# --- LOADING THE SAVED MODEL AND PREPROCESSING OBJECTS ---

# Define the directory where files were saved
load_dir = 'chatbot_model_files'

# 1. Load the inference models
encoder_model = load_model(os.path.join(load_dir, 'encoder_model.keras'))
decoder_model = load_model(os.path.join(load_dir, 'decoder_model.keras'))
print("Inference models loaded.")

# 2. Load the tokenizers and max lengths
with open(os.path.join(load_dir, 'preprocessing_data.pkl'), 'rb') as f:
    preprocessing_data = pickle.load(f)

input_tokenizer = preprocessing_data['input_tokenizer']
target_tokenizer = preprocessing_data['target_tokenizer']
max_encoder_seq_length = preprocessing_data['max_encoder_seq_length']
max_decoder_seq_length = preprocessing_data['max_decoder_seq_length']
print("Preprocessing data loaded.")

# --- END OF LOADING ---


# --- INFERENCE FUNCTION (Needs the loaded components) ---
# Make sure this function is defined using the loaded variables

def generate_response(input_text):
    # Use the loaded input_tokenizer and max_encoder_seq_length
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')

    # Use the loaded encoder_model
    states_value = encoder_model.predict(input_seq)

    # Use the loaded target_tokenizer
    start_token_id = target_tokenizer.word_index['<start>']
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token_id

    response_ids = []
    stop_condition = False
    # Use the loaded max_decoder_seq_length
    max_response_length = max_decoder_seq_length

    while not stop_condition:
        # Use the loaded decoder_model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0) # Added verbose=0

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # Check if the sampled index is in the vocabulary before appending
        # (though argmax should usually give a valid index from the dense layer output)
        # Append only non-padding tokens (index 0 is usually padding/unknown)
        if sampled_token_index != 0:
             response_ids.append(sampled_token_index)

        # Use the loaded target_tokenizer
        end_token_id = target_tokenizer.word_index['<end>']
        if (sampled_token_index == end_token_id or
            sampled_token_index == 0 or # Stop if padding token is generated
            len(response_ids) >= max_response_length):
            stop_condition = True

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    # Use the loaded target_tokenizer's index_word mapping
    # Filter out the <end> token explicitly if it was appended
    response_text = ' '.join(
        target_tokenizer.index_word.get(idx, '')
        for idx in response_ids if idx != target_tokenizer.word_index.get('<end>', -1)
    )
    return response_text.strip() # Use strip() to remove leading/trailing spaces


# --- USING THE LOADED MODEL ---

# Now you can directly use the generate_response function
test_input = "How are you doing?"
response = generate_response(test_input)
print(f"Input: {test_input}")
print(f"Response: {response}")

test_input = "What's your name?"
response = generate_response(test_input)
print(f"Input: {test_input}")
print(f"Response: {response}")