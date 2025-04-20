# What Are we Doing?

Making a Seq2Seq Model for a chatbot.


# What does seq2seq Model include ?

a. Encoder
b. Decoder

The Seq2Seq model is made up of an Encoder (left side) that is linked to the Decoder (right side) by the Attention layer which essentially learns to map the encoder input to the decoder output.

The encoder processes the input sequence and converts it into a fixed-length context vector, which is passed to the decoder. The decoder then uses this context vector to generate the output sequence.



- The input sequence is passed through the encoder one element at a time. For each element in the input sequence, the encoder processes it and updates its internal state.
- When the encoder has processed the entire input sequence, it outputs the final internal state as the context vector. This context vector is a fixed-length representation of the input sequence that captures the key information from it.
- The decoder is then fed the context vector and begins generating the output sequence. It does this by predicting the next element in the sequence based on the context vector and the previously generated elements.
- The decoder continues generating the output sequence until it has produced the desired number of elements, or until it generates a special end-of-sequence token.


Seq2seq models are often used in chatbots because they can handle the variable-length input and output that is common in conversations. For example, a chatbot using a seq2seq model might take a user’s question as input, process it using the encoder, and then generate a response using the decoder. The model can handle questions of varying length and complexity, and generate appropriate responses.


Hot topic in 2018