import string

# ----------------------------
# Load Captions File
# ----------------------------
def load_captions(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# ----------------------------
# Parse CSV Captions (image,caption)
# ----------------------------
def parse_captions(text):
    captions = {}
    lines = text.split('\n')

    for line in lines[1:]:  # skip header (image,caption)
        if len(line) < 2:
            continue

        parts = line.split(',', 1)  # split only at first comma
        if len(parts) < 2:
            continue

        image_id, caption = parts

        if image_id not in captions:
            captions[image_id] = []

        captions[image_id].append(caption)

    return captions


# ----------------------------
# Clean Captions
# ----------------------------
def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)

    for key, caption_list in captions.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.split()
            caption = [word for word in caption if len(word) > 1]
            caption = [word for word in caption if word.isalpha()]
            caption_list[i] = ' '.join(caption)


# ----------------------------
# Main Execution
# ----------------------------
captions_text = load_captions("dataset/captions.txt")
captions_dict = parse_captions(captions_text)

print("Total images:", len(captions_dict))

clean_captions(captions_dict)

# Add startseq and endseq
for key, caption_list in captions_dict.items():
    for i in range(len(caption_list)):
        caption_list[i] = "startseq " + caption_list[i] + " endseq"


#  Print only ONE image to check
for key, captions in list(captions_dict.items())[:1]:
    print("\nSample Image:", key)
    for caption in captions:
        print(caption)
        
        
# ----------------------------
# Build Vocabulary
# ----------------------------
def build_vocabulary(captions):
    vocab = set()
    for key in captions:
        for caption in captions[key]:
            for word in caption.split():
                vocab.add(word)
    return vocab

vocab = build_vocabulary(captions_dict)

print("\nVocabulary Size:", len(vocab))


from tensorflow.keras.preprocessing.text import Tokenizer

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([caption for captions in captions_dict.values() for caption in captions])

vocab_size = len(tokenizer.word_index) + 1

print("Tokenizer Vocabulary Size:", vocab_size)


# ----------------------------
# Find Maximum Caption Length
# ----------------------------
max_length = max(len(caption.split()) for captions in captions_dict.values() for caption in captions)

print("Maximum Caption Length:", max_length)


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

# ----------------------------
# Create Training Sequences
# ----------------------------
def create_sequences(tokenizer, max_length, captions_dict, vocab_size):
    X1, X2, y = [], [], []

    for key, captions in captions_dict.items():
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(key)      # image ID (we’ll replace later with features)
                X2.append(in_seq)   # text input
                y.append(out_seq)   # expected next word

    return X1, X2, y


print("\nCreating training sequences...")

X1, X2, y = create_sequences(tokenizer, max_length, captions_dict, vocab_size)

print("Total training samples:", len(X2))