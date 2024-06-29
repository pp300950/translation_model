#โมเดลเริ่มต้น  
#เเปลได้เเค่ ไทย-จีน, จีน-ไทย
import json
import numpy as np

with open('translation.json', 'r', encoding='utf-8') as f:
    translation_data = json.load(f)

translation_dict = translation_data['translation_dict']

# Create reverse translation dictionary
reverse_translation_dict = {}
for k, v in translation_dict.items():
    if isinstance(v, list):
        for item in v:
            reverse_translation_dict[item] = k
    else:
        reverse_translation_dict[v] = k

example_sentences = translation_data['example']

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def attention_weights(decoder_hidden, encoder_outputs):
    scores = np.dot(encoder_outputs, decoder_hidden)
    weights = softmax(scores)
    return weights

def context_vector(attention_weights, encoder_outputs):
    context = np.sum(attention_weights[:, np.newaxis] * encoder_outputs, axis=0)
    return context

def translate_with_attention(source_words, translation_dict, attention_weights, context_vector):
    translated_words = []
    for i, word in enumerate(source_words):
        translation_prob = attention_weights[i] if i < len(attention_weights) else 1.0 / len(source_words)
        translated_word = translation_dict.get(word, '<unk>')
        translated_words.append(translated_word)
    translated_sentence = ' '.join(translated_words)
    return translated_sentence

def tokenize_sentence(sentence, dictionary):
    words = dictionary.keys()
    max_word_length = max(len(word) for word in words)
    sentence_length = len(sentence)
    tokenized_sentence = []
    i = 0
    while i < sentence_length:
        for j in range(min(max_word_length, sentence_length - i), 0, -1):
            word = sentence[i:i+j]
            if word in words:
                tokenized_sentence.append(word)
                i += j
                break
        else:
            tokenized_sentence.append(sentence[i])
            i += 1
    return ' '.join(tokenized_sentence)

def translate_sentence(source_sentence, direction):
    encoder_outputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    decoder_hidden = np.random.rand(encoder_outputs.shape[1])
    weights = attention_weights(decoder_hidden, encoder_outputs)
    context = context_vector(weights, encoder_outputs)

    if direction == 'zh-th':
        tokenized_source = tokenize_sentence(source_sentence, translation_dict)
        return translate_with_attention(tokenized_source.split(), translation_dict, weights, context)
    elif direction == 'th-zh':
        tokenized_source = tokenize_sentence(source_sentence, reverse_translation_dict)
        return translate_with_attention(tokenized_source.split(), reverse_translation_dict, weights, context)
    else:
        raise ValueError("Invalid direction. Use 'zh-th' for Chinese to Thai or 'th-zh' for Thai to Chinese.")

for chinese_sentence, thai_sentence in example_sentences.items():
    chinese_tokens = chinese_sentence.split()
    thai_tokens = thai_sentence.split()

    for chinese_word, thai_word in zip(chinese_tokens, thai_tokens):
        if chinese_word not in translation_dict:
            translation_dict[chinese_word] = thai_word
        if thai_word not in reverse_translation_dict:
            reverse_translation_dict[thai_word] = chinese_word

direction = input("Enter translation direction (zh-th for Chinese to Thai, th-zh for Thai to Chinese): ")
source_sentence = input("Enter the source sentence: ")

translated_sentence = translate_sentence(source_sentence, direction)
print(f"Source Sentence: {source_sentence}")
print(f"Translated Sentence: {translated_sentence}")
