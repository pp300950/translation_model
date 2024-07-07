import json
import numpy as np

def load_translation_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)
    
    translation_dict = translation_data.get('translation_dict', {})
    example_sentences = translation_data.get('example', [])
    
    reverse_translation_dict_en_th = create_reverse_translation_dict(translation_data.get('translation_dict_en_th', {}))

    return translation_dict, example_sentences, reverse_translation_dict_en_th

def create_reverse_translation_dict(translation_dict):
    reverse_translation_dict = {}
    for k, v in translation_dict.items():
        if isinstance(v, list):
            for item in v:
                reverse_translation_dict[item] = k
        else:
            reverse_translation_dict[tuple(v)] = k
    return reverse_translation_dict

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
        
        if isinstance(translated_word, list):
            #เลือกค่าแรกในลิสต์
            translated_word = translated_word[0]  
        
        if isinstance(translated_word, dict):
            #เลือกค่าแรกในพจนานุกรม
            translated_word = list(translated_word.values())[0]  
        
        translated_words.append(translated_word)
    
    translated_sentence = ' '.join(map(str, translated_words))  #แปลงทุกตัวเป็นสตริงก่อนรวมเข้าด้วยกัน
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

def translate_sentence(source_sentence, direction, translation_dict, reverse_translation_dict, reverse_translation_dict_en_th):
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
    elif direction == 'th-en':
        tokenized_source = tokenize_sentence(source_sentence, reverse_translation_dict_en_th)
        return translate_with_attention(tokenized_source.split(), reverse_translation_dict_en_th, weights, context)
    elif direction == 'en-th':  
        tokenized_source = tokenize_sentence(source_sentence, reverse_translation_dict)
        return translate_with_attention(tokenized_source.split(), translation_dict, weights, context)
    else:
        return None
if __name__ == "__main__":
    filename = 'data_train (2.4).json' 
    translation_dict, example_sentences, reverse_translation_dict_en_th = load_translation_data(filename)
    reverse_translation_dict = create_reverse_translation_dict(translation_dict)

    for example in example_sentences:
        chinese_sentence = example.get('zh', '')
        thai_sentence = example.get('th', '')
        english_sentence = example.get('en', '')

        chinese_tokens = chinese_sentence.split()
        thai_tokens = thai_sentence.split()
        english_tokens = english_sentence.split()

        for chinese_word, thai_word, english_word in zip(chinese_tokens, thai_tokens, english_tokens):
            if chinese_word not in translation_dict:
                translation_dict[chinese_word] = thai_word
            if thai_word not in reverse_translation_dict:
                reverse_translation_dict[thai_word] = chinese_word
            if thai_word not in reverse_translation_dict_en_th:
                reverse_translation_dict_en_th[thai_word] = english_word

    while True:
        direction = input("zh-th,th-zh,th-en,en-th: ")
        source_sentence = input("Enter: ")

        translated_sentence = translate_sentence(source_sentence, direction, translation_dict, reverse_translation_dict, reverse_translation_dict_en_th)
        print(f"Source Sentence: {source_sentence}")
        print(f"Translated Sentence: {translated_sentence}")
