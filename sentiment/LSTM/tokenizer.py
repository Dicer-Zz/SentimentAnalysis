from tensorflow.keras.preprocessing.text import Tokenizer

texts = [
    '今天 天气 好热',
    '小席 是 一头 主',
    '小王 是 小席 的 爸爸',
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)

seq1= tokenizer.texts_to_sequences(['小王', '是', '爸爸'])
seq2 = tokenizer.texts_to_sequences(['小席', '是', '猪'])
print(seq1, seq2)
print(type(seq1))

seq1= tokenizer.texts_to_sequences(['小王', '是', '妈妈'])
print(seq1)
