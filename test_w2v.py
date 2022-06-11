from gensim.models.keyedvectors import KeyedVectors

# 下载googlenews-vectors-negative300.bin.gz
# word_vectors = get_data('word2vec')

# 加载原始二进制格式的模型
word_vectors = KeyedVectors.load_word2vec_format('german_wiki_20200501.bin', binary=True)

# 词向量
# 数组中的每个浮点数表示向量的一个维度
print(word_vectors['thomas'])

print(word_vectors.most_similar(positive=['filme'], topn=5))