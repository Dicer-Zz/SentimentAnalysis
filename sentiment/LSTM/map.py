texts = [
    '今 天 天 气 好 晴 朗',
    '处 处 好 风 光 '
]

max_len = max(map(lambda x: len(x.split()), texts))
print(max_len)