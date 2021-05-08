moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}

for label, mood in moods.items():
    print(label, mood)

for label, mood in enumerate(moods):
    print(label, mood)

print(type(moods.items()), type(enumerate(moods)))