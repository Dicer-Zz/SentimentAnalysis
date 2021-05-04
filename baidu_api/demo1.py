from aip import AipNlp

APP_ID = '24096184'
API_Key = 'X14x1ztUO7IDPG6G7FG3KS7V'
Secret_Key = 'G0wLTd4VLuLHqlGPaLilh0m8Gw38oGXl'

client = AipNlp(APP_ID, API_Key, Secret_Key)

text = '有些人，很奇怪，不爱你，也不放过你。而有些人更奇怪，爱你，还放过你。'
print(client.sentimentClassify(text))

text = '我能想到最美好的事，就是你一直都在爱着我。'
print(client.sentimentClassify(text))
