import json
import xmlrpc.client
import requests
# import jieba.posseg as pseg
# import OpenHowNet
# hownet_dict_advanced = OpenHowNet.HowNetDict(init_babel=True)
# import synonyms

# text = "#钉钉分屏#求求大家 告诉我这个图是真的吗？ 我好慌 我每次都是分屏他们都说可以查到分屏和后台"
# pos_result = pseg.cut(text)
# for word, pos in pos_result:
#     print(word, pos)
#
# syns = synonyms.nearby("好")  # ['好', '不好', '坏', '漂亮', '好看', '用心', '听话', '糟', '踏实', '差劲']
#
# syns = hownet_dict_advanced.get_synset('好')
# print(syns)
# exit(0)

content = ["【", "#", "东北虎", "闯", "民宅", "吓死", "三条", "狗", "#", "】", "1", "月", "11", "日", "，", "黑龙江", "胜利", "农场", "居民", "李界", "坤家", "，", "三只", "小狗", "不见", "了", "。", "狗", "没", "找到", "，", "他", "却", "发现", "了", "一串", "可疑", "的", "足迹", "，", "疑似", "成年", "东北虎", "的", "脚印", "。", "后来", "他", "看到", "了", "小狗", "的", "尸体", "，", "怀疑", "三只", "狗", "是", "被", "吓死", "的", "，", "因为", "前一天", "晚上", "听", "它们", "叫", "了", "几声", "，", "后来", "就", "没音", "了", "。", " ", "哈尔滨", " ", "新闻", "夜航", "的", "微博", "视频", " "]
content = ''.join(content)
# 获取类别
url="http://hz.newsverify.com:9090/getCategory"
headers={'Content-Type': 'application/json;charset=UTF-8', 'Connection': 'close'}
request_param={"content": content}
response=requests.post(url, data=json.dumps(request_param), headers=headers)
print(response, response.json())
category = response.json()["category"]

# 判断是否是谣言
client = xmlrpc.client.ServerProxy("http://www.newsverify.com:8018/")
print(client.test_connection())
pred = float(client.predict(content, category))
# pred > 0.5 真
# pred <= 0.5 假
print("probability:", pred)
