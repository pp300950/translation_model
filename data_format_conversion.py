import json

data = {
    "england": {
      "英格兰": "ประเทศอังกฤษ"
    },
    "english": {
      "英语": [
        "ภาษาอังกฤษ",
        "อังกฤษ"
      ]
    },
    "thailand": {
      "泰国": "ประเทศไทย"
    },
    "thai": {
      "泰国": [
        "ไทย",
        "ภาษาไทย"
      ]
    },
    "china": {
      "中国": "ประเทศจีน"
    },
    "chinese": {
      "中国": [
        "จีน",
        "ภาษาจีน"
      ]
    },
    "chinese songs": {
      "中文 歌曲": "เพลงจีน"
    },
    "english songs": {
      "英文 歌曲": "เพลงอังกฤษ"
    },
    "thai songs": {
      "泰国 歌曲": "เพลงไทย"
    },
}

#ฟังก์ชันสำหรับการเเปลงข้อมูล
def transform_data(data):
    transformed_data = []
    for en, translations in data.items():
        for zh, th_values in translations.items():
            if isinstance(th_values, list):
                for th in th_values:
                    transformed_data.append({
                        "zh": zh,
                        "th": th,
                        "en": en
                    })
            else:
                transformed_data.append({
                    "zh": zh,
                    "th": th_values,
                    "en": en
                })
    return transformed_data
    
new_data = transform_data(data)
output = json.dumps(new_data, ensure_ascii=False, indent=4)
print(output)
