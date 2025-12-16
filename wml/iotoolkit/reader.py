import json

class JsonReader:
    def __call__(self,file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            data_str = f.read()
            json_data = json.loads(data_str)
            return json_data