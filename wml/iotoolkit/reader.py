import json

class JsonReader:
    def __call__(self,file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            data_str = f.read()
            json_data = json.loads(data_str)
            return json_data

class DecryptReader:
    def __init__(self):
        from annotation_crypt import PrpCrypt
        self.pc = PrpCrypt()

    def __call__(self,file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            data_str = f.read()
            try:
                json_data = json.loads(data_str)
                return json_data
            except:
                data_str,is_encrypt = self.pc.decrypt_json(data_str)
                json_data = json.loads(data_str)
                return json_data

class AIOTJsonReader:
    def __init__(self):
        from aiot_platform_trainer import AiLabelDecrypt
        self.aiot = AiLabelDecrypt()

    def __call__(self,filepath):
        return self.aiot.decrypt(filepath)

    