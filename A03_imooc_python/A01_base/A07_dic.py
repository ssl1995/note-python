'''
字典
可以理解为Java的对象
'''
neural_network_config = {
    "layer_1": {"units": 64, "activation": "rele1"},
    "layer_2": {"units": 128, "activation": "rele2"},
    "layer_3": {"units": 129, "activation": "rele3"},
}

# 安全获取元素，不存在返回None
un = neural_network_config.get("aaaa")
print(un)
# 不安全获取元素，不存在会保存
# print(neural_network_config["aaaa"])

del neural_network_config["layer_1"]

neural_network_config["layer_4"] =  {"units": 130, "activation": "rele4"}

print("----------------------")
# 遍历字典
for key, value in neural_network_config.items():
    print(f"{key}:{value['units']}")
