'''
字典
可以理解为Java的对象
'''
neural_network_config = {
    "layer_1": {"units": 64, "activation": "rele"},
    "layer_2": {"units": 128, "activation": "rele"},
}
# 遍历字典
for layer, config in neural_network_config.items():
    print(f"{layer}:{config['units']}")
