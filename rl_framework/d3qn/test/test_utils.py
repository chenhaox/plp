"""
Created on 2025/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
from rl_env.utlis import generate_object_classes
for i in range(1000000):
    v = generate_object_classes((20, 20, 10), 5)
    if len(v) != 5:
        print(v)
        raise ValueError("generate_object_classes error")
