import random
from loguru import logger


# 准备你自己的数据集，返回为train data list以及valid data list。
# 数据格式：[{"source": source_query， "target": target_query}, ...]

def prepare_data():
    # example
    train_data_list = [{"source": "你今天好吗", "target": "挺好的"}]
    valid_data_list = [{"source": "你好吗", "target": "还行"}]

    return train_data_list, valid_data_list
