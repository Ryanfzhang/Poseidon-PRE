import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm
import os

# keys = list(pd.date_range(start="19880501", end="20191231", freq='D'))

# mean, std = [], []
# for key in tqdm(keys):
#     time_str = datetime.strftime(key, '%Y%m%d')
#     year, month, day = time_str[0:4], time_str[4:6], time_str[6:]
#     input = np.load(os.path.join("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms", "{}/{}-{}-{}.npy".format(year, year, month, day)))
#     mean.append(np.nanmean(input, (0,2,3)))
#     std.append(np.nanstd(input, (0,2,3)))

# mean = np.stack(mean, 0)
# mean = np.mean(mean)
# np.save("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms/mean.npy", mean)

# std = np.stack(std, 0)
# std = np.sqrt(np.mean(std**2, 0))
# np.save("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms/std.npy", std)

import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义一个函数来处理每个 key
def process_key(key):
    time_str = datetime.strftime(key, '%Y%m%d')
    year, month, day = time_str[0:4], time_str[4:6], time_str[6:]
    file_path = os.path.join("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms", 
                             "{}/{}-{}-{}.npy".format(year, year, month, day))
    input_data = np.load(file_path)
    mean = np.nanmean(input_data, (2, 3))  # 计算均值
    std = np.nanstd(input_data, (2, 3))   # 计算标准差
    return mean, std

# 主函数
def main(keys):
    mean, std = [], []
    with ThreadPoolExecutor(max_workers=64) as executor:  # 设置线程池的最大线程数
        futures = {executor.submit(process_key, key): key for key in keys}  # 提交任务
        for future in tqdm(as_completed(futures), total=len(keys)):  # 使用 tqdm 显示进度
            try:
                m, s = future.result()  # 获取结果
                mean.append(m)
                std.append(s)
            except Exception as e:
                print(f"Error processing key {futures[future]}: {e}")
    return mean, std

# 示例调用
keys = list(pd.date_range(start="19880501", end="20191231", freq='D'))
mean, std = main(keys)
mean = np.stack(mean, 0)
mean = np.mean(mean, 0)
np.save("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms/mean.npy", mean)

std = np.stack(std, 0)
std = np.sqrt(np.mean(std**2, 0))
np.save("/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms/std.npy", std)

