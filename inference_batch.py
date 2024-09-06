# 对一个长视频序列的帧进行预测，一次预测100帧

import os
import subprocess

# 设置数据集路径和范围
base_data_path = './dataset/davinci_raofei_inference/'
start = 100
end = 2001
step = 100

# 遍历指定范围
for i in range(start, end, step):
    data_path = os.path.join(base_data_path, str(i))

    # 预处理
    print(f"Processing data for {i}...")
    subprocess.run([
        'python', './preprocessing/save_dino_embed_video.py',
        '--config', './config/preprocessing.yaml',
        '--data-path', data_path
    ])

    # 推理
    print(f"Inferring data for {i}...")
    subprocess.run([
        'python', './inference_grid.py',
        '--config', './config/train.yaml',
        '--data-path', data_path,
        '--use-segm-mask'
    ])

    # 可视化
    print(f"Visualizing data for {i}...")
    subprocess.run([
        'python', 'visualization/visualize_rainbow.py',
        '--data-path', data_path
    ])

print("All tasks completed.")