# SYSU 2023级 机器学习大作业

本仓库为使用ReChorus框架对 "[SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation](https://arxiv.org/abs/2405.20878)"[SIGIR'2024]


本项目基于 [ReChorus](https://github.com/THUwangcy/ReChorus) 框架实现。

# 相比ReChorus增加文件Handler.py, SelfGNN.py和Runner.py

* Handler.py: src/helpers/Handler.py
* SelfGNN.py: src/models/SelfGNN.py
* Runner.py: src/helpers/Runner.py

# 环境准备

## 克隆仓库

```bash
git clone https://github.com/tyler55427/SYSU-2023-ReChorus.git
cd SYSU-2023-Rechorus/src
```

## 安装依赖

```bash
conda create -n selfgnn python==3.6.12
conda activate selfgnn
pip install -r requirements.txt
```

# 运行代码

```bash
python main.py --data amazon --reg 1e-2 --lr 1e-3 --temp 0.1 --ssl_reg 1e-6 --save_path amazon --epoch 150  --batch 512 --sslNum 80 --graphNum 5  --pred_num 0 --gnn_layer 3 --test True --att_layer 4 --testSize 1000 --keepRate 0.5 --sampNum 40 --pos_length 200 --regenerate 1
```
