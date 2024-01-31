# MSA

多模态情感分析

## Setup
代码基于Python3实现，我们需要首先安装下面的依赖库:

- transformers==4.37.1
- torch==2.1.0
- pandas==2.0.3
- scikit-learn==1.3.1
- tqdm==4.66.1
- Pillow==10.0.1
- numpy==1.24.4

```
pip install -r requirements.txt
```
我们还需要将使用的数据集解压到/datasets目录下，即
```
|-- datasets
    |-- data/
    |-- test_without_label.txt
    |-- train.txt
|-- multimodal
|-- run_train_model.py
...
```
如果数据集位于别的位置，我们需要在`run_trian_model.py`中更改传递给`TrainDataset, TestDataset`的数据集路径。
## Repository structure

```
|-- multimodals # 模型实现代码
    |-- ALMT.py/ # ALMT[1]的实现
    |-- base.py/  # 多模态模型的基本实现，朴素连接
    |-- LMF.py/  # Low-rank Multimodal Fusion[2]的实现
    |-- MULT.py # Multimodal Transformer[3]的实现
    |-- processor.py # huggingface格式的多模态数据预处理器，包括BERT分词器和ViT的图像预处理器 
    |-- sadatasets.py # 数据集的读取等操作
    |-- TransformerEncoder.py # 基于PyTorch的CrossModal Transformer和Transformer的实现（用于MULT和ALMT）
    |-- utils.py # 一些工具方法，包括训练，评估和预测   
|-- results # 所有实验结果
|-- predict.txt # 在测试集test_without_label.txt上的标签预测结果
|-- readme.md # 本文件
|-- requirements.txt # 依赖
|-- run_train_model.py # 运行模型代码的主文件
|-- train.sh # 训练用脚本

```
## Run code

你可以通过运行`run_train_model.py`来训练并在验证集上验证模型
```
 python run_train_model.py
```

具体的参数为
```
usage: run_train_model.py [-h] [--model MODEL] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                          [--eval_per_epoch] [--modals MODALS] [--predict]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to run, default almt, almt for ALMT, lmf for LMF, mult for MULT, base for base
                        model(naive concat)
  --epochs EPOCHS       number of epochs to train, default 10
  --batch_size BATCH_SIZE
                        batch size for training, default 32
  --lr LR               learning rate, default 1e-4
  --eval_per_epoch      whether to evaluate on validation dataset per epoch
  --modals MODALS       modals to use, default ti, ti for text and image, i for image only, t for text only
  --predict             whether to predict on test dataset
  --predict_path PREDICT_PATH
                        path to save predict result
```

例如，我想采用5e-4的学习率，16的batch size来训练文本和视觉两个模态的MULT模型，训练epoch数为5，并在每个epoch后在验证集上计算评估指标，同时在测试集上预测标签，并保存到`predict_mult.txt`中，可以运行下面的指令：
```
python run_train_model.py --model mult --lr 5e-4 --batch_size 16 --epochs 5 --eval_per_epoch --modals ti --predict --predict_path "predict_mult.txt"
```
## Attribution

本仓库的代码在实现时有参考下面的内容:

- ALMT: https://github.com/Haoyu-ha/ALMT
- LMF: https://github.com/thuiar/MMSA/blob/master/src/MMSA/models/singleTask/LMF.py
- MULT: https://github.com/yaohungt/Multimodal-Transformer

## Reference

[1] H. Zhang, Y. Wang, G. Yin, K. Liu, Y. Liu, and T. Yu, "Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis," *arXiv preprint arXiv:2310.05804*, 2023.

[2] Z. Liu, Y. Shen, V. B. Lakshminarasimhan, P. P. Liang, A. Zadeh, and L.-P. Morency, "Efficient low-rank multimodal fusion with modality-specific factors," *arXiv preprint arXiv:1806.00064*, 2018.

[3] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, "Multimodal Transformer for Unaligned Multimodal Language Sequences," in *Proceedings of the Conference. Association for Computational Linguistics. Meeting*, vol. 2019, pp. 6558, NIH Public Access, 2019.