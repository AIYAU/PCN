## PCN
One-Shot Animal Individual Identification Using Contrastive Learning in Prototypical Networks

## Datasets

所有的数据集都在data目录下，如果你想在训练测试的时候更换动物的种类，请根据动物的名字修改目录`easyfsl/datasets`的`cub.py`文件，根据配置文件的内容进行更改

## QuickStart
论文实验对比方法：
```ssh
python con_train.py
```
PCN对比原型：
```ssh
python train.py
```
## Change models
如果你想更换`con_train.py`中的对比模型进行实验，请根据源代码中的注释将相应对比模型进行替换即可。
## Acknowledgement
