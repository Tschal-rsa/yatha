# pipeline:
原始数据是693个样本，但是协和那边用新的标准划分了一下类别，所以重新试验
然后我手动分成了Raw_Data_New下的训练/验证集，数量大致为3:1（参考睿哥之前的做法）
具体数量为：
Raw_Data
    train
        mild_blue 94 (-4)
        mo 155 (-5)
        no 170 (-0)
        se 100 (-6)
    val
        mi 31 (-0)
        mo 52 (-0)
        no 57 (-0)
        se 34 (-0)
总数量为693个样本
然后运行 python preprocess_main.py，和之前一样以50s为间隔切割了原有文件，前20s丢弃。结果存在Processed/raw/下面。
但这一步会出现电极不匹配的问题，所以有些文件直接忽略了。
Processed/raw/
    train
        mi 3752
        mo 6182
        no 6956
        se 3891
    val
        mi 1650
        mo 2723
        no 3237
        se 1833
然后运行 python generate_main.py。产生Processed/feature/。之前睿哥有个hight的含义是用高阈值来滤波，效果也没有变好
然后运行 python rfcl_main.py