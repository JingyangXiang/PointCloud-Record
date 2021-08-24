<font face="Times New Roman">

# PointCloud-Record
记录点云SemanticKITTI论文阅读记录

## 2021.08.24
### [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821)
[Github](https://github.com/NVIDIA/semantic-segmentation)

纯图像分割方法:可以考虑用在投影图上面，也可以考虑用在多体素分割的上面，用来提升不同大小的物体的分割

<div align=center>
<img src="https://pic4.zhimg.com/80/v2-529271adfb9acf996f9148739ae4393f_720w.jpg">
</div>

```python
# 注意力部分的相关代码
if pred is None:
    pred = cls_out
    aux = aux_out
elif s >= 1.0:
    # downscale previous
    pred = scale_as(pred, cls_out)
    pred = attn_out * cls_out + (1 - attn_out) * pred
    aux = scale_as(aux, cls_out)
    aux = attn_out * aux_out + (1 - attn_out) * aux
else:
    # s < 1.0: upscale current
    cls_out = attn_out * cls_out
    aux_out = attn_out * aux_out

    cls_out = scale_as(cls_out, pred)
    aux_out = scale_as(aux_out, pred)
    attn_out = scale_as(attn_out, pred)

    pred = cls_out + (1 - attn_out) * pred
    aux = aux_out + (1 - attn_out) * aux

# ocrnet.py 产生logit_attn的代码
def _fwd(self, x):
    x_size = x.size()[2:]

    _, _, high_level_features = self.backbone(x)
    cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
    # 产生相关的注意力
    attn = self.scale_attn(ocr_mid_feats)

    aux_out = Upsample(aux_out, x_size)
    cls_out = Upsample(cls_out, x_size)
    attn = Upsample(attn, x_size)

    return {'cls_out': cls_out,
            'aux_out': aux_out,
            'logit_attn': attn}

'''
self.scale_attn 来源    
可以看到是conv+bn+relu+sigmoid的注意力操作   
'''   
def make_attn_head(in_ch, out_ch):
    bot_ch = cfg.MODEL.SEGATTN_BOT_CH
    if cfg.MODEL.MSCALE_OLDARCH:
        return old_make_attn_head(in_ch, bot_ch, out_ch)

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', Norm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    if cfg.MODEL.MSCALE_INNER_3x3:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = Norm2d(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    if cfg.MODEL.MSCALE_DROPOUT:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head
```


#### 主要思路+[解读1](https://blog.csdn.net/m0_47645778/article/details/106279016)+[解读2](https://zhuanlan.zhihu.com/p/142949640)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;输入两个不同分辨率的图像，网络学习不同物体在相应分支上的权重
##### 优点
1. 多尺度分割，目前分割的效果是Cityscapes Dataset SOTA层次的模型

##### 缺点
1. 模型计算量、参数量都十分巨大，很难迁移



### [Polarized Self-Attention: Towards High-quality Pixel-wise Regression](https://arxiv.org/abs/2107.00782)
[Github](https://github.com/DeLightCMU/PSA)
<div align=center>
<img src="https://pic1.zhimg.com/80/v2-b6881ea41fd435f123785c45682b1af4_1440w.jpg">
</div>

``` python
class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()
        '''
            具体请见github代码
        '''
```
#### [主要思路](https://zhuanlan.zhihu.com/p/392148142)
##### 优点
1. **极化滤波**（ Polarized filtering）：在通道和空间维度保持比较高的resolution（在通道上保持C/2的维度，在空间上保持[H,W]的维度 ），这一步能够减少降维度造成的信息损失；
2. **增强**（Enhancement）：组合非线性直接拟合典型细粒度回归的输出分布。
3. 代码比较简单易懂

### [Deep FusionNet for Point Cloud Semantic Segmentation](http://www.feihuzhang.com/papers/ECCV2020-2.pdf)
#### 主要思路+[解读1](https://blog.csdn.net/qq_37109317/article/details/112965128)

<div align=center>
<img src="https://img-blog.csdnimg.cn/2021012116453441.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MTA5MzE3,size_16,color_FFFFFF,t_70#pic_center">
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;左边刻画的是**Neighborhood aggregation**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;右边刻画的是**Inner-voxel aggregation**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;主要是体素融合（依靠稀疏3D卷积spconv）和点云级别的邻点特征聚合（**这里是否可以考虑单个体素的梯度信息，用来刻画物体表面的一个信息，感觉可以比较好的来刻画一个物体**）



#### 算法介绍（下采样）
<div align=center>
<img src="https://img-blog.csdnimg.cn/20210124172004237.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MTA5MzE3,size_16,color_FFFFFF,t_70#pic_center">
</div>

1. 对于每一个体素，去用hash表索引这个点所在体素的身边的体素的点（针对特定的kernel_size，包括自己)
2. 每个点去做空间偏移,即<br>
$\triangle p=(x,y,z)_{point}-(x',y',z')_{voxel}$<br>
3. Feature of Point($F_p$) concat with $\triangle p$ →Fc
4. MaxPooling <br>
$F_{max}=max\{F'_p, p{\in}N(V)\}$
5. $cat(F'_p,F_{max})$→$Fc$→$F^A_p$(更新单点的特征)
#### Point Interpolation（[PointNet](https://arxiv.org/abs/1612.00593)&nbsp;&nbsp;&nbsp;&nbsp;[code](https://github.com/fxia22/pointnet.pytorch)） （上采样，by [PointNet++](https://zhuanlan.zhihu.com/p/44809266)&nbsp;&nbsp;&nbsp;&nbsp;[解读](https://zhuanlan.zhihu.com/p/88238420)&nbsp;&nbsp;&nbsp;&nbsp;[code](https://github.com/erikwijmans/Pointnet2_PyTorch)）

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**论文中描述错误，插值方法应该引用自PointNet++**，总而言之，<font color='red'>**论文PointNet和PointNet++很值得仔细看一看**</font>

<div align=center>
<img src="https://pic1.zhimg.com/v2-8dc76710bd09c25d5c8196d6aff56fec_r.jpg">
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PointNet的代码和结构的图没有差别，主要就是根据Fc去映射特征并且使用最大池化+Fc得到一个[batchsize, kxk]的特征，然后再去+eye(k)最后reshape →[batchsize, k, k]作为一个类似于旋转的刻画矩阵

<div align=center>
<img src="https://pic3.zhimg.com/80/v2-e4ca28cbcd3a802aff9074bace47e056_720w.jpg">
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PointNet++的插值方法，使用CUDA自定义编写的插值，知乎上这么描述：如果将多个这样的处理模块级联组合起来，PointNet++就能像CNN一样从浅层特征得到深层语义特征。**对于分割任务的网络，还需要将下采样后的特征进行上采样，使得原始点云中的每个点都有对应的特征。这个上采样的过程通过最近的k个临近点进行插值计算得到**




#### SemanticKITTI数据集介绍（ECCV2020，介绍的特别好）

Semantic KITTI: The Semantic KITTI dataset is a new large-scale LiDAR
point cloud dataset in driving scenes. It has **22 sequences with 19 valid classes**,
and **each scan contains 10–13k** points in a large space of **160m×160m×20m**. We
use Sequences **0–7 and 9–10 as the training set** (in total 19k frames), **Sequence 8
(4k frames) as the validation set**, and the remaining 11 sequences (20k frames)
as the test set. Different from other point cloud datasets, LiDAR points are
distributed irregularly in a large 3D space. There are many small objects with
only a few points and the points are very sparse in the distance. All these make
it challenging for semantic segmentation of the large-scale LiDAR point clouds.

##### 优点
1. When compared to existing voxel networks, FusionNet can predict
**point-wise** labels and avoid those ambiguous/wrong predictions when **a voxel**
**has points from different classes**（**可以区分更加细粒度的特征：由于点云的嵌入**）.
2.  When compared to the popular PointNets and point-based
convolutions, FusionNet has more effective feature aggregation oper-
ations (including the **efficient** **neighborhood-voxel aggregations** and the fine-
grain **inner-voxel point-level aggregations**). These operations help produce
better accuracy for large-scale point cloud segmentation(**体素聚合依靠稀疏卷积 & 体素内部聚合？？方法是什么**).
3. FusionNet takes full advantage of the **sparsity property** to reduce its memory
footprint. For instance, it can take more than **one million points** in training
and **use only one GPU** to achieve state-of-the-art accuracy（**内存的高效**）.


</font>