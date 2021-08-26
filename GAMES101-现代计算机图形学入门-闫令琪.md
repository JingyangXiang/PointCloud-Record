<font face="Times New Roman">

# [GAMES101-现代计算机图形学入门-闫令琪](https://www.bilibili.com/video/BV1X7411F744)

[课程主页](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
[推荐书籍](GAMES101.pdf)
## 20210825
### 图形学的应用
1. 游戏
2. 电影
3. 动漫、模拟
4. 设计（CAD），汽车、装修
5. 可视化
6. 虚拟现实（VR）、增强现实（AR）
7. 数字设备（PS）
8. 仿真（Simulation）
9. 字体（点阵、尺量）
### 为什么要学图形学
* 基本的挑战 
  * 虚拟世界和真实世界的交互
  * 对真实世界的理解
  * 新的显示方法
* 技术的挑战
  * 数学：曲线、投影、表面
  * 物理的光照、着色、锐度
  * 3D物体的操作
  * 仿真
* **计算机图形学很酷炫**

### 课程的内容
1. 光栅化<br>
    把三维空间的几何形体显示在屏幕上（实时的计算机图形学的应用），每秒钟30帧
2. 曲线图和曲面<br>
    如何表示曲线和曲面，如何用简单的曲面通过细分的方法得到复杂的曲面，如何发生变化时继续保持，如何保持拓扑结构
3. 光线追踪<br>
    生成质量更高的画面，但是慢（trade off）
4. 仿真/动画<br>
    比如一个球掉到地上

### 课程不说什么
1. OpenGL/DirectX/Vulcan<br>
    主要是要理解为什么要这么做，理解工作原理
2. 不会涉及计算机视觉（一切需要猜测的）
3. 计算机视觉是计算机理解图像，图形学的计算机生成图像

## 20210826 第二课 线性代数回顾
### 图形学的依赖
* 基础数学<br>
  * 线性代数，微积分，统计学
* 基础物理<br>
  * 光学，力学
* 杂项<br>
  * 信号处理
  * 数值分析
* 美学<br>

### 线性代数
* 矩阵、向量等等
* 向量的点乘<br>
  * 找到两个方向的夹角
  * 一个方向在另一个方向的投影
* 向量的叉乘<br>
  * $a\times b=-b\times a$，具体遵循右手定则
  * $||a\times b ||=||a||||b||sin\theta$
  * 叉乘以后的得到向量和原有的向量都垂直<br>
  * 可以用来判断左和右，可以判断内和外（比如判断一个点是不是在三角形的内部还是外部）


### 矩阵
#### 图形学变换
##### 向量的点乘和叉乘都可以转化成矩阵乘法的形式
* 点乘<br>
  $
  \overrightarrow{a}\overrightarrow{b}=\overrightarrow a^T\overrightarrow b=
  \left (\begin{matrix}
    x_a,y_a,z_a
    \end {matrix}
  \right )
  \left(\begin{matrix}
    x_b\\y_b\\z_b
    \end {matrix}
  \right )=  \left(\begin{matrix}
    x_ax_b+y_ay_b+z_az_b
    \end {matrix}
  \right )
  $
* 叉乘<br> 
$
  \overrightarrow{a} \times \overrightarrow{b}=A^*b=
  \left (\begin{matrix}
    0&-z_a&y_a\\
    z_a&0&-x_a\\
    -y_a&x_a&0
    \end {matrix}
  \right )
  \left(\begin{matrix}
    x_b\\y_b\\z_b
    \end {matrix}
  \right )
$

## 20210827 第三课 变换
* 为什么学习变换<br>
  * 模型
  * 视图
* 2维变换：旋转、缩放、切片
* 齐次坐标
* 变换的组合
* 三维变换



### 


</font>