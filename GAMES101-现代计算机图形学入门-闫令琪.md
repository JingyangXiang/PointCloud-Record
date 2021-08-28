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
  $$
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
  $$
* 叉乘<br> 
$$
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
$$

## 20210827 第三课 变换
* 为什么学习变换<br>
  * 模型
  * 视图
* 2维变换：旋转、缩放、切边
* 齐次坐标
* 变换的组合
* 三维变换

**这堂课基本是机器人课程里面都学过的**

### 二维的变换
#### 缩放
<div align=center>
<img src="https://img-blog.csdn.net/20180324174537227?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NhbHRyaXZlcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="300">
</div>

#### 对称

#### shear（切边）

####  统一的方法

$$
  \left[\begin{matrix}
    x'\\
    y'
    \end {matrix}
  \right ]=
  \left [\begin{matrix}
    a&b\\
    c&d
    \end {matrix}
  \right ]
  \left[\begin{matrix}
    x\\
    y
    \end {matrix}
  \right ]+
  \left [\begin{matrix}
  t_x\\
  t_y
  \end {matrix}
  \right ]
$$
* 转化成齐次表达的形式

$$
  \left[\begin{matrix}
    x'\\
    y'\\
    w'
    \end {matrix}
  \right ]=
  \left [\begin{matrix}
    1&0&t_x\\
    0&1&t_y\\
    0&0&1
    \end {matrix}
  \right ]
  \left[\begin{matrix}
    x\\
    y\\
    1
    \end {matrix}
  \right ]=
  \left [\begin{matrix}
  x+t_x\\
  y+t_y\\
  1
  \end {matrix}
  \right ]
$$


## 20210828 第四课 变换
This lecture may be difficult.
### 旋转矩阵（很好理解）

$$
  R_\theta =
  \left[\begin{matrix}
    cos\theta&-sin\theta\\
    sin\theta&cos\theta
    \end {matrix}
  \right ]
$$

$$
R_{-\theta} =
  \left[\begin{matrix}
    cos\theta&sin\theta\\
    -sin\theta&cos\theta
    \end {matrix}
  \right ]=R_{\theta}^T
$$

$R_{-\theta}=R_{\theta}^{-1}$ <br>

正交矩阵$R$
### 3D Transformation
和2D基本相同

$$
\left(
  \begin{matrix}
    x'\\
    y'\\
    z'\\
    1
  \end{matrix}
  \right)=
\left(
  \begin{matrix}
    a&b&c&t_x\\
    d&e&f&t_y\\
    g&h&i&t_s\\
    0&0&0&1
    \end{matrix}
  \right)
  \cdot
\left(\begin{matrix}
    x\\
    y\\
    z\\
    1
    \end{matrix}
  \right) 
$$
#### Scale

$$
  S(s_x,s_y,s_z)=
  \left(
    \begin{matrix}
      s_x&0&0&0\\
      0&s_y&0&0\\
      0&0&s_z&0\\
      0&0&0&1
      \end{matrix}
    \right)
$$

#### Translatioh
$$
T(t_x,t_y,t_z)=
\left(
  \begin{matrix}
    1&0&0&t_x\\
    0&1&0&t_y\\
    0&0&1&t_z\\
    0&0&0&1
    \end{matrix}
  \right)
$$
#### Rotation around x-axis
$$
R_x(\alpha)=
\left(
  \begin{matrix}
    1&0&0&0\\
    0&cos\alpha&-sin\alpha&0\\
    0&sin\alpha&cos\alpha&0\\
    0&0&0&1
    \end{matrix}
  \right)
$$
#### Rotation around y-axis

**y的结果是反的,因为xy-z 但是zx得到y**

$$
R_y(\alpha)=
\left(
  \begin{matrix}
    cos\alpha&0&sin\alpha&0\\
    0&1&0\\
    -sin\alpha&0&cos\alpha&0\\
    0&0&0&1
    \end{matrix}
  \right)
$$
#### Rotation around z-axis
$$
R_z(\alpha)=
\left(
  \begin{matrix}
    cos\alpha&-sin\alpha&0&0\\
    sin\alpha&cos\alpha&0&0\\
    0&0&1&0\\
    0&0&0&1
    \end{matrix}
  \right)
$$
#### Rodrigues's Rotation Formula（绕着固定轴旋转，轴过原点）
$$
R(n,\alpha)=cos{\alpha}{\textbf{I}}+(1-cos(\alpha)){\textbf{nn}^T}+sin({\alpha})
\left(
  \begin{matrix}
    0&-n_z&n_y\\
    n_z&0&-n_x\\
    -n_y&n_x&0\\
    \end{matrix}
  \right)
$$
### ModelViewing Transformation
- View(视图)/Camera Transformation

  **相机永远在原点**
- **What is view transformation?**
- **Think about how to take a photo（MVP变换）**
  - Find a good place and arrange people (model transformation)
  - Find a good "angle" to put the camera (view transformation)
    - 放在哪里
    - 往哪里看 
    - 向上方向（俯仰角）
  
  约定俗称，相机先放在标准位置（原点，朝着-z方向看，y是向上方向）<br>
  如何把位置为$\overrightarrow{e}$、$\overrightarrow{t}$、$\overrightarrow{g}$的相机移动到标准位置
    - $M_{view}$ in math($M_{view}=R_{view}T_{view}$)
      - Translates to origin（先做平移）
      - Rotates g to -z
      - Rotates t to y
      - Rotates (g$\times$t) to x
      - Difficult to write!<br>

  $$
    R_{view}^{-1}=
    \left[
    \begin{matrix}
    x_{\hat{g}\times\hat{t}}&x_t&x_{-g}&0\\
    y_{\hat{g}\times\hat{t}}&y_t&y_{-g}&0\\
    z_{\hat{g}\times\hat{t}}&z_t&z_{-g}&0\\
    0&0&0&1\\
    \end{matrix}
    \right]
  $$

  **由于旋转矩阵是正交阵**

  $$
    R_{view}^{-1}=
    \left[
    \begin{matrix}
    x_{\hat{g}\times\hat{t}}&y_{\hat{g}\times\hat{t}}&z_{\hat{g}\times\hat{t}}&0\\
    x_t&y_t&z_t&0\\
    x_{-g}&y_{-g}&_{-g}&0\\
    0&0&0&1\\
    \end{matrix}
    \right]
  $$
  
  模型视图变换（MV）$\rightarrow$接下来是投影变换
  - Cheese! (projection transformation)

找到一个好的地方（场景搭好）$\rightarrow$找一个好的角度和位置$\rightarrow$茄子，拍照投影

- **Projection(投影) Transformation**
  - Orthographic(正交) projection
    
    平行的线依然平行,不会带来近大远小<br>
    认为相机无限远,近和远是完全一样的距离
    - A simple way of understanding
      - camera located at origin, looking at -Z, up at Y
      - Drop Z coordinate
      - Translate and scale the resulting rectangle to [-1,1]
    - 
  - Perspective(透视) projection
    - 近平面保持不变
    平行的线不平行,会带来近大远小<br>
    认为摄像机是一个点,看到是一个锥体
    

## 20210829 第五课 

</font>