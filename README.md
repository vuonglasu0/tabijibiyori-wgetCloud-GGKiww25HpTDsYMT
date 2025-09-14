# 目标检测 | 基于Weiler–Atherton算法的IoU求解

## IoU

**交并比（Intersection over Union, IoU）** 是计算机视觉领域中常用的一个评价指标，尤其在目标检测与图像分割任务中，用于衡量预测结果与真实标注之间的重合程度。

其定义如下：

[![image-20250911114032266](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913155947090-2082000267.png)](https://github.com)

* 如图所示给定任意两个多边形框B1,B2B1,B2（预测框与真实框），其 IoU 的计算公式为：

IoU(B1,B2)=|B1∩B2||B1∪B2|IoU(B1,B2)=|B1∩B2||B1∪B2|

其中|B1∪B2||B1∪B2|为二者并集的面积，|B1∩B2||B1∩B2|为二者交集的面积。

IoU的取值范围为[0,1][0,1]，当IoU=1IoU=1时表示预测框与真实框**完全一致**；当IoU=0IoU=0时表示预测框与真实框**没有任何重叠**；

通过评价IoUIoU可以评估目标检测模型的性能

## 基于Weiler–Atherton算法的IoU求解

### Bounding Box

在目标检测任务中，通常使用 **包围盒（bounding box）** 表示目标的矩形区域。根据任务需求的不同，包围盒可以分为以下几类：

* 轴对齐包围盒（Axis-Aligned Bounding Box，AABB）

  轴对齐包围盒一般应用于2D的目标检测任务，四条边分别与x轴和y轴对齐，可以表达为：

  BAABB=(xc,yc,w,h),w>0,h>0BAABB=(xc,yc,w,h),w>0,h>0

  + 其中(xc,yc)(xc,yc)为中心坐标，w,hw,h分别为包围盒的宽和高，也被称为外延（extent）
* BEV包围盒（Bird’s Eye View Bounding Box，BEV Boudning Box）

  BEV包围盒一般用于自动驾驶任务，在俯视图（BEV）中，每个物体除了位置和尺寸外，还包含一个航向角（yaw）表示方向，可以表达为：

  BBEV=(xc,yc,l,w,θ),l>0,w>0,θ∈[−π,π)BBEV=(xc,yc,l,w,θ),l>0,w>0,θ∈[−π,π)

  + 其中(xc,yc)(xc,yc)为中心坐标，l,wl,w分别为包围盒的长和宽，θθ为全局坐标系的旋转角度
* 3D包围盒（3D Boudning Box）

  3D包围盒在BEV包围盒的基础上增加了高度，也是自动驾驶任务中常用的表示格式

  B3D=(xc,yc,zc,l,w,h,θ),w,l,h>0,θ∈[−π,π)B3D=(xc,yc,zc,l,w,h,θ),w,l,h>0,θ∈[−π,π)

  + 其中(xc,yc,zc)(xc,yc,zc)为中心坐标，l,w,hl,w,h分别为包围盒的长，宽和高，θθ为全局坐标系的旋转角度

在本文中，我们以 **BEV 包围盒** 为例，使用 **Weiler–Atherton 算法**求解 IoU。对于3D包围盒的 IoU 计算，可通过将 BEV 包围盒在俯视平面上的结果拓展到高度方向来实现。

### Corner坐标转换

在计算之前，我们首先需要将多边形从包围盒表示转换为Corner坐标表示（四个顶点的坐标），这个过程可以分为三步，首先给定一个包围盒：

BBEV=(xc,yc,l,w,θ),l>0,w>0,θ∈[−π,π)BBEV=(xc,yc,l,w,θ),l>0,w>0,θ∈[−π,π)

#### 计算局部坐标系下的四个角点

Plocal=⎡⎢
⎢
⎢
⎢
⎢
⎢⎣+l2+w2+l2−w2−l2−w2−l2+w2⎤⎥
⎥
⎥
⎥
⎥
⎥⎦∈R4×2Plocal=[+l2+w2+l2−w2−l2−w2−l2+w2]∈R4×2

#### 绕中心旋转矩阵

Protated=Plocal⋅R(θ)⊤Protated=Plocal⋅R(θ)⊤

其中：

R(θ)=[cosθ−sinθsinθcosθ]R(θ)=[cos⁡θ−sin⁡θsin⁡θcos⁡θ]

#### 平移到全局坐标系

Pglobal=Protated+⎡⎢
⎢
⎢
⎢⎣xcycxcycxcycxcyc⎤⎥
⎥
⎥
⎥⎦Pglobal=Protated+[xcycxcycxcycxcyc]

将上述的三个过程合并为齐次坐标系矩阵：

⎡⎢
⎢
⎢
⎢⎣x1y11x2y21x3y31x4y41⎤⎥
⎥
⎥
⎥⎦=⎡⎢
⎢
⎢
⎢
⎢
⎢⎣+l2+w21+l2−w21−l2−w21−l2+w21⎤⎥
⎥
⎥
⎥
⎥
⎥⎦⋅⎡⎢⎣cosθsinθ0−sinθcosθ0xcyc1⎤⎥⎦[x1y11x2y21x3y31x4y41]=[+l2+w21+l2−w21−l2−w21−l2+w21]⋅[cos⁡θsin⁡θ0−sin⁡θcos⁡θ0xcyc1]

### Weiler–Atherton算法

Weiler–Atherton算法是一种计算任意两个非凹图形交集的算法，可以被分为四个步骤：

* 求解所有相交点
* 求解所有被包围的顶点
* 将相交点和被包围的顶点放入一个数组中，按照逆时针进行排序
* 按照顺序连接为新的多边形求解其面积

给定任意两个包围盒的Cornor坐标表示B1,B2∈R4×2B1,B2∈R4×2

[![image-20250911152406728](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160041420-1146848711.png)](https://github.com)

#### 计算所有相交点

给定任意两条线段L1=(x1,y1),L2=(x2,y2)L1=(x1,y1),L2=(x2,y2)与M1=(x3,y3),M2=(x4,y4)M1=(x3,y3),M2=(x4,y4)，我们可以如下求出其交点：

定义r=L2−L1,s=M2−M1r=L2−L1,s=M2−M1，有：

t=(M1−L1)×sr×s,u=(M1−L1)×rr×st=(M1−L1)×sr×s,u=(M1−L1)×rr×s

其中××为二维向量叉乘，定义如下：

(x1,y1)×(x2,y2)=x1y2−y1x2(x1,y1)×(x2,y2)=x1y2−y1x2

那么线段P,QP,Q的交点为：

Pinsect={L1+tr,if r×s≠0 and t∈[0,1],u∈[0,1]无交点,otherwisePinsect={L1+tr,if r×s≠0 and t∈[0,1],u∈[0,1]无交点,otherwise

通过线段相交算法我们可以求出任意两个线段之间的交点（如图所示的紫色点）

[![image-20250911153300694](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160058724-615682152.png)](https://github.com)

#### 计算所有被对方包围起来的顶点

给定任意一个包围盒B∈R4×2B∈R4×2与点PP，我们可以通过如下过程求解点PP是否在包围盒BB中：

定义Pa=B[0,:],Pb=B[1,:],Pd=B[3,:]Pa=B[0,:],Pb=B[1,:],Pd=B[3,:]

求得：

t=AP⋅AB||AB||2,u=AP⋅AD||AD||2t=AP⋅AB||AB||2,u=AP⋅AD||AD||2

其中AB=Pb−Pa, AD=Pd−Pa, AP=P−PaAB=Pb−Pa, AD=Pd−Pa, AP=P−Pa

如图所示，当t∈[0,1]∧u∈[0,1]t∈[0,1]∧u∈[0,1]时，点PP位于包围盒中BB

[![image-20250912214349621](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160108585-401469849.png)](https://github.com)

通过上述流程我们可以求解所有在对方包围盒的顶点（如图所示的绿色点）

[![image-20250911153246025](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160116473-88922030.png)](https://github.com)

#### 顶点极角排序

为了方便连接每个顶点，接下来我们将所有顶点按照极坐标系下的角度进行排序，给定任意两点P1(x1,y1)P1(x1,y1)与P2(x2,y2)P2(x2,y2)，有比较函数如下：

cmp(P1,P2)={true,ΘP1<ΘP2false,ΘP1≥ΘP2cmp(P1,P2)={true,ΘP1<ΘP2false,ΘP1≥ΘP2

其中

ΘP={arctan2(y,x),arctan2(y,x)≥0arctan2(y,x)+2π,arctan2(y,x)<0ΘP={arctan⁡2(y,x),arctan⁡2(y,x)≥0arctan⁡2(y,x)+2π,arctan⁡2(y,x)<0

但是arctan2arctan⁡2这个操作非常消耗资源，所以我们不会直接计算极角θθ，我们会进行如下优化：

给定极坐标系的坐标$(r,\theta) ，我们可以构建一个关于，我们可以构建一个关于\theta的函数的函数g(\theta)=|\cos\theta|\cos\theta$，这个函数会在第一，二象限递减，第三，四象限递增，接下来有：

g(θ)=r2r2g(θ)=r2|cosθ|cosθr2=|rcosθ|⋅(rcosθ)r2(1)(2)(3)(1)g(θ)=r2r2g(θ)(2)=r2|cos⁡θ|cos⁡θr2(3)=|rcos⁡θ|⋅(rcos⁡θ)r2

其中我们将极坐标系公式代入原式：

x=rcosθ,y=rsinθ,r=√x2+y2x=rcos⁡θ,y=rsin⁡θ,r=x2+y2

得到：

g(θ)=|rcosθ|⋅(rcosθ)r2=|x|⋅xx2+y2(4)(5)(4)g(θ)=|rcos⁡θ|⋅(rcos⁡θ)r2(5)=|x|⋅xx2+y2

在实际计算中为了防止除0我们会在分母加上一个非常小的数εε：

g(θ)=|x|⋅xx2+y2+εg(θ)=|x|⋅xx2+y2+ε

我们可以给出优化版本的比较函数cmp(⋅,⋅)cmp(⋅,⋅)：

cmp(P1,P2)=⎧⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪⎨⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪⎩false,(x1,y1)=(x2,y2)true,y1>0,y2<0false,y1<0,y2>0true,y1>0,y2>0,g(θ1)>g(θ2)true,y1<0,y2<0,g(θ1)<g(θ2)其中 g(θ)=|x|⋅xx2+y2+εcmp(P1,P2)={false,(x1,y1)=(x2,y2)true,y1>0,y2<0false,y1<0,y2>0true,y1>0,y2>0,g(θ1)>g(θ2)true,y1<0,y2<0,g(θ1)<g(θ2)其中 g(θ)=|x|⋅xx2+y2+ε

[![image-20250913111912987](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160203568-1730946312.png)](https://github.com):[楚门加速器官网](https://chuanggeye.com)

#### 形成新的多边形并计算面积

给定任意二维向量L=(x1,y1),M=(x2,y2)L=(x1,y1),M=(x2,y2)，我们可以求解二者共同起点所构成的三角形面积SLMSLM：

SLM=12|L×M|=12|x1y2−y1x2|SLM=12|L×M|=12|x1y2−y1x2|

给定已经按照极角进行排序的点集{P1,…,Pn|n≤8}{P1,…,Pn|n≤8}，我们可以将这些点按照顺序连接为一个闭合的多边形II，这个多边形由n−2n−2个三角形所组成，每个三角形SnSn的面积为Sn=12|(Pn+1−P1)×(Pn+2−P1)|Sn=12|(Pn+1−P1)×(Pn+2−P1)|，那么我们可以算出这个多边形的面积：

SI=n−2∑i=112|(Pi+1−P1)×(Pi+2−P1)|SI=∑i=1n−212|(Pi+1−P1)×(Pi+2−P1)|

[![image-20250913112659777](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160216793-1160568549.png)](https://github.com)

### 计算IoU

上文我们已经得到BEV包围盒B1B1与B2B2的交集面积为SISI，那么我们可以如下算得IoUIoU：

IoU=IntersectionUnion=SISB1+SB2−SIIoU=IntersectionUnion=SISB1+SB2−SI

如果B1,B2B1,B2为3D包围盒，可以如下增加一个高度项：

IoU=IntersectionUnion=HIHUSISB1+SB2−SIIoU=IntersectionUnion=HIHUSISB1+SB2−SI

其中

HI=max(0,min(zB1+hB12,zB1+hB22)−max(zB1−hB12,zB2−hB22))HU=hB1+hB2−HI(6)(7)(6)HI=max(0,min(zB1+hB12,zB1+hB22)−max(zB1−hB12,zB2−hB22))(7)HU=hB1+hB2−HI

[![image-20250913115701052](https://img2024.cnblogs.com/blog/1887071/202509/1887071-20250913160224845-170053882.png)](https://github.com)

## 参考文献

[https://en.wikipedia.org/wiki/Weiler–Atherton\_clipping\_algorithm](https://github.com)

[https://github.com/lilanxiao/Rotated\_IoU](https://github.com)
