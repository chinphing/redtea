### 目标
    这是一个轻量级易用的C++深度学习框架，初衷是为了加深对深度学习算法的理解，其次是希望有一个跨平台嵌入式的深度学习框架，以便集成到将来需要实现的自然语言处理模型当中。

### 设计
    本框架的设计思想主要参考tensorflow框架，所有的操作都是Tensor，模型都是由许多Tensor通过图的方式连接起来，前向传播以及反向求导都是自动进行，使用的时候可以像普通计算表达式一样通过加减乘除符号来构建基本的模型，用户不需要关注内部实现细节。

### 依赖库
     矩阵计算库 eigen 3.3.4 

### 开发环境 
    操作系统 Ubuntu 14.4， 
    编译器 G++ 5.4.0，
    调试器 gdb 7.11.1， 
    构建工具 cmake 3.5.1

### 编译步骤
    1、官网下载eigen的源码，在当前工程的同一级目录解压。比如：项目所在目录为：/home/redtea,那么eigen解压后的目录为：/home/eigen
    2、进入CMakeList.txt文件所在目录，执行命令：cmake ./，这一步用于生成Makefile文件
    3、执行命令：make，这一步直接生成可执行文件
    4、运行测试程序：./test-lsl 1000，是一个基于最小平方差损失的线性回归模型，训练迭代1000次。

