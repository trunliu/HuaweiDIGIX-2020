2020 DIGIX全球校园AI算法精英大赛
=================================
队伍名称 ***[您吃了么]***  

### 比赛简介
* 完成数码设备图像检索任务，即给定一张含有数码设备的查询图片，算法需要在数码设备图像库中查找并返回含有该商品的图片。
* 本次比赛提供的数据集包含两部分：  
	* 训练集(train)和测试集(test)。其中训练集包含不同数码商品，每个数码设备商品会对应数量不等的图片
	* 测试集包含查询图片(query)和检索图像库(gallery)，用于指标评估  
* 采用top-1 accuracy 和mAP@10两种测评指标加权`𝟧𝟢% * 𝚝𝚘𝚙-𝟷 ＋ 𝟧𝟢% * 𝚖𝖠𝖯@𝟷𝟢`  
* A/B榜  

### 本代码机器学习环境搭建
* 系统配置
	* 操作系统（OS）：Win10 x64
	* 显卡（GPU）：NVIDIA GeForce GTX 1060 6GB
* 安装内容
	* Anaconda3
	* Pycharm-2020专业版
	* keras 2.2.0
	* tensorflow_gpu-1.9.0-cp36-cp36m-win_amd64.whl
	* cuda_9.2.148_win10
	* cudnn-9.2-windows10-x64-v7.6.5.32  
具体步骤参考[环境搭建](https://github.com/liuwentao1992/HuaweiDIGIX-2020/blob/master/%E9%85%8D%E7%BD%AE.md)
* 第三方库
	* h5py  
	* numpy  
	* os  
	* csv  
	* keras  
	* tensorflow  
	* matplotlib.pyplot  
	* PIL  
	* shutil  
	
### 算法实现逻辑
* 制作训练验证测试集
* 构建网络
* 迁移学习
* 图像增强
* 训练结果可视化
* 提起图库特征建立特征数据库
* 检索查询输出结果

### 运行调试
