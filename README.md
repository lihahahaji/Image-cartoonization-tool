项目结构：
Bilateral Filter
├─ doc_src -- 课程报告 latex 源码
├─ Experimental_env  -- 实验环境
│  ├─ opencv_version.py -- 基于opencv库的双边滤波器（运行效率高）
│  └─ original_version.py -- 手动实现的双边滤波器
├─ input -- 输入图片
│  ├─ input_1.png
│  ├─ input_2.png
│  └─ input_3.png
├─ output -- 输出图片
│  ├─ output_1.jpg
│  ├─ output_2.jpg
│  └─ output_3.jpg
├─ README.txt
├─ result -- 实验对照结果
│  ├─ res_1.png
│  ├─ res_2.png
│  └─ res_3.png
└─ src -- 工程源码
   ├─ bilateral_filter.py -- 双边滤波器方法
   ├─ cartoonize.py -- 卡通化方法
   ├─ edge_detection.py -- 边缘检测方法
   └─ main.py -- 主函数，程序入口

运行步骤：
1. 设置好输入图片
2. 运行 main.py