# foci-counter：foci统计GUI工具集合



## NucleusSegmentation: 细胞核分割工具

NucleusSegmentation 是一个基于Python的GUI应用程序，专门用于细胞核的分割和分析。它集成了多种图像处理技术，提供了一个用户友好的界面来帮助生物学研究者快速准确地识别和量化细胞核。

### 功能特点：
- **图像加载**：支持多种图像格式，方便用户导入数据。
- **通道选择**：用户可以根据需要选择不同的颜色通道进行分析。
- **高斯模糊**：提供图像模糊功能，以改善图像质量。
- **局部阈值处理**：实现基于Sauvola算法的局部阈值处理，适应不同区域的亮度变化。
- **小区域移除**：自动移除小于特定大小的区域，提高分割的准确性。
- **细胞核分割**：应用图像形态学算法进行细胞核的自动分割。
- **结果保存**：将分割结果和统计数据保存为CSV文件，方便后续分析。

### 技术栈：
- **Tkinter**：用于构建GUI。
- **PIL (Python Imaging Library)**：用于图像处理。
- **scikit-image**：提供图像形态学操作和分析工具。
- **scipy**：提供科学计算支持。
- **numpy**：用于高效的数值计算。
- **pandas**：用于数据处理和分析。
- **os**：提供操作系统接口功能。

### 使用方法：
1. 运行程序并加载您的图像文件。
2. 选择分析的通道。
3. 调整高斯模糊、局部阈值和分割参数以优化结果。
4. 点击“Segmentation”进行细胞核分割。
5. 查看结果并保存分析数据。



## FociSegmentation: 细胞foci信号分割与统计工具

FociSegmentation 是一款基于Python的图形用户界面(GUI)应用程序，专为生物图像分析设计。它利用了先进的图像处理算法，帮助用户识别和量化图像中的细胞焦点区域。

### 功能特点：
- **图像加载与通道选择**：支持多种图像格式，用户可自由选择感兴趣的颜色通道。
- **高通滤波与高斯模糊**：自动去除背景噪声，增强图像中的细胞焦点特征。
- **局部与全局二值化**：提供两种二值化方法，适应不同图像条件。
- **细胞焦点分割**：基于水洗算法的自动分割功能，精确识别细胞焦点。
- **统计分析**：计算并输出焦点区域的面积、中心位置、平均强度等统计信息。
- **结果可视化**：生成包含焦点和细胞标记的图像，直观展示分析结果。
- **数据导出**：支持将统计数据导出为CSV文件，便于进一步分析。

### 技术栈：
- **Tkinter**：构建GUI。
- **PIL (Python Imaging Library)**：图像处理。
- **scikit-image**：图像形态学操作和分析。
- **scipy**：科学计算。
- **scikit-learn**：机器学习工具，用于最近邻搜索。
- **matplotlib**：绘图和可视化。
- **pandas**：数据处理和分析。

### 使用方法：
1. 运行程序并加载您的图像文件。
2. 选择分析的通道和二值化方法。
3. 调整高斯模糊和二值化参数以优化结果。
4. 点击“Segmentation”进行焦点分割。
5. 查看结果并保存分析数据和图像。



# Installation

```shell
conda create -n foci-counter python=3.11 --yes
conda activate foci-counter
conda install scikit-image=0.22.0 --yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn==1.4.0
pip install pandas==2.2.0
pip install matplotlib==3.8.2
```


# Run Apps

```python
import AppNucleusSegmentation as app
import tkinter as tk
root = tk.Tk()
app = app.NucleusSegmentation(root)
app.run()

import AppFociSegmentation as app
import tkinter as tk
root = tk.Tk()
app = app.FociSegmentation(root)
app.run()
```

