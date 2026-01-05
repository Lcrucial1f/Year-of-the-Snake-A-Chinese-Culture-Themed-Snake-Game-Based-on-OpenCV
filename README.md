# 🐍 蛇年畅游：基于 OpenCV 的中华文化贪吃蛇
本项目是一个基于 **OpenCV + 摄像头手势识别** 的贪吃蛇游戏，结合中华文化主题与多种图像处理特效，支持实时交互。

---

## 一、运行环境要求
* 操作系统：Windows 10 / 11
* 摄像头：笔记本自带摄像头
* Python：3.7（必须）
* 推荐使用Conda管理环境

---

## 二、从零配置环境（Conda）

### 1️⃣ 创建并激活环境

```bash
conda create -n snake-opencv python=3.7 -y
conda activate snake-opencv
```

### 2️⃣ 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```



---

## 三、运行程序

确保摄像头可用，在项目根目录执行：

```bash
python main.py
```

程序启动后会自动打开摄像头窗口并进入游戏。

---

## 四、基本玩法说明

* 游戏通过 **手势控制蛇的移动**
* 支持多种 **图像处理特效切换**
* 支持 **护盾技能**（开启后短时间内不会死亡）
* 游戏界面与菜单均为中文显示

---

## 五、项目文件说明（关键）

```
├─ main.py              # 游戏主程序（直接运行这个）
├─ requirements.txt     # 依赖列表
├─ README.md            # 说明文档         
├─ bg.jpg               # 背景图片
├─ dount.png            # 食物图片

```

---

