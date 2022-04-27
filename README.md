# finger_cmp
A simple fingerprint matching program.

## 数据准备

存储的指纹图片放入dic文件夹中 待匹配的指纹图片放入根目录finger

## 开发环境

python3.9

## 安装依赖

在命令行中输入

```shell
pip install opencv-python
pip install scikit-image
pip install numpy
pip install opencv-contrib-python
```

## 使用方法

在main.py中修改待匹配img的路径及cofidence置信度，然后运行。 main.py中已经给出一个demo。 如果匹配成功，在logger.txt中会输出用户名，置信度和时间；
匹配失败，在logger中会输出非法提示，同时将指纹存入invalid文件夹。

## 样例logger

```
INFO:root:用户:1 置信度：1.0时间：2022-04-24 19:37:56
INFO:root:用户:1 置信度：1.0时间：2022-04-24 19:38:38
INFO:root:非法用户访问,时间：2022-04-24 19:38:47
INFO:root:非法用户访问,时间：2022-04-24 19:39:23
```

## 样例输出

```
Fingerprint image: 296x560 pixels
    Minutiae: 71
    Local structures: (71, 208)
Fingerprint image: 388x374 pixels
    Minutiae: 39
    Local structures: (39, 208)
Comparison score: 0.59
不匹配
```

