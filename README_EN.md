# finger_cmp

A simple fingerprint matching program.

## Data preparation

The stored fingerprint picture is put into the dic folder. The fingerprint picture to be matched is put into the root
directory finger.

## Development environment

python3.9

## Installation dependencies

Enter on the command line

```shell
pip install opencv-python
pip install scikit-image
pip install numpy
pip install opencv-contrib-python
```

## How to use

Modify the path and cofidence confidence of img to be matched in main.py, and then run. A demo has been given in
main.py. If the match is successful, the user name, confidence and time will be output in logger.txt;

If the match fails, illegal prompts will be output in the logger, and the fingerprint will be stored in the invalid
folder.

## sample log

```
INFO:root:用户:1 置信度：1.0时间：2022-04-24 19:37:56
INFO:root:用户:1 置信度：1.0时间：2022-04-24 19:38:38
INFO:root:非法用户访问,时间：2022-04-24 19:38:47
INFO:root:非法用户访问,时间：2022-04-24 19:39:23
```

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

