# author: Yuning Ding
# date:2022.04.24
# version:0.3
# 修复了windows系统下文件保存功能无效的bug

import logging
import os
import time

import finger

logging.basicConfig(filename='logger.log', level=logging.INFO)

cmp = './1.png'


# cmp:待匹配图片路径 confidence:置信度,返回0为不匹配，1为匹配
def check(cmp, confidence=0.7):
    g = os.walk(r"./dic")
    max_x = 0
    name = ''
    for path, dir_list, file_list in g:
        for file_name in file_list:
            x, y = finger.Comparison(cmp, os.path.join(path, file_name), confidence=confidence)
            if x > max_x:
                max_x = x
                name = file_name[:-4]

    if max_x < confidence:
        logging.info('非法用户访问,' + '时间：' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # for mac os/linux
        # st = 'cp ' + cmp + ' ./invalid/'
        # for win
        st = 'copy ' + cmp + ' ./invalid/' + cmp
        os.system(st)
        return 1
    else:
        logging.info('用户:' + name + ' 置信度：' + str(max_x) + '时间：' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return 0


check(cmp)
# img1='4.png'
# img2='3.png'
# confidence=0.7
#
# # img1和img2为图片路径，confidence为置信度，函数返回值为置信度，是/否
# x,y=finger.Comparison(img1,img2,confidence=confidence)
#
# print(x,y)
