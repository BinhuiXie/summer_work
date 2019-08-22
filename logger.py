# -*- coding: utf-8 -*-
# 把实时log的filename改成了时间+filename，并且需要传入logroot

import time
import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, logroot, filename, level='info', when='D', fmt='%(message)s'):
        filename = logroot + time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + filename + '.log'
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)
        # 往文件里写入 指定间隔时间自动生成文件的处理器
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


# sample
# if __name__ == '__main__':
#     log = Logger('../../save/all.log', level='debug')
#     i = 10
#     while i:
#         log.logger.debug('acc%.4f', i/100)
#         i -= 1
