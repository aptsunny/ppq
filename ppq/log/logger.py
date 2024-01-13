import sys
import time
from enum import Enum
from typing import Union

R_BEGIN = '\033[38;5;1m'
G_BEGIN = '\033[38;5;2m'
Y_BEGIN = '\033[38;5;3m'

COLOR_END = ' \033[m'

initialized_loggers = {}


def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class LEVEL(Enum):
    DEBUG     = 0
    INFO      = 1
    WARNING   = 2
    ERROR     = 3

    def __le__(self, other):
        return self.value <= other.value

    @ staticmethod
    def convert(level: str):
        level = level.lower()
        if level == 'debug':
            return LEVEL.DEBUG
        elif level == 'info':
            return LEVEL.INFO
        elif level == 'warning':
            return LEVEL.WARNING
        elif level == 'error':
            return LEVEL.ERROR
        else:
            raise ValueError('the lowercase of param level should be one of debug/info/warning/error'
                            f'however {level} is given')


class Handler:
    def __init__(self, file_name: str=None, level: LEVEL=LEVEL.INFO) -> None:
        self._file_name = file_name
        self._level     = level
        self._fd        = sys.stdout if file_name is None else open(file_name, 'w+', encoding='utf-8')

    def process(self, msg: str, level: LEVEL) -> None:
        if self._level <= level:
            self._fd.write('\r' + msg + '\n')
            self._fd.flush()

    def set_level(self, level: LEVEL) -> None:
        self._level = level


class NaiveLogger(object):
    """A very naive implementation of colored logger, but would be suffice in
    single-process situation here where no race condition would happen."""
    __create_key = object()

    def __init__(self, create_key: object, name: str, level: LEVEL, file_name: str) -> None:
        assert (create_key == NaiveLogger.__create_key), \
            'logger instance must be created using NaiveLogger.get_logger'
        self._name     = name
        self._level    = level
        self._handlers = {'stdout' : Handler(level=level)}
        if file_name is not None:
            self.register_handler(file_name)


    def set_level(self, level: Union[str, LEVEL]):
        if isinstance(level, str):
            level = LEVEL.convert(level)
        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        self._level = level
        for handler in self._handlers.values():
            handler.set_level(level)

    @ classmethod
    def get_logger(cls, name: str, level: Union[str, LEVEL]=LEVEL.INFO, file_name: str=None):
        if name in initialized_loggers:
            return initialized_loggers[name]

        if isinstance(level, str):
            level = LEVEL.convert(level)

        assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
        logger = NaiveLogger(cls.__create_key, name, level, file_name)
        initialized_loggers[name] = logger
        return logger

    def wrap_header(self, type:str) -> str:
        cur_time = get_current_time()
        return '[{}][{}][{}]: '.format(type, self._name, cur_time)

    def info(self, msg: str):
        header = self.wrap_header('INFO')
        print_msg = G_BEGIN + header + COLOR_END + msg

        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.INFO)
            else:
                handler.process(print_msg, LEVEL.INFO)

    def warning(self, msg:str):
        header = self.wrap_header('WARNING')
        print_msg = Y_BEGIN + header + COLOR_END + msg

        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.WARNING)
            else:
                handler.process(print_msg, LEVEL.WARNING)

    def error(self, msg:str):
        header = self.wrap_header('ERROR')
        print_msg = R_BEGIN + header + COLOR_END + msg

        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.ERROR)
            else:
                handler.process(print_msg, LEVEL.ERROR)

    def debug(self, msg: str):
        header = self.wrap_header('DEBUG')
        print_msg = G_BEGIN + header + COLOR_END + msg

        for handler in self._handlers.values():
            if handler._file_name is not None:
                handler.process(msg, LEVEL.DEBUG)
            else:
                handler.process(print_msg, LEVEL.DEBUG)

    def register_handler(self, file_name: str, level: Union[str, LEVEL]=LEVEL.INFO):
        if file_name not in self._handlers:
            if isinstance(level, str):
                level = LEVEL.convert(level)
            assert isinstance(level, LEVEL), 'level should be given by either str or LEVEL instances'
            self._handlers[file_name] = Handler(file_name, level)


    def remove_handler(self, file_name: str):
        if file_name in self._handlers:
            handler = self._handlers[file_name]
            if handler._file_name is not None:
                handler._fd.close()
            self._handlers.pop(file_name)


def print_dict_table(dictionary):
    # 计算列宽
    key_width = max([len(key) for key in dictionary.keys()]) + 2
    value_width = max([len(str(value)) for value in dictionary.values()]) + 2
    index_width = len(str(len(dictionary))) + 2

    # 打印表头
    print(f"+{'-' * index_width}+{'-' * key_width}+{'-' * value_width}+")
    print(f"| {'Index':<{index_width}} | {'Key':<{key_width}} | {'Value':<{value_width}} |")

    # 打印表格内容
    for index, (key, value) in enumerate(dictionary.items(), start=1):
        print(f"| {index:<{index_width}} | {key:<{key_width}} | {str(value):<{value_width}} |")

    # 打印表格底部线
    print(f"+{'-' * index_width}+{'-' * key_width}+{'-' * value_width}+")


def print_table(list1, list2):
    # 计算列宽
    col1_width = max([len(str(i)) for i in range(1, len(list1) + 1)]) + 2
    col2_width = max([len(item) for item in list1]) + 2
    col3_width = max([len(item) for item in list2 if item in list1]) + 2

    # 创建一个表格并打印表头
    table = [['Index', 'OP', 'QuantOP']]
    print(f"+{'-' * col1_width}+{'-' * col2_width}+{'-' * col3_width}+")
    print(f"| {'Index':<{col1_width}} | {'OP':<{col2_width}} | {'QuantOP':<{col3_width}} |")

    # 打印表格内容
    for index, item1 in enumerate(list1, start=1):
        row = [index, item1, "" if item1 not in list2 else item1]
        print(f"| {row[0]:<{col1_width}} | {row[1]:<{col2_width}} | {'':<{col3_width}} |" if row[2] == "" else f"| {row[0]:<{col1_width}} | {row[1]:<{col2_width}} | {row[2]:<{col3_width}} |")

    # 打印表格底部线
    print(f"+{'-' * col1_width}+{'-' * col2_width}+{'-' * col3_width}+")