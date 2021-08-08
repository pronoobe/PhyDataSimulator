import configparser as cf

import MainWorld


# 保存和读取参数，运行结果等

class ConfigureControl(MainWorld.World):
    """配置加载类，主要功能：保存配置和数据
    至cfg文件、从cfg中读取配置、修改、刷新cfg
    配置文件、根据文件提供索引"""

    def __init__(self):
        """初始化时，把World中全部信息存储进设置文件"""
        super().__init__()
        self.__config_file__ = None

    def SaveWorld(self):
        with open("WorldConfig.ini", "w+") as world_config_ini:
            self.__config_file__ = cf.ConfigParser()
            self.__config_file__.read(world_config_ini)
            self.__config_file__.add_section(section="World")
            for item in dir(self):
                if str(item).find('__') == -1 and str(item).find('_') != 0 and str(item).find('_') != -1:
                    self.__config_file__.set('World', str(item), str(getattr(self, item)))
            self.__config_file__.write(world_config_ini)

    def LoadWorld(self, path):
        with open(path, 'r') as world_config_ini:
            config = cf.ConfigParser()
            config.read(world_config_ini)
            super().time_set = config.get("World", "time_set")
            super().gravity = config.get("World", "gravity")
            super().temperature = config.get("World", "temperature")
            super().floor = config.get("World", "floor")
