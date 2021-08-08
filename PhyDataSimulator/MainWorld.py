import configparser as cf
import os
import time
from math import exp
import numpy as np
import pygame

import component

world_count = 0
count_solid = 0
count_liquid = 1  # 开始有空气
count_planet = 0
file_saved = 0

"""生成高级物理参数"""


class PhyConstant(object):

    def __init__(self, gravity=None, atom_height=None, height_default=None, temperature=None):
        self.gravity = gravity  # 获取重力
        self.atom_height = atom_height  # 获取大气高度
        self.temperature = temperature  # 获取温度
        self.height_default = height_default  # 获取默认高度
        self.air_list = []  # 空气列表初始化
        self.P = 0
        self.h = 0
        self.t = 0
        self.density = 0
        self.miu = 1

    def GetAir(self, pos_y):
        return self.air_list[int(pos_y)]

    def DensityInit(self):
        gravity = self.gravity
        if gravity is not None:
            P = 8000 * (1 + (self.temperature / 273))
            self.P = P  # 自动生成气压值
            self.h = self.atom_height
            self.t = self.temperature
            self.density = 0.05 * (P / 1.01e+5) * (25 / self.temperature) * 1.2  # 自动生成空气密度

    def AirInit(self):  # 自动生成空气随高度的减小情况
        self.air_list = [exp(-x) for x in range(int(self.atom_height))]
        component.AirList = self.air_list
        for h in range(len(self.air_list)):
            if self.air_list[h] >= self.atom_height:
                self.air_list[h] = 0

        component.InitPhyConfig.append(self)

    def SavePhysics(self):
        """

        保存配置至ini文件

        :return: None
        """
        config = cf.ConfigParser()
        with open("WorldConfig.ini", "r+") as configure:
            config.read(configure)
            config.add_section("Physics")
            for item in dir(self):
                if str(item).find("__") == -1:
                    config.set("Physics", str(item), str(getattr(self, item)))


class World(object):

    def __init__(self, d_time=None, Gravity=None, atom_height=None, default_height=None, temperature=None,
                 get_data_tick=None, floor=None, friction=None):
        """
                    初始化世界参数和物理常数
                    使用牛顿经典力学时空观(支持以场的方式进行扩展)
                    支持多体引力计算
                    采用多线程（一个天体一个线程）
                    保证性能

        """
        pygame.init()
        pygame.display.set_caption('1')
        self.screen = pygame.display.set_mode([640, 480])  # 定义窗口大小为640*480
        self.screen.fill([255, 255, 255])  # 用白色填充窗口

        self.time_set = d_time
        self.tim_rec = 0
        self._able = True
        self.Gravity = Gravity  # 重力初始化
        self.atom_height = atom_height  # 大气层高度初始化
        self.default_height = default_height  # 默认海拔初始化
        self.temperature = temperature  # 温度初始化
        self._Solids_list = []  # 固体初始化
        self._Liquids_list = []  # 流体初始化
        self._Planet_list = []  # 星球初始化
        self._filed_list = []
        self.solid_count = count_solid  # 固体计数
        self.liquid_count = count_liquid  # 流体计数
        self.tick = get_data_tick  # 设置参数获取频率
        self.floor = floor  # 设置地面（如果物体和地面接触则反弹）
        self.floor_friction = friction  # 默认摩擦力为0
        self.collide_depth = 256
        self.now_time = time.time()
        """创造一个世界,没有输入的初级参数由下面自动填充"""
        if d_time is None:
            self.time_set = 0.001  # 设置默认dt
        if Gravity is None:
            self.Gravity = 9.8  # 默认重力
        if atom_height is None:
            self.atom_height = 100000  # 默认大气高度
        if default_height is None:
            self.default_height = 0
        if temperature is None:
            self.temperature = 25  # 设置默认温度
        if floor is None:
            self.floor = 0  # 设置默认最低高度
        self.physics = PhyConstant(gravity=self.Gravity, atom_height=self.atom_height,
                                   height_default=self.default_height,
                                   temperature=self.temperature)  # 存储物理参数
        PhyConstant.DensityInit(self.physics)
        PhyConstant.AirInit(self.physics)
        if self.tick is None:
            self.tick = 100
        component.world = self
        if friction is None:
            self.floor_friction = 1
        self.gravity = self.Gravity
        """
        保存

        """
        self.config = None
        with open("WorldConfig.ini", "w+") as config:
            self._c = config

    def __del__(self):  # 删除世界
        self._able = False

    def __str__(self):
        return self.time_set, self.atom_height, self.now_time, self.gravity

    def Execute(self, running_time=None):  # 开始模拟
        """

        :param running_time float'
        模拟时间dt长度可以自行设定

        """
        import TimeManager
        if running_time is None or running_time == "me":
            TimeManager.Process().Run()
        TimeManager.Process().Run(float(running_time) * 1 / self.time_set)
        return

    @staticmethod
    def GetTimeData(switch_time, observe_type=None):
        """

        获取时间字典内的信息
        :param switch_time: 运行的秒钟
        :param observe_type: 观察的种类，Solid或Liquid或Planet或Field
        :return:ReloadDict

        """
        return component.ReloadDict(this_time=switch_time, reload_type=observe_type)

    def SetBasicSolid(self, solid_type=None, init_pos=None, speed=None, angel=None, solid_round=None,
                      name=None,
                      mass=None, electronic=None, magnet=None, friction=1):
        """

        添加基本固体（圆和正方形）

        :param magnet: Solid
        :param solid_type: C or S
        :param init_pos: list or tuple初始位置
        :param speed: 初速率
        :param angel: 初始运动方向
        :param solid_round: Solid的特征长度
        :param name: 名字
        :param mass: 质量
        :param electronic:是否有电场
        :param friction 摩檫系数
        :return: Solid

        """
        import Solids
        if init_pos is not None:
            init_x, init_y = init_pos[0], init_pos[1]
        else:
            init_x, init_y = None, None
        solid_set = Solids.Solid(solid_type=solid_type, init_x=init_x, init_y=init_y, vector_length=speed,
                                 vector_angle=angel,
                                 name=name, solid_len=solid_round, mass=mass, electronic=electronic,
                                 mag_material=magnet, friction=friction)  # 添加固体实例
        self._Solids_list.append(solid_set)
        self.solid_count += 1
        component.SolidList.append(solid_set)  # 存储至组件文件
        return solid_set

    def SetPoly(self, init_pos, solid_v, name, mass, electronic=0, *solid_pos):
        """

        添加多边形

        :param init_pos: 同上
        :param solid_v: 同上
        :param name: 同上
        :param mass: 同上
        :param electronic: 同上
        :param solid_pos: 多个点的二位列表
        :return: Solid

        """
        import Solids
        if init_pos is not None:
            init_x, init_y = init_pos[0], init_pos[1]
        else:
            init_x, init_y = None, None
        solid_set = Solids.Solid(init_x=init_x, init_y=init_y, vector_angle=solid_v[0], vector_length=solid_v[1],
                                 name=name, solid_len=None, mass=mass, electronic=electronic, poly=True,
                                 poly_pos=solid_pos)
        solid_set._SetPolyShape()  # 激活多边形
        self._Solids_list.append(solid_set)

    def SetLiquid(self, pos=None, speed=None, name=None):  # 添加水实例
        import Liquids
        self.solid_count += 1
        liquid_set = Liquids.Liquid(density=1, point=pos, speed=speed, name=name)
        self._Liquids_list.append(liquid_set)
        component.LiquidList.append(liquid_set)  # 存储至组件文件
        liquid_set._InitCalculate()
        return liquid_set

    def SetPlanet(self, mass=None, planet_radios=None, planet_shell_ball_radios=None, plant_init_x=None,
                  planet_init_y=None, name=None):  # 添加行星
        import Planets
        planet_set = Planets.Planet(mass=mass, planet_radios=planet_radios,
                                    planet_shell_ball_radios=planet_shell_ball_radios,
                                    plant_init_x=plant_init_x, planet_init_y=planet_init_y, name=name)
        self._Planet_list.append(planet_set)  # 生成一个星球实例
        component.PlanetList.append(planet_set)  # 存储至组件文件
        return planet_set

    def SetField(self, field_type, filed_data, field_name):
        """

        添加一个场

        :param field_type: 场类型
        :param filed_data: 场向量或标量
        :param field_name: 场名
        :return:field


        """
        from Functions import _AutoSelectFieldName
        field = _AutoSelectFieldName(field_type=field_type, filed_data=filed_data, name=field_name)
        self._filed_list.append(field)
        component.FieldList.append(field)
        return field

    @staticmethod
    def GetSolid(*name):
        """

        :param name:Solid.name
        获取名字对应的Solid

        :return Solid
        """
        result = []
        for item_name in name:
            result.append(component.Gets("Solid", item_name).NameToObject())
        return result

    @staticmethod
    def GetLiquid(*name):
        """

        :param name
        根据名字选取流体
        :return Liquid


        """
        result = []
        for item_name in name:
            result.append(component.Gets("Liquid", item_name).NameToObject())
        return result

    @staticmethod
    def GetPlanet(*name):
        """

        :param name str
        根据名字选取天体
        :return Planet

        """
        result = []
        for item_name in name:
            result.append(component.Gets("Planet", item_name).NameToObject())
        return result

    @staticmethod
    def GetField(*name):
        """

        名字获得场

        :param name: str
        :return: Field
        """
        result = []
        for item_name in name:
            result.append(component.Gets("Liquid", item_name).NameToObject())
        return result

    @staticmethod
    def GetSpeedNeedsSolid(speed: int or float or list or tuple):
        return component.Gets(type_name="Solid", needs=(component.Gets("Solid", needs=speed).SpeedToName()))

    @staticmethod
    def GetMassNeedsSolid(mass: int or float or list or tuple):
        """

                根据质量选择固体
                :param mass: int or iterable
                :return: Solid


        """
        return component.Gets(type_name="Solid", needs=(component.Gets("Solid", needs=mass).MassToName()))

    @staticmethod
    def GetMassNeedsPlanet(mass: int or float or list or tuple):
        """

        根据质量选择天体
        :param mass: int
        :return: Planet


        """
        return component.Gets(type_name="Planet", needs=(component.Gets("Planet", needs=mass).MassToName()))

    @staticmethod
    def GetLaPoint(planet1, planet2, point_type):
        """

                        :param planet2: Planet
                        :param planet1: Planet
                        :param point_type:int or list or tuple
                        根据输入的要求获得拉格朗日点：
                        L1，L2，L3拉格朗日点位于两行星中线上，L4和L5位于行星两侧


        """
        from Functions import _GetLaPoint

        return _GetLaPoint(planet1, planet2, point_type)

    def RemoveSolid(self, *item_name):
        """

        删除对应名字的Solid

        :param item_name: 名字（可以有多个）
        :return: Bool

        """
        for name in item_name:
            for solid_ in self.GetSolid(name):
                component.SolidList.remove(solid_)
        return True

    def RemoveField(self, *item_name):
        """

        删除对应名字的Field

        :param item_name: 名字（可以有多个）
        :return: Bool
        """
        for name in item_name:
            for solid_ in self.GetField(name):
                component.FieldList.remove(solid_)
            return True

    @staticmethod
    def RemoveTrack():
        component.ObserveList = []

    def DestroyPlanet(self, *item_name):
        """

        碎星者（划掉）
        删除对应名字的Planet

        :param item_name:
        :return: Bool
        """
        for name in item_name:
            for solid_ in self.GetPlanet(name):
                component.PlanetList.remove(solid_)
            return True

    def RemoveLiquid(self, *item_name):
        """

        删除对应名字的Liquid

        :param item_name:
        :return:
        """
        for name in item_name:
            for solid_ in self.GetLiquid(name):
                component.LiquidList.remove(solid_)
            return True

    def SetCollideDepth(self, depth):
        """

        设置碰撞检测深度，超出范围则不检测
        默认为1024m

        """
        self.collide_depth = depth

    @staticmethod
    def SetTrack(pos: list, radio, observe_mode, requirement=None, command=None):
        """

        :param command:
        :param requirement:
        :param observe_mode: "P" or "W"
        :param radio int or float
        :param pos:list
        添加“观测点”，这种观测点会无条件记录所有进入观测点的实体数据,并且根据条件执行命令

        """
        track = component.Observe(pos=pos, radio=radio, observe_mode=observe_mode, requirement=requirement,
                                  command=command)
        component.ObserveList.append(track)
        return track

    def LoadConfig(self, file_name):
        """

        加载配置
        :return:Bool

        """
        from Functions import _Recovery
        _Recovery()
        with open(file_name, "r") as input_config:
            config = cf.ConfigParser()
            config.read(input_config)
            self.time_set = config.get("World", "time_set")
            self.gravity = config.get("World", "gravity")
            self.atom_height = config.get("World", "atom_height")

    def ReSave(self):
        import FileManager
        self.config = FileManager.ConfigureControl()
        self.config.SaveWorld()
        print("保存成功,配置文件在%s" % os.path)
        self.physics.SavePhysics()

    @staticmethod
    def SaveSolid():
        for solid in component.SolidList:
            solid.SaveSolid()
        print("固体保存成功，配置文件在：%s" % os.path)

    def get_all_data(self) -> np.ndarray:
        """
        向量化
        :return:
        """
        for planet in self._Planet_list:
            planet.data_matrix.concatenate(np.array([planet.]), axis=1)
