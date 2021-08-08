import configparser as cf
from math import pi
from math import sqrt

import numpy as np

import MainWorld

count = 0


class Planet(MainWorld.World):  # 终于到我最擅长的天体物理环节了(500小时KSP玩家狂喜)

    def __init__(self, mass=None, planet_radios=None, planet_shell_ball_radios=None, plant_init_x=None,
                 planet_init_y=None, name=None):
        super().__init__()
        global count
        count += 1  # 生成标识码以及默认名字
        self.data_matrix = np.array((6, None))
        self.mass = mass  # 行星质量
        self.planet_radios = planet_radios  # 行星半径
        self.planet_boundary = planet_shell_ball_radios  # 行星希尔球半径（引力有效范围）
        self.planet_pos_x = plant_init_x  # 初始坐标设定
        self.planet_pos_y = planet_init_y
        self.name = name
        self.label = count  # 生成标识符
        if self.mass is None:
            self.mass = 100000000000
        if self.planet_radios is None:
            self.planet_radios = 3500000
        if self.planet_boundary is None:
            self.planet_boundary = 1024 * self.planet_radios
        if self.planet_pos_x is None or self.planet_pos_y is None:
            self.planet_pos_x, self.planet_pos_y = 10000000000, 10000000000
        if self.name is None:
            self.name = "未命名星球%d" % count
        with open("WorldConfig.ini", "r+") as config:  # 星球参数固定，所以直接存储
            configure = cf.ConfigParser()
            configure.read(config)
            configure.add_section("Planet%d" % count)
            configure.set("Planet%d" % count, "mass", str(self.mass))
            configure.set("Planet%d" % count, "radio", str(self.planet_radios))
            configure.set("Planet%d" % count, "pos", str([self.planet_pos_x, self.planet_pos_y]))
            configure.set("Planet%d" % count, "name", str(self.name))

    def __sizeof__(self):  # 返回行星希尔球大小
        return pi * self.planet_boundary ** 2

    def __len__(self):
        return pi * self.planet_radios ** 2  # 返回行星大小

    def __str__(self):
        return [self.mass, self.planet_radios, self.planet_pos_x, self.planet_pos_y, self.planet_boundary]  # 返回全部参数

    def __del__(self):
        self._able = False  # 删除星球
        with open("WorldConfig.ini", "r+") as config:
            configure = cf.ConfigParser()
            configure.read(config)
            configure.set("Planet%d" % self.label, "mass", "0")

    def Lagrangianpoint(self, others):
        """:param others Planet
        求拉格朗日点的，给出两个planet，返回拉格朗日点(xy坐标)"""
        from Functions import LG_calculate
        assert isinstance(others, Planet), "你要输入1个天体来计算拉格朗日点"
        return LG_calculate(self, others)  # 返回拉格朗日点

    def PlanOrbit(self, vector):  # 轨道规划
        """

        vector:param Vector
        :return a function（use sympy）

        """
        ...

    def UpdateGravity(self):
        from component import SolidList
        from Functions import GetGravitationForce
        for solid in SolidList:
            if solid.solid_type == "C":  # 刷新引力向量(在多线程中)
                solid.a_vector += GetGravitationForce(self, solid)
            if solid.solid_type == "S":
                solid.a_vector1 += GetGravitationForce(self, solid)
                solid.a_vector2 += GetGravitationForce(self, solid)
                solid.a_vector3 += GetGravitationForce(self, solid)
                solid.a_vector4 += GetGravitationForce(self, solid)

    def GetPlanetOrbit(self):
        return {"v1": sqrt(6.67 * 10e-21 * self.mass / self.planet_radios),
                "v2": sqrt(2) * sqrt(6.67 * 10e-21 * self.mass / self.planet_radios)}

    def SetOrbit(self, solid, orbit_height):
        """
        设定Solid环绕行星
        :param solid: Solid
        :param orbit_height: 环绕高度
        :return: Solid
        """
        solid.v_vector.r = sqrt(6.67 * 10e-21 * self.mass / orbit_height)
        solid.a_vector.o = 90
        solid.pos_x = self.planet_pos_x + self.planet_radios
        solid.pos_y = 0
        return solid


