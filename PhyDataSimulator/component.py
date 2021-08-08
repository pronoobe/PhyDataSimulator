import configparser as cf
from math import sqrt

count_1 = 0
SolidList = []
PlanetList = []
"""世界全部参数"""
InitPhyConfig = []
"""全部物理参数"""
InitSolidConfig = []
"""向量管理"""
VVector = []
"""速度向量（每个点只有一个）结构：名字对应[位置X,位置Y,r,o]"""
aVector = []
"""加速度向量（可以有多个）"""
LiquidList = []  # 流体列表
AirList = []  # 空气随高度变化的列表
FieldList = []  # 场列表
ObserveList = []
SolidDict = {}  # 静态存储对象名字和速度(只有这2个,保持数据简洁性)
PlanetDict = {}
LiquidDict = {}
FieldDict = {}
RuntimeData = {}  # 存储每个时间点的所有信息,字典的key为时间，value为Gets.all_running_items
world = None
count = 0  # 观测点计数
default_field_set = ['Speed', 'Accelerate', 'pos', 'Electric', 'Wind', 'Heat', 'density', 'Magnetic']


def _Len(pos1, pos2):
    return sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))  # 求距离


def _AllRunningItems():
    return [SolidList, LiquidList, PlanetList, InitSolidConfig]


class Gets(object):
    """:param:类名，限制名，限制范围
    :return:符合条件的类
    这个类是用来统一管理符合条件的类并返回的"""

    def __init__(self, type_name: str, needs: list or tuple or float or int or str):
        """这是一个“智能”查找器
        :param:输入要找的实体类别以及要求（要求可以是名字，速度大小，速度范围等，主要是给World作为接口使用）
        比如说，在world类中的GetSolidSpeed函数，type_name就是“Solid”，needs就是想找的物体名字
        而在  World中应该调用Gets.FindSpeed函数，这样就会自动返回这个物体名字对应的速度，格式如下：
        needs:param
        needs(str):"ALL":全部符合type_name条件下的实体（比如固体，流体等）
                   "Name":某一特定实体的名字（name）
        needs(int or str):符合实体某个属性的值匹配
        needs(str or tuple):符合某个属性的范围的匹配
        """
        self.type_name = type_name
        self.data_range = needs

    @property  # 动态属性
    def auto_list(self):  # 自动返回对应列表
        if self.type_name == "Solid":
            return SolidList
        if self.type_name == "Liquid":
            return LiquidList
        if self.type_name == "Planet":
            return PlanetList
        if self.type_name == "Filed":
            return FieldList

    @property
    def auto_dict(self):  # 自动返回相应字典
        if self.type_name == "Solid":
            return SolidDict
        if self.type_name == "Liquid":
            return LiquidDict
        if self.type_name == "Planet":
            return PlanetDict
        if self.type_name == "Field":
            return FieldDict

    def NameToObject(self):
        """:param：寻找符合名字的实体
        :return:返回一个object"""

        assert isinstance(self.data_range, str), "请输入名字(字符串)来寻找名字"
        if self.type_name == "Solid":
            for solid in SolidList:
                if solid.name == self.data_range:
                    return solid

        #  寻找符合名称的固体_range, self.auto_list)
        #  寻找合适名字的液体
        if self.type_name == "Liquid":
            for liquid in LiquidList:
                if liquid.name == self.data_range:
                    return liquid
        if self.type_name == "Planet":
            for liquid in PlanetList:
                if liquid.name == self.data_range:
                    return liquid
                #  寻找合适名字的星球
                #  寻找合适名字的星球

    def NameToSpeed(self):  # 根据实体名字返回实体速度
        """
        :return:一个类的速度或者一些类的速度"""
        if self.data_range == "ALL" and self.type_name == "Solid":  # 寻找所有符合条件的实体的固体
            result = []  # ↑固体情况
            for x in SolidList:
                result.append(x.v_vector)
            return result
        if self.data_range != "ALL" and self.type_name == "Solid":  # 寻找一个符合条件的固体
            for x in SolidList:
                if x.name == self.data_range:  # 寻找符合条件的固体
                    return x.v_vector  # 返回速度向量
        if self.data_range == "ALL" and self.type_name == "Liquid":  # 情况为流体，寻找速度(同上)
            result = []
            for x in LiquidList:
                result.append(x.speed)
            return result
        if self.data_range != "ALL" and self.type_name == "Liquid":
            for x in LiquidList:
                if x.name == self.data_range:
                    return x.speed  # 返回流速
        if self.type_name == "Planet":
            return 0  # 行星速度为0

    def SpeedToName(self):  # 根据速度寻找名字
        for item in self.auto_list:
            if isinstance(self.data_range, list) or isinstance(self.data_range, tuple):
                if self.data_range[0] <= item.get(item.name) <= self.data_range[1]:
                    return item.get(item.name)
            if isinstance(self.data_range, int or float):
                if item.get(item.name) == self.data_range:
                    return item.get(item.name)

    def MassToName(self):  # 根据质量选择实体
        """:return：返回符合条件的实体"""
        result = []
        for x in self.auto_list:
            if isinstance(self.data_range, int or float):
                if self.data_range - x.mass < 1e-6:  # 当输入为一个固定值
                    result.append(x.name)  # 把符合条件的实体添加进返回值
            if isinstance(self.data_range, list or tuple):  # 输入一个范围
                if self.data_range[0] < x.mass < self.data_range[1]:
                    result.append(x.name)
        return result

    def NameToMass(self):
        """:return相应的质量"""
        if isinstance(self.data_range, str):
            for item in self.auto_list:
                if item.name == self.data_range:
                    return item.mass

    def FindTime(self):
        """寻找特定时间的所有实体"""
        if isinstance(self.data_range, float or int):
            return RuntimeData.get(self.data_range)


class Observe(object):
    def __init__(self, pos=None, radio=None, observe_mode=None, requirement=None, command=None):

        """

        :param pos:list(1*2)
        :param radio num
        :param observe_mode:str

        """

        global count
        print(observe_mode)
        self.pos = (0, 0) if pos is None else pos
        self.radio = 10 if radio is None else radio
        self.observe_mode = "P" if observe_mode is None else observe_mode  # 所有类型：P：print模式，S：save模式
        self.activate = 1
        self.item_dict = {}
        self.label = "ObservePoint%d" % count
        with open("ObservePoint%d.txt" % count, 'w+') as config_file:
            self.data_save = cf.ConfigParser()
            self.data_save.read(config_file)
        self.doing_thing = CommandInterpreter(requirement, command)

    @property
    def _auto_list(self):
        return SolidList

    def Activate(self):
        """

        激活观测（默认启动，并且文件保存模式为False）

        """
        self.activate = 1

    def Pause(self):
        """

        关闭观测点，停止记录信息


        :return:
        """
        self.activate = 0

    def MainStep(self):
        """

        启用观测（使用多线程）


        :return:print or dict
        """

        for items in SolidList:
            """

            一段垃圾代码，不要看了，就是一些二分类，保存进入文件之类的

            """
            if _Len(self.pos, items.now_pos) <= self.radio:
                if self.observe_mode == "P":
                    for SOLID in self._auto_list:
                        print("位置：", SOLID.now_pos, "name:",
                              SOLID.name)
                if self.observe_mode == "List":
                    result = []
                    for SOLID in self._auto_list:
                        result.append(SOLID.now_pos)
                        result.append(SOLID.name)
                    return result
                if self.observe_mode == "S":
                    global count
                    global count_1
                    with open("ObservePoint%d.txt" % count, 'r+') as config:

                        configuration = cf.ConfigParser()
                        configuration.read(config)
                        configuration.add_section(self.label)
                        for SOLID in self._auto_list:
                            config.write('时间:[%d]位置：[%.2f,%2f], 速度:%.2f, 加速度：%.2f\n' % (count_1,
                                                                                        SOLID.now_pos[0],
                                                                                        SOLID.now_pos[1],
                                                                                        SOLID.v_vector.r,
                                                                                        SOLID.a_vector.r))
                            count_1 += 1


class ReloadDict(object):
    def __init__(self, this_time, reload_type):
        self.time = this_time
        self.reload_type = reload_type

    def GetThisTimeData(self):
        """

        获得此时符合要求的实体数据

        :return: str(item)
        """
        temp_data = RuntimeData.get(self.time)
        for kind in temp_data:
            for item in kind:
                if item.name == self.reload_type:
                    return str(item)


class CommandInterpreter(object):
    def __init__(self, requirement=None, command=None):
        """
        命令解释器，传入Tracker的命令进行解释，比如说传入 (Solid(name:str) speed[0,2],break)则如果这个solid速度在0-2时则停止循环
        speed[a,b], collide_with[name], pos_len[a,b]
        :param command: 命令语句
        """
        if requirement is None or command is None:
            self.require = "null "
            self.require_item = 'null'
            self.command = "null"
            requirement = self.require + self.require_item
        self.require_item, self.require = requirement.split(' ')
        self.command = command
        self._all_things = ["print", "break", "time_break"]
        self._all_require_ = ["speed", "collide_with", "pos_len"]
        self._active = False
        left = self.require_item.find('(')
        right = self.require_item.find(')')
        if left != -1 and right != -1 and self.require_item.find("Solid") == 0:
            self.require_item = Gets("Solid", self.require_item[left + 1:right]).NameToObject()

    def Refresh(self):
        left = self.require_item.find('(')
        right = self.require_item.find(')')
        if left != -1 and right != -1 and self.require_item.find("Solid") == 0:
            self.require_item = Gets("Solid", self.require_item[left + 1:right]).NameToObject()

    def Active(self):
        if self._active is True:
            if self.command == "print":
                for solid in SolidList:
                    print("%s\n[%f,%f]\n[%f,%f]\n" % (
                        solid.name, solid.pos_x, solid.pos_y, solid.v_vector[0], solid.v_vector[1]))
            elif self.command == "break":
                for solid in SolidList:
                    solid.v_vector, solid.a_vector = solid.v_vector.AddZeroVector(), solid.a_vector.AddZeroVector()
                    break
            elif self.command == "time_break":
                return SolidList
            else:
                raise TypeError("没有这个command语句")

    def Check(self):
        def find_str():
            return eval(self.require[self.require.find("["):self.require.find("]") + 1])

        if self.require.find("speed") != -1:
            self.require = find_str() + "s"
            if Gets("Solid", Gets("Solid", self.require).SpeedToName()).NameToObject() is not None:
                self._active = True
        elif self.require.find("collide_with") != -1:
            self.require = find_str()
        else:
            self.require = find_str() + "p"
        if self.require[-1] == "s":
            self.require = eval(self.require[:-1])
            for solid in SolidList:
                if min(self.require) <= solid.v_vector.r <= max(self.require):
                    self._active = True
        else:
            self.require = eval(self.require[:-1])
            for solid in SolidList:
                if min(self.require) <= solid.v_vector.r <= max(self.require):
                    self._active = True
