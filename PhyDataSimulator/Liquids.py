import Functions
import MainWorld
import configparser as cf

count = 0


class Liquid(MainWorld.World):

    def __init__(self, density=None, point=None, speed=None, name=None):

        """

        point:param 2D list or tuple
        density:param float
        speed:param float
        name:param str

        """

        global count
        count += 1  # 调用计数函数，生成名称
        super().__init__()
        if density is None:
            density = 1
        if point is None:
            point = [[0, 0], [0, 1], [1, 0], [1, 1]]
        if name is None:
            name = "未命名流体%d" % count
        if speed is None:
            speed = [0, 0]
        self.density = density  # 流体密度
        self.point = point  # 流体位置
        self.speed = speed  # 流体速度【速率，方向角度】
        self.name = name  # 流体名字
        self.solid_count = 0  # 用来给进入流体的固体计数
        self.status = 0  # 状态参数，0为流体休眠，1为流体启动
        self.calculate = None
        self._able = True

    def __str__(self):  # 返回流体物理信息
        return {"流体名称": self.name, "密度": self.density, "位置": self.point, "流速": self.speed}

    def __sizeof__(self):
        return 65536

    def __abs__(self):
        return self.solid_count

    def __getitem__(self, item):
        assert 1 <= item <= 4, "输入1-4来获取流体四个点坐标"
        if item == 1:
            return self.point[0]
        if item == 2:
            return self.point[1]
        if item == 3:
            return self.point[2]
        if item == 4:
            return self.point[3]

    def __del__(self):
        self._able = False

    def _InitCalculate(self):
        self.calculate = Functions._LCalculate()  # 生成流体计算对象

    def Trigger(self):  # 判断流体是否为触发状态

        return self.calculate.Trigger()

    @staticmethod
    def _Save():
        with open("WorldConfig.ini", "w+") as config:
            configure = cf.ConfigParser()
            configure.read(config)
            configure.add_section("Liquid%d" % count)
            configure.set("Liquid%d" % count, "name")
