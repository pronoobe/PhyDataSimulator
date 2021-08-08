"""


1，角度转换函数，把直角坐标转换成极坐标，然后传入2和3进行计算
2.牛顿运动函数
3.流体力学运算，主要是空气阻力（在2中调用3）
4，微分，用来提供2和3的微元运算，并且用TimeManage来保存每个dt物体的状态（精度是0.01）
5.进行场的定义和运算，并且内置了一些实例
6.其他杂项函数，比如说投影、线性变换等
结构如下：
----------1.数学函数，比如两点间距离----------
----------2.碰撞检测-----------------------
----------3.角度和位置变换------------------
----------4.碰撞动力学运算------------------
----------5.流体运算-----------------------
----------6.天体运算-----------------------
----------7，场定义和场运算-----------------
注解：1和3属于数学运算类，其他属于物理运算类
举个例子：假设两个物体碰撞，首先传入各种参数，然后调用2，计算出初始加速度，然后调用4，计算出dt之后速度和加速度方向，然后时间+dt，反复循环



"""
import abc
import cmath

from numpy import *

import component


# ————————————————————————————————————————————一些通用函数——————————————————————————————————————————————————


def _Len(pos1, pos2):
    return sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))  # 求距离


import Liquids  # 必须放在这才能import上面那个函数，否则会报错


def AirForce(C, density, wind_square, speed):  # 空气阻力运算

    return 0.5 * C * density * speed ** 2 * wind_square


def SquirePrescription(length: float or int, angel: float or int):  # 正方形任意角度iD面积（等效为长度）投影计算
    return length * (1 + sqrt(math.sin(angel)))


def XYtoRO(x_x, y):
    cn = complex(x_x, y)
    ro = cmath.polar(cn)
    return ro


def _SpinMatrix(angels, *points):
    spin_matrix = [[eval(format(cos(angels), ".4f")), eval(format(-sin(angels), ".4f"))],
                   [eval(format(sin(angels), ".4f")), eval(format(cos(angels), ".4f"))]]

    result = []
    for _point in points:
        for point in _point:
            point_x = point[0] * spin_matrix[0][0] + point[1] * spin_matrix[0][1]
            point_y = point[0] * spin_matrix[1][0] + point[1] * spin_matrix[1][1]
            result.append([point_x, point_y])
    return result


# ————————————————————————————————————————碰撞预备函数——————————————————————————————————


def _InSquare(square_pos: list, point_pos: list):  # 判断某点是否在矩形内(矩形应该与xy平行)
    one_angel = 0
    """思路：统计point点和正方形任意两点间的夹角，然后把他们加起来，如果等于360°，则在正方形内部"""
    for pos in square_pos[1:]:  # 遍历角度
        angel = arccos(
            (_Len(square_pos[0], point_pos) ** 2 + _Len(pos, point_pos) ** 2 - _Len(square_pos[0], pos) ** 2) / (
                    2 * _Len(square_pos[0], point_pos) ** 2 * _Len(pos, point_pos) ** 2))
        one_angel += angel
        if one_angel == 2 * math.pi:
            return True, point_pos

    return False


def _InCircle(circle, pos):
    """circle:param Solid
    :return bool  思路：判断半径和和圆心和点之间的距离"""
    if _Len(circle.center, pos) <= circle.solid_round:
        return True
    else:
        return False


def ROtoXY(r, o):
    xy = cmath.rect(r, o)
    return xy.real, xy.imag


class QuadTree(object):
    """

    四叉树，用于多个物体碰撞时的连续检测
    对于多个物体(物体数量大于4)，首先使用四叉树确定碰撞发生的范围
    然后使用jgk算法，降低算法复杂度，满足及时性要求

    """

    def __init__(self, max_levels, level, pos_x, pos_y, boundary_x=None, boundary_y=None, max_objects=None):
        """        最大节点总数:param max_levels int
        最大深度:param level ont
        四个节点（右上左上左下右下）:param pos_x, pos_y:
        """
        if boundary_x is None:
            boundary_x = component.world.collide_depth
        if boundary_y is None:
            boundary_y = component.world.collide_depth
        if max_levels is None:
            max_levels = 32
        if level is None:
            level = 5
        self.max_levels = max_levels
        self.level = level
        self.boundary = [pos_x, pos_y, boundary_x, boundary_y]
        self.objects = []
        self.nodes = []  # 四叉树节点
        self.max_objects = max_objects
        self._x, self._y, self._r, self._o = None, None, None, None

    def Split(self):

        """

        分割四叉树

        """
        next_level = self.level + 1
        sub_width = self.boundary[2] / 2
        sub_height = self.boundary[3] / 2
        x = self.boundary[0]
        y = self.boundary[1]
        # 右上节点
        self.nodes[0] = QuadTree(self.max_levels, next_level, pos_x=x + sub_width, pos_y=y + sub_height,
                                 boundary_x=self.boundary[2], boundary_y=self.boundary[3],
                                 max_objects=self.max_objects)
        # 左上节点
        self.nodes[1] = QuadTree(self.max_levels, next_level, x, y, self.boundary[2], self.boundary[3],
                                 max_objects=self.max_objects)
        # 左下节点
        self.nodes[2] = QuadTree(self.max_levels, next_level, x, y + sub_height, self.boundary[2], self.boundary[3],
                                 max_objects=self.max_objects)
        # 右下节点
        self.nodes[3] = QuadTree(self.max_levels, next_level, x + self.boundary[2], y + self.boundary[3],
                                 self.boundary[2], self.boundary[3], self.max_objects)

    def GetIndex(self, solid):
        """

        确定固体位于哪个节点内
        :param solid: Solid
        :return array


        """

        index = []
        vertical_midpoint = self.boundary[0] + self.boundary[2] / 2
        horizontal_midpoint = self.boundary[0] + self.boundary[3] / 2
        start_north = solid.p_rect[1] < horizontal_midpoint
        start_west = solid.p_rect[0] < vertical_midpoint
        end_east = solid.p_rect[0] + solid.p_rect[2] > vertical_midpoint
        end_south = solid.p_rect[1] + solid.p_rect[3] > horizontal_midpoint

        if start_north and end_east:
            index.append(0)
        if start_west and start_north:
            index.append(1)
        if start_west and end_south:
            index.append(2)
        if end_east and end_south:
            index.append(3)
        return index

    def Insert(self, solid):
        if len(self.nodes) >= 1:  # 如果节点中有元素，则进行插入
            index = self.GetIndex(solid)
            for i in range(len(index)):
                self.nodes[index[i]].append(solid.p_rect)
                return 1
        else:
            self.objects.append(solid.p_rect)
        if self.level < len(self.objects) < self.max_objects:
            if len(self.nodes) != 0:  # 分割节点
                self.Split()
            for i in range(len(self.objects)):
                index = self.GetIndex(self.objects[i])
                for K in range(len(index)):
                    self.nodes[index[K]].append(self.objects[i])
        self.objects = []  # 清空节点固体

    def LoadSolid(self):
        """加载所有固体"""
        for solid in component.SolidList:
            self.Insert(solid.p_rect)

    def Retrieve(self, solid):
        index = self.GetIndex(solid)
        result = self.objects
        # 发生碰撞则返回
        if len(self.nodes) != 0:
            for i in range(len(index)):
                result.append(self.Retrieve(self.nodes[index[i]]))
        self.nodes = []
        return result


"""输入坐标时的转换（扩展用）"""


# ——————————————————————————————————————位置和向量计算——————————————————————————————#


class Position(object):
    """ x, y或 r, o 其中o是角度(0~360)"""

    def __init__(self, a, b, pos_type):
        """
        _type = "xy" or "ro"
        """
        self.SetPos(a, b, pos_type)

    def _set_x_(self, value):
        self._x = value
        self._set_ro()

    def _set_y_(self, value):
        self._y = value
        self._set_ro()

    def _set_r_(self, value):
        self._r = value
        self._set_xy()

    def _set_o_(self, value):
        self._o = value
        self._set_xy()

    def _set_xy_(self, value):
        self.SetPos(*value)

    def _set_ro_(self, value):
        self.SetPos(*value, "ro")

    # property
    x = property(lambda self: self._x, _set_x_)
    y = property(lambda self: self._y, _set_y_)
    r = property(lambda self: self._r, _set_r_)
    o = property(lambda self: self._o, _set_o_)
    xy = property(lambda self: (self._x, self._y), _set_xy_)
    ro = property(lambda self: (self._r, self._o), _set_ro_)

    def SetPos(self, a, b, _type="xy"):
        if _type == "xy":
            self._x, self._y = a, b
            self._set_ro()
        elif _type == "ro":
            self._r, self._o = a, b
            self._set_xy()
        else:
            raise Exception("Un_support type of " + _type)

    # -----------------------------------------------------------------

    @staticmethod
    def _GetTheta(a, b):
        """
        计算与x轴正向的标准夹角(逆时针为正)
        """
        if a == 0 and b == 0:
            return 0
        do = 0
        if a == 0:
            if b > 0:
                return 90
            else:
                return 270
        elif a < 0:
            if b > 0:
                do = 180
            elif b < 0:
                do = -180
            else:
                return 180
        return math.degrees(math.atan(b / a)) + do

    def _set_ro(self):
        """由xy确定ro"""
        self._r = (self._x ** 2 + self._y ** 2) ** .5
        self._o = self._GetTheta(self._x, self._y)

    def _set_xy(self):
        """由ro确定xy"""
        self._x = self._r * math.cos(math.radians(self._o))
        self._y = self._r * math.sin(math.radians(self._o))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "(x, y):{}\n(r, o):{}".format((round(self._x, 4), round(self._y, 4)),
                                             (round(self._r, 4), round(self._o, 4)))


class Vector(object):  # 向量，支持xy ro坐标

    @staticmethod
    def _InitV(p):  # 创建实例化向量
        v = Vector.__new__(Vector)
        v.r = p.r
        v.o = p.o
        return v

    def __init__(self, r=None, o=None):
        self.r = r
        self.o = o

    @property  # 设为动态属性
    def ang(self):
        return Position(self.r, self.o, pos_type="ro")

    def __add__(self, other):
        if isinstance(other, Vector):
            result_x = self.ang.x + other.ang.x
            result_y = self.ang.y + other.ang.y
            return Vector(r=sqrt(result_y ** 2 + result_x ** 2), o=180 * (arctan(result_y / result_x)) / pi)
        elif hasattr(other, "__getitem__"):
            other = Vector(r=other[0], o=other[1])
            result_x = self.ang.x + other.ang.x
            result_y = self.ang.y + other.ang.y
            return Vector(r=sqrt(result_y ** 2 + result_x ** 2), o=180 * (arctan(result_y / result_x)) / pi)
        else:
            return Vector(self.r + other, self.o + other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Vector):
            result_x = self.ang.x - other.ang.x
            result_y = self.ang.y - other.ang.y
            if result_x == 0:
                return Vector(r=sqrt(result_y ** 2 + result_x ** 2), o=self.o)
            return Vector(r=sqrt(result_y ** 2 + result_x ** 2), o=180 * (arctan(result_y / result_x)) / pi)
        elif isinstance(other, list) or isinstance(other, tuple):
            other = Vector(r=other[0], o=other[1])
            result_x = self.ang.x - other.ang.x
            result_y = self.ang.y - other.ang.y
            return Vector(r=sqrt(result_y ** 2 + result_x ** 2), o=180 * (arctan(result_y / result_x)) / pi)
        else:
            return Vector(self.r - other, self.o - other)

    def __abs__(self):
        return sqrt(self.ang.x ** 2 + self.ang.y ** 2)

    def __len__(self):
        return abs(self.r)

    def __mul__(self, other):  # 向量数乘或叉乘
        if isinstance(other, Vector):
            return Vector(r=self.r * other.r * sin(abs(self.o - other.o)), o=0)
        elif isinstance(other, float):  # 向量数乘
            return Vector(r=self.r * other, o=self.o)
        elif isinstance(other, list) or isinstance(other, tuple):
            return Vector(r=self.r * other[0], o=self.o * other[1])

        else:
            raise ValueError("你输错了，向量不能和她相乘")

    def __ne__(self, other):
        assert isinstance(other, Vector), "必须是两个向量作比较"
        if self.r == other.r and self.o == other.o:
            return False
        else:
            return True

    def __eq__(self, other):
        assert isinstance(other, Vector), "输入一个向量来比较"
        return True if self.r == other.r and self.o == other.o else False

    def __getitem__(self, item):
        if item == 0:
            return self.r
        else:
            return self.o

    @staticmethod
    def AddBasicVector():  # 添加正交基X
        return Vector(1, 0)

    @staticmethod
    def AddZeroVector():  # 添加零向量
        return Vector(0, 0)

    @staticmethod
    def AddVerticalVector():  # 添加正交基Y
        return Vector(1, 0.5 * pi)

    @staticmethod
    def _IsVVector(name):
        if component.VVector.get(name) is not None:
            return True
        else:
            return False

    """判断这个实例化向量是不是速度向量"""

    def GetForm(self, other):
        if isinstance(other, list) or isinstance(other, tuple):
            return Vector(other[0], other[1])
        if hasattr(self, other):
            return other
        """把数组转化向量"""

    def _IsSpin(self, others: list):
        if self.o != others[0]:
            return True
        else:
            return False

    def Spin(self, other):
        if isinstance(other, Vector):
            self.r += other.r
            self.o += other.o

    def dot(self, other):
        """The dot product between the vector and other vector

        :return: The dot product
        点乘，算动能定理之类的
        """
        if isinstance(other, Vector):
            sx, sy = self.ang.x, self.ang.y
            otx, oty = other.ang.x, other.ang.y
            return sx * otx + sy * oty
        else:
            return self.ang.x * other[0] + self.ang.y * other[1]

    def _Get2VectorAngel(self, other):
        return self.o - other.o

    def _Get2VectorLength(self, other):
        return self.r - other.r

    def Reverse(self):
        return Vector(r=-self.r, o=-self.o)

    def ProperOrthogonalDecomposition(self):  # 正交分解法
        """:param:object(Vector)
        :return: 2 vertical Vector"""
        return Vector(r=self.r * cos(self.o * 3.14 / 180), o=0), Vector(r=self.r * sin(self.o * 3.14 / 180), o=90)

    def Standardization(self):
        """

        单位化

        """
        self.r = 1

    def Standard(self):
        return Vector(r=1, o=self.o)

    def _UpdateRO(self):
        self.r = sqrt(self.ang.x ** 2 + self.ang.y ** 2)
        self.o = arctan(self.ang.y / self.ang.o)


# 碰撞
"""每添加一个物体，就调用一次这个函数"""


# ————————————————————————————————————————————碰撞的运算————————————————————————————
def _PlanetCollide(solid):
    solid.v_vector -= 2 * solid.v_vector


def _IsCollide(item1, item2):  # 检测2个圆发生碰撞（通用）
    return True if sqrt(abs(item1.pos_x - item2.pos_x) ** 2 + abs(item1.pos_y - item2.pos_y) ** 2) <= max(
        item1.solid_round,
        item2.solid_round) else False


def _IsCollideSquare(item1, item2):
    count = 0
    """:param item1:solid
    :param item2:solid
    判断是都发生碰撞（2个正方形之间）"""
    for pos in item1.square_pos:
        a = _InSquare(item2.square_pos, pos)
        if a[0]:
            count += 1
        return True, a[1] if count >= 1 else False


def _CircleP2PCollide(circle1: object, circle2: object):
    """

    solid:param circle1
    solid:param circle2
    使用动量定理进行两个球体之间的碰撞运算
    自动交换球体速度等

    """
    if _IsCollide(circle1, circle2) is False:  # 不碰撞返回0
        return 0
    if abs(circle1.mass - circle2.mass) <= 1e-6:  # m1 = m2，速度交换
        circle1.v_vector, circle2.v_vector = circle2.v_vector, circle1.v_vector
    elif 1e-6 < abs(circle1.mass - circle2.mass) < 1e6:  # m1 !=m2,调用动量定理
        circle1.v_vector.r += ((circle1.mass - circle2) / (circle1.mass + circle2) * circle1.v_vector.r)
        circle2.v_vector.r += 2 * circle1.v_vector.r * circle1.mass / (circle1.mass + circle2.mass)
    else:
        if circle1.mass > circle2.mass:
            circle2.v_vector += -circle2.v_vector + circle1.v_vector
        else:
            circle1.v_vector += -circle1.v_vector + circle2.v_vector  # 质量远大于，忽略不计


def _CircleNP2PCollide(solid_circle1, solid_circle2):
    """

    两个球的非对心碰撞

    :param solid_circle1: Solid
    :param solid_circle2: Solid

    """
    epsilon = min(solid_circle1.friction, solid_circle2.friction)  # 设定碰撞系数
    c_vector = Vector(r=_Len(solid_circle1.now_pos, solid_circle2.now_pos), o=arctan(
        (solid_circle1.pos_y - solid_circle2.pos_y) / (solid_circle1.pos_x - solid_circle2.pos_y)))  # 获取相对方向向量
    c_vector.Standardization()  # 方向向量标准化
    save_r = solid_circle2.v_vector.r
    solid_circle2.v_vector.r = 0
    solid_circle1.v_vector += solid_circle2.v_vector  # 变换以m2的坐标系
    if abs(solid_circle1.mass - solid_circle2.mass) <= 1e-2:  # m1==m2时的近似
        solid_circle1.v_vector = solid_circle1.v_vector - c_vector * float(0.5 * (1 + epsilon) * (
            solid_circle1.v_vector.dot(c_vector)))  # 公式代入
        solid_circle2.v_vector = c_vector * float(0.5 * (1 + epsilon) * (
            solid_circle1.v_vector.dot(c_vector)))
        solid_circle1.v_vector -= solid_circle2.v_vector  # 变回原来参考系
    elif max(solid_circle1.mass, solid_circle2.mass) / min(solid_circle1.mass, solid_circle2.mass) >= 1e2:
        solid_circle1.v_vector = solid_circle1.v_vector - c_vector * 2 * (solid_circle1.v_vector.dot(c_vector))
        solid_circle1.v_vector -= solid_circle2.v_vector  # 变回原来参考系

    else:
        solid_circle1.v_vector = solid_circle1.v_vector - c_vector * ((1 + epsilon) * solid_circle2.mass / (
                solid_circle1.mass + solid_circle2.mass)) * (solid_circle1.v_vector.dot(c_vector))
        solid_circle2.v_vector = c_vector * ((1 + epsilon) * solid_circle2.mass / (
                solid_circle1.mass + solid_circle2.mass)) * (solid_circle1.v_vector.dot(c_vector))
        solid_circle1.v_vector -= solid_circle2.v_vector  # 变回原来参考系
    solid_circle2.v_vector.r = save_r


def _IsCentralCircleCollide(solid_circle1, solid_circle2):
    if abs((solid_circle1.v_vector - solid_circle2.v_vector).r - arctan(
            (solid_circle1.now_pos[1] - solid_circle2.now_pos[1]) / (solid_circle1.now_pos[1] + solid_circle2.now_pos[
                1]))) < 1e-4:
        return True
    else:
        return False


def CircleCollide(solid_circle1, solid_circle2):
    """

    碰撞模拟(两个圆)

    :param solid_circle1: solid
    :param solid_circle2: solid
    :return:None
    """
    if _IsCollide(solid_circle1, solid_circle2):
        if _IsCentralCircleCollide(solid_circle1, solid_circle2) and solid_circle1._isc is False:
            _CircleP2PCollide(solid_circle1, solid_circle2)
        elif solid_circle1._isc is False:
            _CircleNP2PCollide(solid_circle1, solid_circle2)
        solid_circle1._isc = True
        solid_circle2._isc = True

    else:
        solid_circle1._isc = False
        solid_circle2._isc = False


def LineCircleCollide(solid1, solid2):
    """

    线--圆碰撞

    :param solid1: Solid
    :param solid2: Solid
    :return: None
    """
    assert solid1.solid_type == "C" or solid2.solid_type == "C", "必须有一个圆"


def PointCollide(solid1, solid2):
    """点--线碰撞，适用于两个多边形
    原理：力矩以及动量定理
    Solid:param solid1:Solid
    Solid:param solid2:Solid"""
    upper_result = _IsCollideSquare(solid1, solid2)
    if not upper_result[0]:
        return 0  # 不碰撞返回0
    else:
        if solid1.solid_type == "S" and solid2.solid_pos == "S":
            pos = upper_result[1]
            for point in solid1.all_square_v_vector:
                point.o += solid2.mass * solid2.solid_round * _Len(pos, solid1.now_pos) / (
                        solid1.solid_round * solid1.mass)
                point.r += solid2.mass * solid2.v_vector[0].r * _Len(pos, solid1.now_pos) / (
                        solid1.solid_round * solid1.mass)
            for point in solid2.all_square_v_vector:
                point.o += solid1.mass * solid1.solid_round * _Len(pos, solid2.now + pos) / (
                        solid2.solid_round * solid2.mass)
                point.r += solid1.mass * solid1.v_vector[0].r * _Len(pos, solid2.now_pos) / (
                        solid2.solid_round * solid2.mass)
        else:
            if solid2.solid_type == "C":
                solid1, solid2 = solid2, solid1
            elif solid1.solid_type == "C":
                solid1, solid2 = solid1, solid1
            m1 = solid1.mass
            m2 = solid2.mass
            l1 = solid1.radio
            l2 = solid2.radio
            ...


def _GetCCDSolid(solid):
    solid_tree = QuadTree(pos_x=solid.pos_x, pos_y=solid.pos_y, level=10, max_levels=2047)
    solid_tree.LoadSolid()
    return solid_tree.Retrieve(solid)


class Shape(object):

    def __init__(self, solid):
        import SimpleGeo
        assert solid.solid_type == "O", "输入非圆非矩形来创建多边形形状实例"
        self.poly_point = solid.init_all_solid_point
        self.solid = solid
        self.kernel = None
        self.solid_type = 0 if solid.solid_type == "S" else 1
        self.space = SimpleGeo.space.LinearSpace()
        self.shape_rect = None

    def GetKernel(self):
        """


        计算出运算核 参考文献:https://www.doc88.com/p-1905424832385.html?s=rel&id=1

        :return: kernel(list)
        """
        if self.solid_type == 0:
            return self.solid.square_pos
        else:
            temp_len = []
            for X in range(0, len(self.solid.init_all_solid_point) - 1):
                temp_len.append(_Len(self.solid.init_all_solid_point[X], self.solid.init_all_solid_point[X + 1]))
            max_length_side = [self.solid.init_all_solid_point[temp_len.index(max(temp_len))],
                               self.solid.init_all_solid_point[temp_len.index(max(temp_len)) + 1]]
            max_length_side_angel = arctan(
                (max_length_side[0][1] - max_length_side[1][1]) / (max_length_side[0][0] - max_length_side[1][0]))
            spinning_solid_points = _SpinMatrix(max_length_side_angel,
                                                self.solid.init_all_solid_point)  # 把图像按照最长边旋转至平行于xy轴
            point1 = [min([i[0] for i in spinning_solid_points]), max(i[1] for i in spinning_solid_points)]
            point2 = [max([i[0] for i in spinning_solid_points]), max(i[1] for i in spinning_solid_points)]
            point3 = [min([i[0] for i in spinning_solid_points]), min(i[1] for i in spinning_solid_points)]
            point4 = [max([i[0] for i in spinning_solid_points]), min(i[1] for i in spinning_solid_points)]
            kernel = _SpinMatrix(-max_length_side_angel, [point1, point2, point3, point4])
            self.shape_rect = self.space.SetRect(kernel, self.solid.name + "kernel")
            self.kernel = kernel
            return kernel

    @property
    def kernel_size(self):
        return self.shape_rect.__sizeof__()

    @property
    def now_pos(self):
        return _SpinMatrix(self.solid.spin_angel, self.poly_point)

    def pos(self, point):
        return self.now_pos[point]


def CCD(solid):
    """

    CCD ConsecutiveCollideDetection
    连续碰撞检测及实现
    :param solid:Solid
    :return: Bool

    """
    ccd_solid = _GetCCDSolid(solid)
    for single_solid in ccd_solid:
        if _IsCollideSquare(solid.shape.kernel, single_solid.shape.kernel):
            PointCollide(single_solid, solid)


def SquareBound(solid):
    """:param solid
    判断solid是否与世界边界（底面）碰撞"""
    import numpy as NP
    square_pos = NP.array(solid.square_pos)
    if NP.min(square_pos[..., 1:2]) <= component.world.floor:
        for vector in solid.all_square_v_vector:
            vector.Reverse()


# ————————————————————————————————————————流体计算——————————————————————————————————————


class _LCalculate(Liquids.Liquid):
    """水分为两种状态，一种是触发状态，另一种是原始状态。
    触发状态是指固体进入水的过程中的一种状态，可以进行微分计算
    ，而原始状态是以静态形式进行运算的一种状态"""

    def __init__(self):
        super(_LCalculate, self).__init__()
        self.x_len = abs(self.point[0][0] - self.point[0][1])
        self.y_len = abs(self.point[1][0] - self.point[1][1])  # 初始长度
        self.mass = self.density * self.x_len * self.y_len  # 获取总质量，用来做整体运算

    def Trigger(self):
        """触发检测器，检测有没有物体已经进入流体
        :return bool"""
        boundary_count = 0
        for solid in component.SolidList:
            if solid.solid_type == "S":  # 固体为矩形，使用矩形检测器
                for X in solid.square_pos:
                    if _InSquare(self.point, X):  # 遍历正方形和水之间的位置关系，确定是否进入水中
                        self.status = 1  # 改变状态参量，启动流体边界
                        solid.in_liquid = 1  # 在流体内
                        return True
            if solid.solid_type == "C":
                for liquid_pos in self.point:  # 遍历四个点
                    if _InCircle(solid, liquid_pos):
                        self.status = 1  # 启用流体边界
                        boundary_count += 1
                if boundary_count == 4:
                    solid.in_liquid = 1
                    return True
                elif 1 <= boundary_count <= 4:
                    solid.in_liquid = 0
                    return True
                else:
                    solid.in_liquid = 0
                    return False
            if solid.solid_type == "O":
                for point in solid.poly_pos:
                    if _InSquare(point, self.point):
                        self.status = 1
                        boundary_count += 1
                if boundary_count == len(solid.poly_pos):
                    solid.in_liquid = 1
                    return True
                elif 1 <= boundary_count <= len(solid.poly_pos):
                    solid.in_liquid = 0
                    return True
                else:
                    return False
        return False

    def GetStaticBuoyancy(self):  # 计算此时的静态浮力
        """自动返回物体在流体内部的浮力
        :return None"""
        from component import InitPhyConfig
        for solid in component.SolidList:
            if solid.in_liquid == 1:
                if solid.solid_type == "C":
                    solid.a_vector += Vector(self.density * InitPhyConfig[0].gravity * pi * solid.solid_round ** 2,
                                             90)  # 圆形浮力公式
                if solid.solid_type == "S":
                    solid.a_vector += Vector(
                        self.density * InitPhyConfig[0].gravity * solid.side_length ** 2, 90)  # 正方形浮力公式
                if solid.solid_type == "O":
                    solid.a_vector += Vector(self.density * InitPhyConfig[0].gravity * solid.shape.kernel_size, 90)

    def UpdateDynamicBuoyancy(self):  # 计算此时动态浮力
        """"思路：把物体分成圆形和矩形，圆形使用简单几何学来计算浮力
        ，而矩形则对底面积分求出压力差"""
        from SimpleGeo import LinearSpace
        space = LinearSpace()
        liquid_space = space.SetRect(position_point=self.point, name="Geo%s" % self.name)
        if len(component.SolidList) == 0:
            return 0
        for solid in component.SolidList:
            if solid.in_liquid == 1:
                if solid.solid_type == "C":
                    # abolished code 废弃代码段，改为调用SimpleGeo库
                    # if solid.pos_y >= self.point[0][1]:  # 只有上半部分进入圆（如果是从上面进入，没错，如果从其他方向进入，也没错，因为比较的点是固定的）
                    # theta = 2 * (pi - arccos((solid.pos_y - self.point[0][1]) / solid.solid_round))
                    #  solid.a_vector += Vector(self.density * InitPhyConfig[0].gravity * (
                    #     theta * solid.solid_round - 0.5 * solid.solid_round ** 2 * sin(
                    #   theta)), o=0.5 * pi)  # 根据公式求出实时浮力，返回浮力向量
                    # if self.point[0][1] >= solid.pos_y >= self.point[3][1]:  # 浸没大部分
                    #  theta = 2 * arccos(solid.pos_y - self.point[0][1])
                    #  solid.a_vector += Vector(
                    #     0.5 * pi * solid.solid_round ** 2 * InitPhyConfig[
                    #      0].gravity * self.density + self.density *
                    #  InitPhyConfig[0].gravity * (
                    #          theta * solid.solid_round - 0.5 * solid.solid_round ** 2 * sin(
                    #      theta)), o=0.5 * pi)  # 根据公式求出实时浮力，返回浮力向量

                    #  if solid.solid_type == "S" or solid.solid_type == "O":
                    #      max_point_x = max(solid.pos1[0], solid.pos2[0], solid.pos3[0],
                    #                        solid.pos4[0])  # 定义边界点,分别是左边和右边，以进行实时浮力演算
                    #  min_point_x = min(solid.pos1[0], solid.pos2[0], solid.pos3[0],
                    #       solid.pos4[0])
                    # min_point_y = min(solid.pos1[0], solid.pos2[0], solid.pos3[0],
                    #  solid.pos4[0])  # 从上下进入
                    #    if solid.pos_y >= self.point[0][1] and self.point[0][0] <= solid.pos_x <= self.point[1][0] or \
                    #      self.point[3][1] <= solid.pos_y <= self.point[3][1] and solid.pos_x < self.point[0][0] or \
                    #    self.point[3][1] <= solid.pos_y <= self.point[3][1] and solid.pos_x >= self.point[2][0]:
                    #    sub_pressure = 0.5 * InitPhyConfig[0].gravity * self.density * (
                    #   min(min(solid.square_pos[0]) - self.point[0][1],
                    #     min(solid.square_pos[0]) - self.point[1][1],
                    #     min(solid.square_pos[0]) - self.point[2][1],
                    #      min(solid.square_pos[0]) - self.point[3][1]))  # 求浸入深度（浸入点最小值的之差的最大值）
                    #   solid.a_vector += Vector(sub_pressure, 0.5 * pi)
                    #    solid.a_vector.o += arcsin((solid.pos1[0] - solid.pos2[0]) / _Len(solid.pos1, solid.pos2))
                    # 利用矩形两边平行，等价转化角度
                    # 从左到右进入
                    # if self.point[0][0] <= max_point_x <= self.point[1][0] and self.point[3][1] <= max_point_y <= \
                    #  self.point[0][1] or self.point[0][0] <= min_point_x <= self.point[1][0] and self.point[3][
                    # 2] <= min_point_y <= \
                    #   self.point[0][1]:
                    #  sub_pressure = 0.5 * InitPhyConfig[0].gravity * self.density * (
                    #         max_point_y + min_point_y) * 0.5  # 这里用平均值近似面积
                    #   solid.a_vector += Vector(sub_pressure, 0.5 * pi)
                    # solid.a_vector.o += arcsin((solid.pos1[0] - solid.pos2[0]) / _Len(solid.pos1, solid.pos2))
                    circle = space.SetCircle(r=solid.solid_round, o_point=solid.now_pos, name=solid.name)
                    square_ = space.GetCrossSquare(liquid_space, circle, "and")
                    liquid_force = self.physics.gravity * self.density * square_
                    solid.a_vector += Vector(r=liquid_force / solid.mass, o=90)

    def UpdateFriction(self):  # 获取水的摩擦力
        for solid in component.SolidList:
            if solid.in_liquid == 1:
                if solid.solid_type == "C":
                    solid.a_vector += Vector(
                        -800 * AirForce(0.5, self.density, pi * solid.solid_round ** 2, solid.v_vector.r),
                        -solid.v_vector.o)  # 水的阻力为空气800倍
                if solid.solid_type == "S" or solid.solid_type == "O":
                    if _InSquare(self.point, solid.square_pos[0]):
                        solid.a_vector += Vector(
                            r=(800 * AirForce(0.5, self.density, solid.side_length, solid.v_vector1.r / solid.mass)),
                            o=-solid.v_vector1.o)


# ——————————————————————————————————————————————天体计算————————————————————————————————————————————————————


def PlanetForce(MASS, mass, r):
    return 6.67e-11 * MASS * mass / r ** 2  # F引＝GMm/r^2


def InShellBall(solid, planet):  # 判断物体是否在希尔球内，从而确定物体是否受到引力
    if sqrt((solid.pos_x - planet.planet_pos_x) ** 2 + (
            solid.pos_y - planet.planet_pos_y) ** 2) <= planet.planet_boundary:
        return True
    else:
        return False


def UpdateGravitationalAttractionVector(solid, planet):
    if not InShellBall(solid, planet):
        return 0
    else:  # ___________________↓这个r是半径的意思
        gravity_vector = Vector(r=PlanetForce(planet.mass, 1, r=sqrt((solid.pos_x - planet.planet_pos_x) ** 2 + (
                solid.pos_y - planet.planet_pos_y) ** 2)),
                                o=arctan((solid.pos_y - planet.planet_pos_y) / (
                                        solid.pos_x - planet.planet_pos_x)))
        if solid.solid_type == "C":
            solid.a_vector += gravity_vector  # 添加引力的向量
        if solid.solid_type == "S":
            solid.a_vector += gravity_vector


def LG_calculate(planet1, planet2):
    """:param:object:planet
    :return:list or tuple"""
    R = sqrt(((planet1.planet_pos_x - planet2.planet_pos_x) ** 2 + (
            planet1.planet_pos_y - planet2.planet_pos_y) ** 2))  # 两星体距离
    angle_ = arctan(
        (planet1.planet_pos_y - planet2.planet_pos_y) / abs(planet1.planet_pos_x / planet2.planet_pos_x))  # 获取和正交坐标的角度
    if R >= planet1.planet_boundary + planet2.planet_boundary:

        return 0  # 如果位于希尔球外，则没有拉格朗日点
    else:  # 获取以两球质心连线为X轴的坐标的拉格朗日点
        a = planet2.mass / (planet1.mass + planet2.mass)  # 拉格朗日点公式中a参数
        L1 = [R * (1 - (a / 3) ** 0.33), 0]  # L1拉格朗日点
        L2 = [R * (1 + (a / 3) ** 0.33), 0]  # L2拉格朗日点
        L3 = [-R * (1 + 5 * a / 12), 0]  # L3拉格朗日点
        L4 = [0.5 * R * (planet1.mass - planet2.mass) / planet1.mass + planet2.mass, R * (3 ** 0.5) / 2]  # L4拉格朗日点
        L5 = [-0.5 * R * (planet1.mass - planet2.mass) / planet1.mass + planet2.mass, -R * (3 ** 0.5) / 2]  # L5拉格朗日点
        angle_ = -angle_  # 旋转矩阵角度应该为负的（因为要旋转回去）
        lg_point = [L1, L2, L3, L4, L5]  # 生成一个拉格朗日点的二位列表
        lg_point = _SpinMatrix(angle_, lg_point)
        lg_point[4][1] = -lg_point[4][1]
    return lg_point

    # def OrbitController(planet, solid):
    # """:param planet, solid
    # 给出一个初速度，求这个初速度下物体围绕行星运动的近似方程(运算步数越多，精度越高)
    #  return 函数方程"""

    # x = solid.now_pos
    # ...


def _GetLaPoint(planet1, planet2, point_type):
    """

        :param planet2: Planet
        :param planet1: Planet
        :param point_type:int or list or tuple


    """
    if point_type is None:
        point_type = -1
    la_result = planet1.Lagrangianpoint(planet2)
    if isinstance(point_type, int):
        if point_type == -1:
            return la_result
        if 1 <= point_type <= 5:
            return la_result[point_type]
        if point_type >= 5:
            raise ValueError("请输入1-5之间的整数，输入返回对应拉格朗日点")
    if isinstance(point_type, list or tuple):
        if len(point_type) == 2:
            return la_result[point_type[0], point_type[1]]
        else:
            raise ValueError(r"请输入2个数，比如L2--L4就输入[2,4]")
    else:
        raise ValueError("请检查输入得是否正确，不能求出拉格朗日点")


def GetGravitationForce(planet, solid):
    """

    :param planet Planet
    :param solid Solid
    :return Vector 返回一个加速度向量

    """
    return Vector(r=PlanetForce(planet.mass, mass=1, r=_Len([planet.planet_pos_x, planet.planet_pos_y], solid.now_pos)))


# ——————————————————————————————————————————————场相关————————————————————————————————————————————


def _Recovery():
    """

    重置配置信息，用于读取新的信息

    """
    component.SolidList = []
    component.LiquidList = []
    component.PlanetList = []
    component.FieldList = []
    component.FieldDict = {}
    component.SolidDict = {}
    component.PlanetDict = {}
    component.LiquidDict = {}


ele_count = 0
"""

下面是一些演示场，或者叫预置场，预置了一些场参与运算，可以添加新的

"""


class Field(abc.ABC, metaclass=abc.ABCMeta):
    """

    定义场抽象类，限制场的属性和方法
    抽象属性：
    1.field_type:
    根据场论，场分为标量场和向量场
    2.pos:
    根据场内固体所在的位置，获得一系列信息（比如势等）
    场参与运算的方法：根据实体位置，赋予每个实体一个加速度（时间轴上在动力学之后，与水的运算并行）

    """

    @property
    @abc.abstractmethod
    def field_type(self):
        """

        场类型，1为标量场，2为向量场

        """
        pass

    @property
    @abc.abstractmethod
    def field_effect(self):
        """


        场作用的范围，默认为Solid，可以是World


        """
        pass

    @property
    @abc.abstractmethod
    def AffectAttribute(self):
        """

        定义场影响Solid的属性。
        向量场："Speed"：速度场，
        ”Accelerate“：加速度场
        ”pos“：非惯性系
        ”Electric“：电场，
        ”Magnetic”：磁场
        "Wind"：风 场
        标量场：
        “Heat”：温度场
        “Density”：空气密度场

        """
        pass

    @abc.abstractmethod
    def GetPotential(self, pos, pos_zero):
        """

        获得场的势(基准点和当前点)


        :return num

        """
        pass

    @abc.abstractmethod
    def SetValue(self, solid):
        """

        根据位置和物体属性获取场对应的值
        例如：电场---电场力
              重力场--引力
              温度场--温度
        :return  num

        """
        pass

    @abc.abstractmethod
    def GetValue(self, pos):
        """

        根据位置获取此时场的数值（向量或数字）

        :param pos: list or tuple
        :return: Vector
        """


class ElectricField(Field):
    """

    电场类

    """

    def __init__(self, field_vector, name=None):
        global ele_count
        assert isinstance(field_vector, Vector), "请输入电场向量来生成匀强电场，r为场强，o为方向！"
        self.affect_type = "Electric"
        self._field_type = 1
        self._field_effect = "Solid"
        self._electronic_strength = field_vector.r
        self._electronic_direction = field_vector.o
        self.name = name if name is not None else "未命名电场%d" % ele_count
        ele_count += 1

    def __add__(self, other):
        assert isinstance(other, ElectricField), "请输入电场来相加"
        return ElectricField(field_vector=Vector(r=self._electronic_strength + other._electronic_strength,
                                                 o=self._electronic_direction + other._electronic_direction))

    def __sub__(self, other):
        assert isinstance(other, ElectricField), "请输入电场来相减"
        return ElectricField(field_vector=Vector(r=self._electronic_strength - other._electronic_strength,
                                                 o=self._electronic_direction - other._electronic_direction))

    def __getitem__(self, item):
        if item == 0:
            return self._electronic_strength
        if item == 1:
            return self._electronic_direction
        else:
            raise ValueError("输入1或者0来选取场的参数")

    def GetPotential(self, pos, pos_zero):
        """
        获得电势

        :param pos: list or tuple
        :param pos_zero: list or tuple
        :return: U
        """
        return _Len(pos, pos_zero) * self._electronic_strength

    def SetValue(self, solid):
        for solid in component.SolidList:
            solid.a_vector += Vector(r=self._electronic_strength * solid.electronic / solid.mass,
                                     o=self._electronic_direction)

    def GetValue(self, solid):
        return Vector(r=self._electronic_strength * solid.electronic / solid.mass,
                      o=self._electronic_direction)

    @property
    def AffectAttribute(self):
        return self.affect_type

    @property
    def field_type(self):
        return self._field_type

    @property
    def field_effect(self):
        return self._field_effect


class MagneticField(Field):
    """

    磁场类

    """

    def __init__(self, field_vector, field_name):
        assert isinstance(field_vector, Vector), "请输入磁场向量来生成匀强磁场，r为磁场强度H，o为方向！"
        self.vector = field_vector
        self.H = field_vector.r
        self.direction = field_vector.o
        self.miu = 1
        self.B = self.H * self.miu
        self.field_name = field_name

    def __add__(self, other):
        assert isinstance(other, MagneticField), "请输入两个磁场来叠加"
        return MagneticField(field_vector=self.vector + other.vector,
                             field_name=self.field_name + "+" + other.field_name)

    def __sub__(self, other):
        assert isinstance(other, MagneticField), "请输入两个磁场来相减"
        return MagneticField(field_vector=self.vector - other.vector,
                             field_name=self.field_name + "-" + other.field_name)

    def __getitem__(self, item):
        if item == 0:
            return self.B
        if item == 1:
            return self.direction
        else:
            raise ValueError("输入1或者0来选取场的参数")

    @property
    def AffectAttribute(self):
        return 1

    @property
    def field_type(self):
        return "Magnetic"

    @property
    def field_effect(self):
        return "Solid"

    def GetPotential(self, pos, pos_zero):
        return _Len(pos, pos_zero) * self.H

    def SetValue(self, solid):
        for solid in component.SolidList:
            solid.a_vector += Vector(r=solid.mag_material * self.B, o=self.direction)

    def GetValue(self, pos):
        return self.B

    def GetLorenzForce(self, solid):
        return Vector(r=solid.electronic * solid.v_vector.r * self.B, o=solid.v_vector.r + 90)


class WindField(Field):
    """

    风场类(匀速)

    """

    def __init__(self, field_vector, field_name):
        assert isinstance(field_vector, Vector), "输入风向量来创建风场，r为风速，o为风向"
        self.wind_speed = field_vector.r
        self.wind_direction = field_vector.o
        self.field_name = field_name
        self.vector = field_vector

    @property
    def AffectAttribute(self):
        return 1

    @property
    def field_type(self):
        return "Wind"

    @property
    def field_effect(self):
        return "Solid"

    def __add__(self, other):
        assert isinstance(other, WindField), "请输入两个磁场来叠加"
        return MagneticField(field_vector=self.vector + other.vector,
                             field_name=self.field_name + "+" + other.field_name)

    def __sub__(self, other):
        assert isinstance(other, WindField), "请输入两个磁场来相减"
        return MagneticField(field_vector=self.vector - other.vector,
                             field_name=self.field_name + "-" + other.field_name)

    def GetPotential(self, pos, pos_zero):
        return 0

    def GetValue(self, pos):
        return self.wind_speed

    def GetWindForce(self, solid):
        return AirForce(C=0.5, density=component.world.physics.density, speed=solid.v_vector.r + self.wind_speed,
                        wind_square=solid.solid_round)

    def SetValue(self, solid):
        for solid in component.SolidList:
            solid.a_vector += Vector(r=self.GetWindForce(solid), o=-solid.v_vector.o) * 0.001


class HeatField(Field):
    """

    温度场

    """

    def __init__(self, field_vector, field_name):
        self.heat = field_vector[0]
        self.name = field_name

    def __add__(self, other):
        if isinstance(self, other):
            return HeatField(field_vector=self.heat + other.heat, field_name=self.name + "+" + other.heat)
        elif hasattr(other, "__getitem__"):
            return HeatField(field_vector=self.heat + other[0], field_name=self.name)
        elif isinstance(other, float) or isinstance(other, int):
            return HeatField(field_vector=self.heat + other, field_name=self.name)
        else:
            raise ValueError("请输入数字，数组或场来相加")

    def __sub__(self, other):
        if isinstance(self, other):
            return HeatField(field_vector=self.heat - other.heat, field_name=self.name + "-" + other.heat)
        elif hasattr(other, "__getitem__"):
            return HeatField(field_vector=self.heat - other[0], field_name=self.name)
        elif isinstance(other, float) or isinstance(other, int):
            return HeatField(field_vector=self.heat - other, field_name=self.name)
        else:
            raise ValueError("请输入数字，数组或场来相减")

    @property
    def AffectAttribute(self):
        return "World"

    @property
    def field_type(self):
        return "Heat"

    @property
    def field_effect(self):
        return "World"

    def GetValue(self, pos):
        return component.world.temperature + self.heat - 6 * int(component.world.atom_height / 1000) * pos[1]

    def GetPotential(self, pos, pos_zero):
        return 0

    def SetValue(self, solid):
        raise TypeError("标量场不能给solid进行设置")


class DensityField(Field):
    """

    空气密度场，向量格式r：海拔为0处空气密度o:高度减小率

    """

    def __init__(self, field_vector, field_name):
        self.density = field_vector.r
        self.rate = field_vector.o
        self.name = field_name

    @property
    def AffectAttribute(self):
        return "World"

    @property
    def field_type(self):
        return "Density"

    @property
    def field_effect(self):
        return "World"

    def GetValue(self, pos):
        return component.world.temperature + self.density - pos[1] * self.rate

    def GetPotential(self, pos, pos_zero):
        return 0

    def SetValue(self, solid):
        raise TypeError("标量场不能给solid进行设置")


class TransSpeedField(Field):
    """

       速度变化场

       """

    def __init__(self, field_vector, field_name):
        assert isinstance(field_vector, Vector), "输入场向量"
        self.field_vector = field_vector
        self.name = field_name

    def __add__(self, other):
        if isinstance(other, TransSpeedField):
            return TransSpeedField(field_vector=self.field_vector + other.field_vector,
                                   field_name=self.name + other.name)
        elif isinstance(other, iterable):
            return TransSpeedField(
                field_vector=Vector(r=self.field_vector[0] + other[0], o=self.field_vector[1] + other[1]),
                field_name=self.name)

    def __sub__(self, other):
        if isinstance(other, TransSpeedField):
            return TransSpeedField(field_vector=self.field_vector - other.field_vector,
                                   field_name=self.name - other.name)
        elif isinstance(other, iterable):
            return TransSpeedField(
                field_vector=Vector(r=self.field_vector[0] - other[0], o=self.field_vector[1] - other[1]),
                field_name=self.name)

    @property
    def AffectAttribute(self):
        return "Solid"

    @property
    def field_type(self):
        return "Speed"

    @property
    def field_effect(self):
        return "World"

    def GetValue(self, pos):
        ...

    def GetPotential(self, pos, pos_zero):
        return 0

    def SetValue(self, solid):
        raise TypeError("标量场不能给solid进行设置")


class OtherField(Field):
    """

    其他自定义场

    """

    def __init__(self, field_vector, field_type, field_name: str):
        self.able = 0  # 不激活
        self.function = None
        assert field_name.find("Field") != -1, "场名必须带有Field"
        self.name = field_name
        for name in component.default_field_set:
            if name == field_name:
                raise ValueError("与已有场重名")
            if field_type == 0:
                self.field_value = field_vector
            elif field_type == 1:
                self.vector = field_vector
                self.field_value = field_vector.r
                self.field_direction = field_vector.o
        self._field_type_ = field_type

    def GetFunction(self, func):
        """

        获取场方程（根据位置确定值或向量）

        :param func: lambda x,y or x:(some functions)

        """
        assert callable(func), "请输入lambda函数来激活场"
        self.function = func
        self.able = 1

    @property
    def field_type(self):
        return self._field_type_

    @property
    def AffectAttribute(self):
        if self.field_type == 1:
            return "World"
        else:
            return "Solid"

    @property
    def field_effect(self):
        return self.name + "effect"

    def __add__(self, other):
        if self.field_type == 0 and isinstance(other, OtherField) and other.field_type == 0:
            return OtherField(field_vector=self.field_value + other.field_value, field_type=0,
                              field_name=self.name + "+" + other.name)
        elif self.field_type == 1 and isinstance(other, OtherField):
            return OtherField(field_vector=self.vector + other.vector, field_type=1,
                              field_name=self.name + "+" + other.name)
        elif self.field_type == 0 and (isinstance(other, list) or isinstance(other, tuple)):
            return OtherField(field_vector=Vector(r=self.vector[0] + other[0], o=self.vector[1] + other[1]),
                              field_name=self.name, field_type=self.field_type)
        elif self.field_type == 0 and isinstance(self, float):
            return

    def __sub__(self, other):
        if self.field_type == 0 and isinstance(other, OtherField) and other.field_type == 0:
            return OtherField(field_vector=self.field_value - other.field_value, field_type=0,
                              field_name=self.name + "-" + other.name)
        if self.field_type == 1 and isinstance(other, OtherField):
            return OtherField(field_vector=self.vector - other.vector, field_type=1,
                              field_name=self.name + "-" + other.name)

    def __getattribute__(self, item):
        if item.find("__") == -1 and self.able == 0 and item != "GetFunction":
            raise AttributeError("场%s没有激活，请使用GetFunction输入一个函数来激活它" % self.name)

    def GetValue(self, pos):
        if self.field_type == 0:
            return self.function(pos)
        else:
            return self.function(pos[0], pos[1])

    def GetPotential(self, pos, pos_zero):
        if self.field_type == 0:
            return 0
        else:
            return _Len(pos, pos_zero) * self.GetValue(pos)

    def SetValue(self, solid):
        if self.field_type == 0:
            raise EnvironmentError("不能把标量场设置给Solid")
        if self.field_type == 1:
            solid.a_vector += Vector(r=self.GetValue(pos=solid.now_pos), o=self.vector.o)


field_d = {"Wind": WindField, "Heat": HeatField}


def _AutoSelectFieldName(field_type, filed_data, name):
    """

    返回名字对应的场（根据名称自动选择）

    :param field_type: str
    :param filed_data: Vector
    :param name: str
    :return:Field


    """
    return field_d.get(field_type)(field_vector=filed_data, field_name=name)
