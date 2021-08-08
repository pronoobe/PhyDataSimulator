from math import *

import sympy

from support import _Len, _SpinMatrix, _2Rect, _CircleRect, _2Circle

# _________________________________Private_________________________________

_Len, _SpinMatrix, _2Rect, _CircleRect, _2Circle = _Len, _SpinMatrix, _2Rect, _CircleRect, _2Circle


# _________________________________Public__________________________________
class Point(object):
    def __init__(self, pos, name=None):
        self._pos_x, self._pos_y = pos
        self._pos = pos
        self._x = sympy.Symbol("x")
        self._y = sympy.Symbol("y")
        self.name = name if name.find("Point") != -1 else name + "Point"
        self.has_square = 0

    def __eq__(self, other):
        if self.pos == other.pos:
            return True
        else:
            return False

    def __ne__(self, other):
        if not self == other:
            return True
        else:
            return False

    def __getitem__(self, item):
        if 0 <= item <= 1:
            return self.pos[item]
        else:
            raise ValueError("输入0-1，0为x，1为y")

    def __mul__(self, other):
        return 0

    def __add__(self, other):
        if isinstance(other, Line) or isinstance(other, Point):
            return 0
        else:
            return other.__sizeof__()

    def __sub__(self, other):
        return self.__add__(other)

    def __sizeof__(self):
        return 0

    @property
    def pos(self):
        return self._pos

    @property
    def pos_x(self):
        return self._pos_x

    @property
    def pos_y(self):
        return self._pos_y

    def Len(self, other):
        return _Len(self, other)

    def Spin(self, angel):
        return _SpinMatrix(angel, self._pos)


class Line(object):
    def __init__(self, point, name=None):
        """
        :param point: Point or list or tuple
        """
        self._point1, self._point2 = point
        self._x1, self._y1 = point[0]
        self._x2, self._y2 = point[1]
        self._k = (self._y2 - self._y1) / (self._x2 - self._x1) if self._x2 - self._x1 != 0 else inf  # 初始斜率
        self._b = self._y1 - self._x1
        self.boundary = [-inf, inf]
        self._x = sympy.Symbol("x")
        self.name = name if name.find("Line") != -1 else name + "Line"
        self.has_square = 0

    def __sizeof__(self):
        return 0

    def __len__(self):
        return sqrt((self._x2 - self._x1) ** 2 + (self._y2 - self._y1) ** 2)

    def __str__(self):
        return "y-(%.2f)=%.2f(x-(%.2f))" % (self._y1, self.k, self._x1)

    def __mul__(self, other):
        return 0

    def __add__(self, other):
        if isinstance(other, Line) or isinstance(other, Point):
            return 0
        else:
            return other.__sizeof__()

    def __sub__(self, other):
        return self.__add__(other)

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    @property
    def y1(self):
        return self._x1

    @property
    def y2(self):
        return self._y2

    @property
    def point1(self):
        return self._x1, self.y1

    @property
    def point2(self):
        return self.x2, self.y2

    def get_y(self, x):
        return self.k * (x - self._x1) + self._y1 if self.k != inf else inf

    def get_x(self, y):
        return (y - self._y1) / self.k + self._x1 if self.k != 0 else inf

    @property
    def x_len(self):
        return self.get_x(0)

    @property
    def y_len(self):
        return self.get_y(0)

    @property
    def k(self):
        return (self._y2 - self._y1) / (self._x2 - self._x1) if self._x2 - self._x1 != 0 else inf

    def Boundary(self):
        return self.boundary

    def OnLine(self, *pos):
        """

        判断点是否在直线上

        :param pos: position
        :return: Bool
        """
        result = []
        for point in pos:
            if self.k != inf:
                if abs(self.get_y(point[0]) - point[1]) <= 1e-6 and self.boundary[0] <= point[0] <= self.boundary[1]:
                    result.append(True)
                else:
                    result.append(False)
            else:
                if point[0] == self.get_x(0):
                    result.append(True)
                else:
                    result.append(False)
        return result

    def _ClosetOnlinePoint(self, point):
        """

        求线段外一点距离线段上最近的一点坐标（如果垂足在范围内，就返回垂足，否则返回最近点）

        :param point:
        :return:Point
        """
        a = self._k
        b = -1
        c = self._y1 - self._k * self._x1
        result_point = [((b * b * point[0] - a * b * point[1] - a * c) / (a * a + b * b),
                         (a * a * point[1] - a * b * point[0] - b * c) / (a * a + b * b))]
        if self.OnLine(result_point)[0] is True:
            return result_point
        else:
            pos1 = [self.boundary[0], self.get_y(self.boundary[0])]
            pos2 = [self.boundary[1], self.get_y(self.boundary[1])]
            len1 = _Len(pos1, result_point)
            len2 = _Len(pos2, result_point)
            if len1 >= len2:
                return pos1
            else:
                return pos2

    def SwapXY(self):
        self._x1, self._x2, self._y1, self._y2 = self._x2, self._x1, self._y2, self._y1

    def Boundarylization(self, *boundary):
        """
        边界化
        :param boundary: 取值范围的X坐标

        """
        self.boundary = [min(*boundary), max(*boundary)]


class Circle(object):
    def __init__(self, pos, radio, name=None):
        self.pos = pos
        self.radio = radio
        self._x = sympy.Symbol("x")
        self._y = sympy.Symbol("y")
        self.name = name if name.find("Circle") != -1 else name + "Circle"
        self.has_square = 1

    def __sizeof__(self):
        return pi * self.radio ** 2

    def __len__(self):
        return 2 * pi * self.radio

    def __str__(self):
        return "x**2+y**2=%.2f" % self.radio ** 2

    def __add__(self, other):
        """
        取并集面积
        :param other: shape
        :return:
        """
        if isinstance(other, Circle):
            return _2Circle(self, other, "or")
        elif isinstance(other, Rect):
            return _CircleRect(self, other, "or")
        else:
            return 0

    def __mul__(self, other):
        """

        取交集面积

        :param other:
        :return: float
        """
        if isinstance(other, Circle):
            return _2Circle(self, other, "and")
        elif isinstance(other, Rect):
            return _CircleRect(self, other, "and")
        else:
            return 0

    def __sub__(self, other):
        """
        返回异或面积
        :param other:
        :return: float
        """
        if isinstance(other, Circle):
            return _2Circle(self, other, "xor")
        elif isinstance(other, Rect):
            return _CircleRect(self, other, "xor")
        else:
            return 0

    def __radd__(self, other):
        return self.__add__(other)

    def get_x(self, y):
        return sympy.solve(self._x ** 2 + y ** 2 - self.radio ** 2, self._x)

    def get_y(self, x):
        return sympy.solve(self._y ** 2 + x ** 2 - self.radio ** 2, self._y)

    def PointRelation(self, pos) -> str:
        if _Len(pos, self.pos) == self.radio:
            return "on"
        elif _Len(pos, self.pos) > self.radio:
            return "out"
        else:
            return "in"

    def LineRelation(self, line) -> str:
        res_x = sympy.solve(self._x ** 2 + (line._k * self._x + line._b) ** 2 - self.radio ** 2,
                            self._x)
        # res_x[0] = list(res_x[0])
        # res_x[1] = list(res_x[1])
        if len(res_x) > 1 and isinstance(res_x[0], sympy.core.numbers.RealNumber) and (
                (line.boundary[0] <= res_x[0] <= line.boundary[1]) or (
                line.boundary[0] <= res_x[1] <= line.boundary[1])):
            return "across"
        elif isinstance(res_x[0], sympy.core.numbers.RealNumber):
            return "tangent"
        else:
            return "separate"

    def LineCrossPos(self, line):
        if self.LineRelation(line) == "separate":
            return False
        else:
            res_x = sympy.solve(self._x ** 2 + (line._k * self._x + line._b) ** 2 - self.radio ** 2,
                                self._x)
            if len(res_x) > 1:
                res_x = [[float(res_x[0]), self.get_y(float(res_x[0]))[0]],
                         [float(res_x[1]), self.get_y(float(res_x[1]))[1]]]
                return res_x
            else:
                res_x = [float(res_x[0]), self.get_y(float(res_x[0]))] if line.OnLine(
                    [float(res_x[0]), self.get_y(float(res_x[0]))]) else [float(res_x[1]), self.get_y(float(res_x[1]))]
                return res_x


class Rect(object):
    def __init__(self, pos, name=None):
        """
        按照左上右上左下右下的顺序生成正方形

        :param pos: 2维列表
        """
        self.pos1, self.pos2, self.pos3, self.pos4 = pos
        self.line1 = Line([self.pos1, self.pos2])
        self.line2 = Line([self.pos2, self.pos4])
        self.line3 = Line([self.pos3, self.pos4])
        self.line4 = Line([self.pos1, self.pos2])
        self.square = (self.pos2[0] - self.pos1[0]) * (self.pos4[1] - self.pos2[1])  # 返回面积
        self.name = name if name.find("Rect") != -1 else name + "Rect"
        self.has_square = 1
        self._center = [(self.pos1[0] + self.pos2[0]) * 0.5, (self.pos1[1] + self.pos3[1]) * 0.5]

    def __sizeof__(self):
        return self.square

    def __getitem__(self, item):
        if item == 0:
            return self.pos1
        elif item == 1:
            return self.pos2
        elif item == 2:
            return self.pos3
        elif item == 3:
            return self.pos4
        else:
            raise ValueError("输入0--3在获取正方形的四个点")

    def __iter__(self):
        return [self.pos1, self.pos2, self.pos3, self.pos4]

    def __add__(self, other):
        """
        取并集
        :param other:几何体
        :return:float
        """
        if isinstance(other, Circle):
            return _CircleRect(self, other, "or")
        elif isinstance(other, Rect):
            return _2Rect(self, other, "or")
        else:
            return 0

    def __sub__(self, other):
        """
        取异或面积
        :param other:
        :return: float
        """
        if isinstance(other, Circle):
            return _CircleRect(self, other, "xor")
        elif isinstance(other, Rect):
            return _2Rect(self, other, "xor")
        else:
            return 0

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Circle):
            return _CircleRect(circle=self, rect=other, across_type="and")

    def GetLine(self, line_num):  # 获取某一个线
        if line_num == 1:
            return self.line1
        elif line_num == 2:
            return self.line2
        elif line_num == 3:
            return self.line3
        elif line_num == 4:
            return self.line4
        else:
            raise ValueError("最多4条线，输入1---4之间的整数")

    def PointRelation(self, point):
        if self.pos1[0] <= point <= self.pos2[0] and self.pos3[1] <= point <= self.pos1[1]:
            return "in"
        else:
            return "separate"
