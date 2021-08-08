from math import *

from geometry import Point, Line, Circle, Rect

# _________________________________Private_________________________________
_s = sqrt(2)
Point, Line, Circle, Rect = Point, Line, Circle, Rect
import sys

sys.path.append(r"C:\Users\Mili\PycharmProjects\EngineDev\SimpleGeo\space.py")


# _________________________________Public_________________________________

class LinearSpace(object):
    """

    生成一个线性空间


    """

    def __init__(self):
        self.zero_point = [0, 0]
        self._all_point = []
        self._all_line = []
        self._all_circle = []
        self._all_rect = []

    def SetPoint(self, pos, name=None):
        if name is None:
            name = str(len(self._all_point)) + "Point"
        point = Point(pos=pos, name=name)
        self._all_point.append(point)
        return point

    def SetLine(self, point: list, name=None):
        if name is None:
            name = str(len(self._all_line)) + "Line"
        line = Line(point, name=name)
        self._all_line.append(line)
        return line

    def SetCircle(self, r, o_point, name=None):
        if name is None:
            name = str(len(self._all_circle)) + "Circle"
        circle = Circle(pos=o_point, radio=r, name=name)
        self._all_circle.append(circle)
        return circle

    def SetRect(self, position_point, name=None):
        if name is None:
            name = str(len(self._all_rect)) + "Rect"
        rec = Rect(pos=position_point, name=name)
        self._all_rect.append(rec)
        return rec

    def Find(self, name):
        for item in self._all_rect + self._all_line + self._all_point + self._all_circle:
            if item.name == name:
                return item
        return False

    def Point(self):
        return self._all_point

    def Line(self):
        return self._all_line

    def Circle(self):
        return self._all_circle

    def Rect(self):
        return self._all_rect

    @staticmethod
    def GetCrossSquare(item1, item2, across_type):
        def get_square(shape1, shape2, shape_type):
            if shape_type == "and":
                return shape1 * shape2
            elif shape_type == "or":
                return shape1 + shape2
            elif shape_type == "xor":
                return shape1 - shape2
            else:
                raise TypeError("输入and or xor 获取相交图形信息")

        return 0 if item2.has_square == 1 and item1.has_square == 1 else get_square(item1, item2, across_type)
