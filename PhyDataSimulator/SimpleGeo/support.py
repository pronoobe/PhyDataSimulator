from math import *


def _Len(pos1, pos2):
    return sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))  # 求距离


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


def _2Circle(circle1, circle2, across_type):
    r1 = circle1.radio
    r2 = circle2.radio
    d = _Len(circle1.pos, circle2.pos)
    if r1 + r2 >= d:
        square = 0
    elif _Len(circle1.pos, circle2.pos) <= abs(r2 - r1):
        square = min(circle1.__sizeof__(), circle2.__sizeof__())
    else:
        ang1 = acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d))
        ang2 = acos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d))
        square = ang1 * r1 * r1 + ang2 * r2 * r2 - r1 * d * sin(ang1)
    if across_type == "and":
        return square
    elif across_type == "or":
        return circle1.__sizeof__() + circle2.__sizeof__() - square
    elif across_type == "xor":
        return circle1.__sizeof__() + circle2.__sizeof__() - 2 * square
    else:
        return False


def _2Rect(rect1, rect2, across_type):
    """
    2个矩形的相交
    :param rect1:
    :param rect2:
    :param across_type:
    :return:square
    """
    pos1 = rect1.pos1
    pos2 = rect1.pos4
    pos3 = rect2.pos1
    pos4 = rect2.pos4
    w1 = pos2[0] - pos1[0]
    w2 = pos4[0] - pos3[0]
    h1 = pos1[1] - pos2[1]
    h2 = pos3[1] - pos4[1]
    iou_w = min(pos1[0], pos2[0], pos3[0], pos4[0]) - max(pos1[0], pos2[0], pos3[0], pos4[0]) + w2 + w1
    iou_h = min(pos1[1], pos2[1], pos3[1], pos4[1]) - max(pos1[1], pos2[1], pos3[1], pos4[1]) + h2 + h1
    square = iou_h * iou_w
    if across_type == "and":
        return square
    elif across_type == "or":
        return rect1.__sizeof__() + rect2.__sizeof__() - square
    elif across_type == "xor":
        return rect1.__sizeof__() + rect2.__sizeof__() - square * 2
    else:
        return 0


def _CircleRect(circle, rect, across_type) -> float:
    """
    计算圆和正方形的相交面积求解器，分为若干情况求解
    zero_point_0 :0个情况时（共六种，外离，内含， 一边交圆心内含， 一边交圆心外离， 两边交圆心内含， 两边交圆心外离）
    zero_point(1--4):各有一种情况
    :param circle:
    :param rect:
    :param across_type:
    :return:float
    """

    def zero_point_1() -> float:
        """
        下面为情况列举
        :return:
        """
        if across_type == "and":
            return 0
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__()
        else:
            return rect.__sizeof__() + circle.__sizeof__()

    def zero_point_2() -> float:
        if across_type == "and":
            return circle.__sizeof__()
        elif across_type == "or":
            return rect.__sizeof__()
        else:
            return rect.__sizeof__() - circle.__sizeof__()

    def zero_point_3() -> float:
        l3 = rect.pos1[1] - rect.pos3[1] if circle.LineRelation(rect.line1) == "across" else rect.pos2[0] - rect.pos1[0]
        # across_point = circle.LineCrossPos(rect.line1) + circle.LineCrossPos(rect.line3)
        l1 = circle.radio
        l2 = circle.radio
        theta = acos((l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2))
        square = circle.radio ** 2 * theta + l1 ** 2 * sin(pi - theta)

        if across_type == "and":
            return square
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__() - square
        else:
            return rect.__sizeof__() + circle.__sizeof__() - 2 * square

    def zero_point_0() -> float:
        ...

    def zero_point_4() -> float:
        for line in [rect.line1, rect.line3, rect.line3, rect.line4]:
            if circle.LineRelation(line) == "across":
                pos = circle.LineCrossPos(line)
            else:
                pos = [[0, 0], [0, 0]]
        theta = acos((2 * circle.radio ** 2 - _Len(pos[0], pos[1]) ** 2 / (2 * circle.radio ** 2)))
        square = 0.5 * (circle.radio ** 2 * theta) - 0.5 * circle.radio ** 2 * sin(theta)
        if across_type == "and":
            return square
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__() - square

        else:
            return rect.__sizeof__() + circle.__sizeof__() - 2 * square

    def zero_point_5() -> float:
        square = zero_point_4() + 0.5 * circle.__sizeof__()
        if across_type == "and":
            return square
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__() - square

        else:
            return rect.__sizeof__() + circle.__sizeof__() - 2 * square

    def _point_1() -> float:
        if circle.PointRelation(rect[1]) == "in":  # 你管这叫迭代对象？
            across_line = rect.line1, rect.line2
            pos_r = rect[1]
        elif circle.PointRelation(rect[2]) == "in":
            pos_r = rect[2]
            across_line = rect.line2, rect.line3
        elif circle.PointRelation(rect[3]) == "in":
            pos_r = rect[3]
            across_line = rect.line4, rect.line3
        else:
            pos_r = rect[4]
            across_line = rect.line3, rect.line4

        across_point = [circle.LineCrossPos(across_line[0]), circle.LineCrossPos(across_line[1])]  # 公式推导
        d = _Len(circle.pos, pos_r)
        r = circle.radio
        l1 = _Len(pos_r, across_point[0])
        l2 = _Len(pos_r, across_point[1])
        theta1 = acos((r ** 2 + d ** 2 - l1 ** 2) / (2 * r * d))
        theta2 = acos((r ** 2 + d ** 2 - l2 ** 2) / (2 * r * d))
        fan_shape = 0.5 * r ** 2 * (theta1 + theta2)
        s1 = 0.5 * r * d * sin(theta1)
        s2 = 0.5 * r * d * sin(theta2)
        if rect.PointRelation(circle.pos) == "in":
            square = s1 + s2 + fan_shape
        else:
            square = s1 + s2 + fan_shape
        if across_type == "and":
            return square
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__() - square

        else:
            return rect.__sizeof__() + circle.__sizeof__() - 2 * square

    def _point_2() -> float:
        if rect.PointRelation(circle.pos) == "in":
            encircle_point = []
            for point in rect:
                if circle.LineRelation(point) == "in":
                    encircle_point.append(point)
        else:
            ...

    def _point_3() -> float:
        if circle.PointRelation(rect[1]) != "in":  # 你管这叫迭代对象？*2
            across_line = rect.line1, rect.line2
            pos_r = rect[1]
        elif circle.PointRelation(rect[2]) != "in":
            pos_r = rect[2]
            across_line = rect.line2, rect.line3
        elif circle.PointRelation(rect[3]) != "in":
            pos_r = rect[3]
            across_line = rect.line4, rect.line3
        else:
            pos_r = rect[4]
            across_line = rect.line3, rect.line4
        r = circle.radio
        across_point = circle.LineCrossPos(across_line[0]), circle.LineCrossPos(across_line[1])
        cut_width = _Len(pos_r, across_point[0])
        cut_height = _Len(pos_r, across_point[1])
        cut_len = _Len(across_point[0], across_point[1])
        cut_square = rect.__sizeof__() - cut_width * cut_height * 0.5
        theta = acos((2 * r ** 2 - cut_len ** 2) / (2 * r ** 2))
        arc_square = 0.5 * (r ** 2 * theta - r ** 2 * sin(theta))
        square = cut_square + arc_square
        if across_type == "and":
            return square
        elif across_type == "or":
            return rect.__sizeof__() + circle.__sizeof__() - square

        else:
            return rect.__sizeof__() + circle.__sizeof__() - 2 * square

    def _point_4() -> float:
        if across_type == "and":
            return rect.__sizeof__()
        elif across_type == "or":
            return circle.__sizeof__()
        else:
            return circle.__sizeof__() - rect.__sizeof__()

    in_circle = 0
    for include_info in [circle.PointRelation(x) for x in [iter(rect)]]:  # 分情况讨论，圆内点数量
        if include_info == "in" or include_info == "on":
            in_circle += 1
    if in_circle == 1:
        return _point_1()
    elif in_circle == 2:
        return _point_2()
    elif in_circle == 3:
        return _point_3()
    elif in_circle == 4:
        return _point_4()
    else:
        line_count = 0
        for lin in [rect.line1, rect.line2, rect.line3, rect.line4]:

            if circle.LineRelation(lin) == "across":
                line_count += 1
        if line_count == 2:
            return zero_point_3() if rect.PointRelation(circle.pos) == "in" else zero_point_0()
        elif line_count == 1:
            return zero_point_4() if rect.PointRelation(circle.pos) != "in" else zero_point_5()
        elif line_count == 0:
            return zero_point_1() if rect.PointRelation(circle.pos) == "in" else zero_point_2()
