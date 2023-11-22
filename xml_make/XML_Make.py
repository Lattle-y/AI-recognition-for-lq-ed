import cv2 as cv
import numpy as np
import numpy as np
import math
import copy
import xml.dom.minidom as minidom
import re
import matplotlib.pyplot as plt

# #############
# The code is used to generate an XML file, which requires a txt file of the location where the symbols are recognised,
# as well as an image of the lines remaining after the symbols have been removed.
#

###############

def getLineWidth(img,Line,angel):
    if angel == 'vertical':
        x = Line[0]
        y = int((Line[1]+Line[3])/2)
        count = 0
        for i in range(16):
            if img[y,x-8+i]==255:
                count += 1
        width = count
    if angel == 'horizontal':
        x = int((Line[0]+Line[2])/2)
        y = Line[1]
        count = 0
        for i in range(16):
            if img[y-8+i,x]==255:
                count += 1
        width = count
    return width

def getBusLine(img,Lines,angel):
    width_all = 0
    if angel == 'vertical':
        for Line in Lines:
            width = getLineWidth(img,Line,angel)
            width_all += width
        width_avg = width_all/len(Lines) *1.5
        vertical_connection_line = []
        vertical_busline = []
        for Line in Lines:
            width = getLineWidth(img,Line,angel)
            if width > width_avg and Line[3]-Line[1]>500:
                print(Line,width)
                Line.append('母线-v-'+str(len(vertical_busline)+1))
                vertical_busline.append(Line)
            else:
                Line.append('line-v-'+str(len(vertical_connection_line)+1))
                vertical_connection_line.append(Line)
        return vertical_connection_line, vertical_busline
    if angel == 'horizontal':
        for Line in Lines:
            width = getLineWidth(img,Line,angel)
            width_all += width
        width_avg = width_all/len(Lines) *1.5
        horizontal_connection_line = []
        horizontal_busline = []
        for Line in Lines:
            width = getLineWidth(img,Line,angel)
            if width > width_avg and Line[2]-Line[0]>500:
                Line.append('母线-h-'+str(len(horizontal_busline)+1))
                horizontal_busline.append(Line)
            else:
                Line.append('line-h-'+str(len(horizontal_connection_line)+1))
                horizontal_connection_line.append(Line)
        return horizontal_connection_line, horizontal_busline


# def draw_line(h,w, lines):
#     # 创建白色背景图像
#     image = np.ones((h, w, 3), dtype=np.uint8) * 255
#     image = cv.imread('line.png')
#     for line in lines:
#         point1 = (line[0],line[1])
#         point2 = (line[2],line[3])
#         label = line[-1]
#         # 将浮点坐标转换为整数
#         point1 = tuple(map(int, point1))
#         point2 = tuple(map(int, point2))
#         label_position = ((point1[0] + point2[0]) // 2+10, (point1[1] + point2[1]) // 2)
#         # 在图像上画线
#         cv.line(image, point1, point2, (0, 0, 0), 1)
#         cv.putText(image, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#     # 显示图像
#     cv.imwrite('line.png',image)
# draw_line(4133,5837,lines_h)

# 点到线段最短距离
def point_to_segment_distance(x, y, x1, y1, x2, y2):
    # 计算点到线段的最短距离

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    def vector_subtraction(v1, v2):
        return (v1[0] - v2[0], v1[1] - v2[1])

    def vector_magnitude(v):
        return (v[0]**2 + v[1]**2)**0.5

    # 计算线段的方向向量
    segment_vector = vector_subtraction((x2, y2), (x1, y1))

    # 检查线段长度是否为零
    if vector_magnitude(segment_vector) == 0:
        # 线段长度为零，直接返回点到线段起点的距离
        start_to_point_vector = vector_subtraction((x, y), (x1, y1))
        return vector_magnitude(start_to_point_vector)

    # 计算点到线段起点的向量
    start_to_point_vector = vector_subtraction((x, y), (x1, y1))

    # 计算点到线段的投影长度
    projection_length = dot_product(segment_vector, start_to_point_vector) / vector_magnitude(segment_vector)

    if projection_length <= 0:
        # 如果投影在线段起点之前，点到线段的最短距离为点到线段起点的距离
        return vector_magnitude(start_to_point_vector)
    elif projection_length >= vector_magnitude(segment_vector):
        # 如果投影在线段终点之后，点到线段的最短距离为点到线段终点的距离
        end_to_point_vector = vector_subtraction((x, y), (x2, y2))
        return vector_magnitude(end_to_point_vector)
    else:
        # 如果投影在线段上，点到线段的最短距离为点到线段上的投影点的距离
        projection_point = (
            x1 + (projection_length / vector_magnitude(segment_vector)) * (x2 - x1),
            y1 + (projection_length / vector_magnitude(segment_vector)) * (y2 - y1)
        )
        distance_to_projection_point = ((x - projection_point[0])**2 + (y - projection_point[1])**2)**0.5
        return distance_to_projection_point

# 点到直线的距离
def point_line_dist(x_l1, y_l1, x_l2, y_l2, x, y):
    global dist
    A = y_l2 - y_l1
    B = x_l1 - x_l2
    C = x_l1 * (y_l1 - y_l2) + y_l1 * (x_l2 - x_l1)
    dist = math.fabs(A * x + B * y + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
    return dist

def function(elements, lines_h, lines_v, bus_lines_h, bus_lines_v, threshold=15):
    new_lines_h = copy.deepcopy(lines_h)
    new_lines_v = copy.deepcopy(lines_v)
    n = len(lines_h)
    m = len(elements)
    o = len(bus_lines_v)
    p = len(lines_v)
    u = len(bus_lines_h)

    # 每条连接线必定连接图元或母线或相垂直的连接线

    # 水平线的连接关系
    for i in range(n):
        short1 = 50
        short2 = 50
        x_l1, y_l1, x_l2, y_l2 = lines_h[i][:4]
        connection = ['', '']
        # 先确定是否连接母线
        for j in range(o):
            x_b1, y_b1, x_b2, y_b2 = bus_lines_v[j][:4]

            if math.fabs(x_l1 - x_b1) <= 10 and y_b1 < y_l1 < y_b2:
                connection[0] = bus_lines_v[j][4]
            if math.fabs(x_l2 - x_b2) <= 10 and y_b1 < y_l2 < y_b2:
                connection[1] = bus_lines_v[j][4]

        # # 再确定 是否连接图元
        # for k in range(m):
        #     x_e1, y_e1, x_e2, y_e2 = elements[k][:4]
        #     # 取连接线端点与 图元相邻边中点的欧式距离进行比较
        #     # 若左端点没有连接母线,考虑是否连接了图元
        #     if connection[0] == '':
        #         x = (x_e1 + x_e2) / 2
        #         y = (y_e1 + y_e2) / 2
        #         dist1 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e1, 2))
        #         dist2 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e2, 2))
        #         dist3 = math.sqrt(math.pow(x_l1 - x_e1, 2) + math.pow(y_l1 - y, 2))
        #         dist4 = math.sqrt(math.pow(x_l1 - x_e2, 2) + math.pow(y_l1 - y, 2))
        #         dist = min(dist1, dist2, dist3, dist4)
        #         if dist <= threshold:
        #             connection[0] = elements[k][4]
        #     #  若右端没有连接母线，考虑是否连接了图元
        #     if connection[1] == '':
        #         x = (x_e1 + x_e2) / 2
        #         y = (y_e1 + y_e2) / 2
        #         dist1 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e1, 2))
        #         dist2 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e2, 2))
        #         dist3 = math.sqrt(math.pow(x_l2 - x_e1, 2) + math.pow(y_l2 - y, 2))
        #         dist4 = math.sqrt(math.pow(x_l2 - x_e2, 2) + math.pow(y_l2 - y, 2))
        #         dist = min(dist1, dist2, dist3, dist4)
        #         if dist <= threshold:
        #             connection[1] = elements[k][4]

        # 在确定是否连接了其它连接线
        for t in range(p):
            x_l3, y_l3, x_l4, y_l4 = lines_v[t][:4]
            if connection[0] == '':
                dist1 = math.sqrt(math.pow(x_l1 - x_l3, 2) + math.pow(y_l1 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l1 - x_l4, 2) + math.pow(y_l1 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 3:
                    connection[0] = lines_v[t][4]
                if math.fabs(x_l1 - x_l3) <= 2 and y_l3 < y_l1 < y_l4:
                    connection[0] = lines_v[t][4]

            if connection[1] == '':
                dist1 = math.sqrt(math.pow(x_l2 - x_l3, 2) + math.pow(y_l2 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l2 - x_l4, 2) + math.pow(y_l2 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 3:
                    connection[1] = lines_v[t][4]
                if math.fabs(x_l2 - x_l4) <= 2 and y_l3 < y_l2 < y_l4:
                    connection[1] = lines_v[t][4]

        # 再确定 是否连接图元
        for k in range(m):
            x_e1, y_e1, x_e2, y_e2 = elements[k][:4]
            # 取连接线端点与 图元相邻边中点的欧式距离进行比较
            # 若左端点没有连接母线,考虑是否连接了图元
            dist1 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e1, x_e1, y_e2)
            dist2 = point_to_segment_distance(x_l1,y_l1,x_e2, y_e1, x_e2, y_e2)
            dist3 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e1, x_e2, y_e1)
            dist4 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e2, x_e2, y_e2)

            dist_top = min(dist1, dist2, dist3, dist4)
            if dist_top < short1:
                short1 = dist_top
                label1 = elements[k][4]

            dist1 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e1, x_e1, y_e2)
            dist2 = point_to_segment_distance(x_l2,y_l2,x_e2, y_e1, x_e2, y_e2)
            dist3 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e1, x_e2, y_e1)
            dist4 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e2, x_e2, y_e2)

            dist_bot = min(dist1, dist2, dist3, dist4)
            if dist_bot < short2:
                short2 = dist_bot
                label2 = elements[k][4]
            # if connection[0] == '':
            #     x = (x_e1 + x_e2) / 2
            #     y = (y_e1 + y_e2) / 2
            #     dist1 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e1, 2))
            #     dist2 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e2, 2))
            #     dist3 = math.sqrt(math.pow(x_l1 - x_e1, 2) + math.pow(y_l1 - y, 2))
            #     dist4 = math.sqrt(math.pow(x_l1 - x_e2, 2) + math.pow(y_l1 - y, 2))
            #     dist = min(dist1, dist2, dist3, dist4)
            #     if dist <= threshold:
            #         connection[0] = elements[k][4]
            # #  若右端没有连接母线，考虑是否连接了图元
            # if connection[1] == '':
            #     x = (x_e1 + x_e2) / 2
            #     y = (y_e1 + y_e2) / 2
            #     dist1 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e1, 2))
            #     dist2 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e2, 2))
            #     dist3 = math.sqrt(math.pow(x_l2 - x_e1, 2) + math.pow(y_l2 - y, 2))
            #     dist4 = math.sqrt(math.pow(x_l2 - x_e2, 2) + math.pow(y_l2 - y, 2))
            #     dist = min(dist1, dist2, dist3, dist4)
            #     if dist <= threshold:
            #         connection[1] = elements[k][4]
        if connection[0] == '' and short1<=20:
            connection[0] = label1
        if connection[1] == '' and short2<=20:
            connection[1] = label2

        for t in range(p):
            x_l3, y_l3, x_l4, y_l4 = lines_v[t][:4]
            if connection[0] == '':
                dist1 = math.sqrt(math.pow(x_l1 - x_l3, 2) + math.pow(y_l1 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l1 - x_l4, 2) + math.pow(y_l1 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 5:
                    connection[0] = lines_v[t][4]
                if math.fabs(x_l1 - x_l3) <= 4 and y_l3 < y_l1 < y_l4:
                    connection[0] = lines_v[t][4]

            if connection[1] == '':
                dist1 = math.sqrt(math.pow(x_l2 - x_l3, 2) + math.pow(y_l2 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l2 - x_l4, 2) + math.pow(y_l2 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 5:
                    connection[1] = lines_v[t][4]
                if math.fabs(x_l2 - x_l4) <= 4 and y_l3 < y_l2 < y_l4:
                    connection[1] = lines_v[t][4]


        new_lines_h[i].append(connection)
    # 垂直水平线
    for i in range(p):
        short1=50
        short2=50
        x_l1, y_l1, x_l2, y_l2 = lines_v[i][:4]
        connection = ['', '']
        for j in range(u):
            x_b1, y_b1, x_b2, y_b2 = bus_lines_h[j][:4]
            if math.fabs(y_l1 - y_b1) <= 10 and x_b1 < x_l1 < x_b2:
                connection[0] = bus_lines_h[j][4]
            if math.fabs(y_l2 - y_b2) <= 10 and x_b1 < x_l2 < x_b2:
                connection[1] = bus_lines_h[j][4]
        # for k in range(m):
        #     x_e1, y_e1, x_e2, y_e2 = elements[k][:4]
        #     # 取连接线端点与 图元相邻边中点的欧式距离进行比较
        #     # 若左端点没有连接母线,考虑是否连接了图元
        #     if connection[0] == '':
        #         x = (x_e1 + x_e2) / 2
        #         y = (y_e1 + y_e2) / 2
        #         dist1 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e1, 2))
        #         dist2 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e2, 2))
        #         dist3 = math.sqrt(math.pow(x_l1 - x_e1, 2) + math.pow(y_l1 - y, 2))
        #         dist4 = math.sqrt(math.pow(x_l1 - x_e2, 2) + math.pow(y_l1 - y, 2))
        #         dist = min(dist1, dist2, dist3, dist4)
        #         if dist <= threshold:
        #             connection[0] = elements[k][4]
        #     #  若右端没有连接母线，考虑是否连接了图元
        #     if connection[1] == '':
        #         x = (x_e1 + x_e2) / 2
        #         y = (y_e1 + y_e2) / 2
        #         dist1 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e1, 2))
        #         dist2 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e2, 2))
        #         dist3 = math.sqrt(math.pow(x_l2 - x_e1, 2) + math.pow(y_l2 - y, 2))
        #         dist4 = math.sqrt(math.pow(x_l2 - x_e2, 2) + math.pow(y_l2 - y, 2))
        #         dist = min(dist1, dist2, dist3, dist4)
        #         if dist <= threshold:
        #             connection[1] = elements[k][4]
        for t in range(n):
            x_l3, y_l3, x_l4, y_l4 = lines_h[t][:4]
            if connection[0] == '':
                dist1 = math.sqrt(math.pow(x_l1 - x_l3, 2) + math.pow(y_l1 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l1 - x_l4, 2) + math.pow(y_l1 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 3:
                    connection[0] = lines_h[t][4]
                if math.fabs(y_l1 - y_l3) <= 2 and x_l3 < x_l1 < x_l4:
                    connection[0] = lines_h[t][4]

            if connection[1] == '':
                dist1 = math.sqrt(math.pow(x_l2 - x_l3, 2) + math.pow(y_l2 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l2 - x_l4, 2) + math.pow(y_l2 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 3:
                    connection[1] = lines_h[t][4]
                if math.fabs(y_l2 - y_l4) <= 2 and x_l3 < x_l2 < x_l4:
                    connection[1] = lines_h[t][4]

        for k in range(m):
            x_e1, y_e1, x_e2, y_e2 = elements[k][:4]
            # 取连接线端点与 图元相邻边中点的欧式距离进行比较
            # 若左端点没有连接母线,考虑是否连接了图元
            dist1 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e1, x_e1, y_e2)
            dist2 = point_to_segment_distance(x_l1,y_l1,x_e2, y_e1, x_e2, y_e2)
            dist3 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e1, x_e2, y_e1)
            dist4 = point_to_segment_distance(x_l1,y_l1,x_e1, y_e2, x_e2, y_e2)
            dist_top = min(dist1, dist2, dist3, dist4)
            if dist_top<short1:
                short1 = dist_top
                label1 = elements[k][4]

            dist1 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e1, x_e1, y_e2)
            dist2 = point_to_segment_distance(x_l2,y_l2,x_e2, y_e1, x_e2, y_e2)
            dist3 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e1, x_e2, y_e1)
            dist4 = point_to_segment_distance(x_l2,y_l2,x_e1, y_e2, x_e2, y_e2)

            dist_bot = min(dist1, dist2, dist3, dist4)
            if dist_bot<short2:
                short2 = dist_bot
                label2 = elements[k][4]
            # if connection[0] == '':
                # x = (x_e1 + x_e2) / 2
                # y = (y_e1 + y_e2) / 2
                # dist1 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e1, 2))
                # dist2 = math.sqrt(math.pow(x_l1 - x, 2) + math.pow(y_l1 - y_e2, 2))
                # dist3 = math.sqrt(math.pow(x_l1 - x_e1, 2) + math.pow(y_l1 - y, 2))
                # dist4 = math.sqrt(math.pow(x_l1 - x_e2, 2) + math.pow(y_l1 - y, 2))
                # dist = min(dist1, dist2, dist3, dist4)
                # if dist <= threshold:
                #     connection[0] = elements[k][4]

            #  若右端没有连接母线，考虑是否连接了图元
            # if connection[1] == '':
            #     x = (x_e1 + x_e2) / 2
            #     y = (y_e1 + y_e2) / 2
            #     dist1 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e1, 2))
            #     dist2 = math.sqrt(math.pow(x_l2 - x, 2) + math.pow(y_l2 - y_e2, 2))
            #     dist3 = math.sqrt(math.pow(x_l2 - x_e1, 2) + math.pow(y_l2 - y, 2))
            #     dist4 = math.sqrt(math.pow(x_l2 - x_e2, 2) + math.pow(y_l2 - y, 2))
            #     dist = min(dist1, dist2, dist3, dist4)
            #     if dist <= threshold:
            #         connection[1] = elements[k][4]
        if connection[0] == '' and short1 <= 20:
            connection[0] = label1
        if connection[1] == '' and short2 <= 20:
            connection[1] = label2

        for t in range(n):
            x_l3, y_l3, x_l4, y_l4 = lines_h[t][:4]
            if connection[0] == '':
                dist1 = math.sqrt(math.pow(x_l1 - x_l3, 2) + math.pow(y_l1 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l1 - x_l4, 2) + math.pow(y_l1 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 5:
                    connection[0] = lines_h[t][4]
                if math.fabs(y_l1 - y_l3) <= 4 and x_l3 < x_l1 < x_l4:
                    connection[0] = lines_h[t][4]

            if connection[1] == '':
                dist1 = math.sqrt(math.pow(x_l2 - x_l3, 2) + math.pow(y_l2 - y_l3, 2))
                dist2 = math.sqrt(math.pow(x_l2 - x_l4, 2) + math.pow(y_l2 - y_l4, 2))
                dist = min(dist1, dist2)
                if dist <= 5:
                    connection[1] = lines_h[t][4]
                if math.fabs(y_l2 - y_l4) <= 4 and x_l3 < x_l2 < x_l4:
                    connection[1] = lines_h[t][4]

        new_lines_v[i].append(connection)

    return new_lines_h, new_lines_v

# 区域到区域的最小距离  (x1, y1, x2, y2) ~ (x3, y3, x4, y4) 根据文字区域与图元区域关系
def area_area_dist(x1, y1, x2, y2, x3, y3, x4, y4):
    global dist_
    if x1 > x4 or x2 < x3:
        dist_1 = point_line_dist(x4, y3, x4, y4, x1, y1)
        dist_2 = point_line_dist(x3, y3, x3, y4, x2, y2)
        dist_ = min(dist_1, dist_2)
    if y2 < y3 or y1 > y4:
        dist_1 = point_line_dist(x3, y3, x4, y3, x2, y2)
        dist_2 = point_line_dist(x3, y4, x4, y4, x1, y1)
        dist_ = min(dist_1, dist_2)
    if (x3 < x1 < x4 and y3 < y1 < y4) or (x3 < x1 < x4 and y3 < y2 < y4) or (x3 < x2 < x4 and y3 < y1 < y4) or (
            x3 < x2 < x4 and y3 < y2 < y4):
        x = (x3 + x4) / 2
        y = (y3 + y4) / 2
        dist_1 = math.sqrt(math.pow(x - x1, 2) + math.pow(y - y1, 2))
        dist_2 = math.sqrt(math.pow(x - x1, 2) + math.pow(y - y2, 2))
        dist_3 = math.sqrt(math.pow(x - x2, 2) + math.pow(y - y1, 2))
        dist_4 = math.sqrt(math.pow(x - x2, 2) + math.pow(y - y2, 2))
        dist_ = -min(dist_1, dist_2, dist_3, dist_4)
    return dist_


# 文字匹配图元   text = [x1, y1, x2 ,y2, 'text] element = [x1 ,y1, x2 ,y2, name] buslines = [x1, y1, x2, y2, name]
def text_matching(texts, elements, buslines):
    new_elements = copy.deepcopy(elements)
    new_buslines = copy.deepcopy(buslines)
    n = len(texts)
    m = len(elements)
    l = len(buslines)
    # 先确定母线
    for i in range(l):
        selections = []
        distance = []
        x_b1, y_b1, x_b2, y_b2 = buslines[i][:4]
        for j in range(n):
            if '母线' in texts[j][4]:
                x_t1, y_t1, x_t2, y_t2 = texts[j][:4]
                dist_1 = point_line_dist(x_b1, y_b1, x_b2, y_b2, x_t1, y_t1)
                dist_2 = point_line_dist(x_b1, y_b1, x_b2, y_b2, x_t1, y_t2)
                dist_3 = point_line_dist(x_b1, y_b1, x_b2, y_b2, x_t2, y_t1)
                dist_4 = point_line_dist(x_b1, y_b1, x_b2, y_b2, x_t2, y_t2)
                dist = min(dist_1, dist_2, dist_3, dist_4)
                distance.append(dist)
                selections.append(texts[j][4])
        distance = np.array(distance)
        index = distance.argmin()
        new_buslines[i].append(selections[index])

    # 在确定每个图元对应的字符串
    for i in range(m):
        x_e1, y_e1, x_e2, y_e2 = elements[i][:4]
        distance = []
        texts_list = []
        for j in range(n):
            x_t1, y_t1, x_t2, y_t2 = texts[j][:4]
            new_text = texts[j][4]
            if x_t1 > x_e1 - 100 and x_t2 < x_e2 + 100 and y_t1 > y_e1 - 100 and y_t2 < y_e2 + 100:
                # 补全省略的文字
                if texts[j][4][0] == '-':
                    texts_list_ = []
                    dist = []
                    for k in range(n):
                        if k != j:
                            x1, y1, x2, y2 = texts[k][:4]
                            if x1 > x_t1 - 100 and x2 < x_t2 + 100 and y1 > y_t1 - 100 and y2 < y_t2 + 100:
                                if texts[k][4].isdigit() == True:
                                    texts_list_.append(texts[k][4] + texts[j][4])
                                    dist.append(math.sqrt(math.pow((x1 + x2) / 2 - (x_t1 + x_t2) / 2, 2) + math.pow(
                                        (y1 + y2) / 2 - (y_t1 + y_t2) / 2, 2)))
                    dist = np.array(dist)
                    index1 = np.argmin(dist)
                    new_text = texts_list_[index1]

                d = area_area_dist(x_t1, y_t1, x_t2, y_t2, x_e1, y_e1, x_e2, y_e2)
                distance.append(d)
                texts_list.append(new_text)
        distance = np.array(distance)
        if '隔离开关' in elements[i][4]:
            index = np.argmin(distance)
            text = texts_list[index]
        if '变压器' in elements[i][4]:
            flag = False
            while flag == False:
                index = np.argmin(distance)
                text = texts_list[index]
                if '变' in text:
                    flag = True
                else:
                    distance = np.delete(distance, index)
                    texts_list.pop(index)
        if 'ACLine' in elements[i][4]:
            flag = False
            while flag == False:
                index = np.argmin(distance)
                text = texts_list[index]
                if '线' in text:
                    flag = True
                else:
                    distance = np.delete(distance, index)
                    texts_list.pop(index)
        new_elements[i].append(text)

    return new_elements, new_buslines

def judge(link1,link2):
    for a in link1:
        for b in link2:
            if a == b:
                return True
def arradd(link1,link2):
    is_same = False
    for link_i in link1:
        for link_j in link2:
            if link_i == link_j:
                is_same = True
    return is_same
def arrayAdd(arr1,arr2):
    new_arr = []
    for arr in arr1:
        new_arr.append(arr)
    for arr in arr2:
        if arr not in new_arr:
            new_arr.append(arr)
    return new_arr
def extract_related_lines(lines):
    related_lines = []

    for line in lines:
        related_line = [line]

        for other_line in lines:
            if other_line != line:
                if line[-1][0] == other_line[4] or line[-1][1] ==other_line[4]:
                    related_line.append(other_line)
        if related_line not in related_lines:
            related_lines.append(related_line)


    return related_lines
def filter_longer_arrays(array_of_arrays):
    result_array = []
    final_result = []
    for i, array1 in enumerate(array_of_arrays):
        is_longer = True

        for j, array2 in enumerate(array_of_arrays):
            if i != j and len(array1)<= len(array2):
                for array in array1:
                    if array in array2:
                        is_longer = False
                        break

        if is_longer:
            result_array.append(array1)
    for array in result_array:
        # print("array:",array)
        result =array
        # print("array_of_arrays:",array_of_arrays)
        for array1 in array_of_arrays:
            for a in array1:
                if a in array:
                    result = arrayAdd(result,array1)
        # print("result:",result)
        final_result.append(result)
    return final_result
def merge(related_lines):
    new_links = []
    for i,link_i in enumerate(related_lines):
        links = link_i
        for j,link_j in enumerate(related_lines):
            if i!=j:
                is_same = arradd(link_i,link_j)
                if is_same:
                    links = arrayAdd(links,link_j)
        new_links.append(links)
    # print('new_links', new_links)
    final_links = []

    for i in range(len(new_links)-1):
        flag = False
        for j in range(i+1,len(new_links)):
            flag =  judge(new_links[i],new_links[j])
            if flag == True:
                break
        if flag == True:
            continue
        else:
            final_links.append(new_links[i])
                # print("2")
    final_links.append(new_links[-1])
    # print('final_links',final_links)



    return  final_links

def mergeLinks(related_links):
    new_links = []
    for related_link in related_links:
        cor = []
        name = []
        con = []
        for i in range(len(related_link)):
            cor.append((related_link[i][0],related_link[i][1],related_link[i][2],related_link[i][3]))
            name.append(related_link[i][4])
            if 'line' not in related_link[i][-1][0]:
                con.append(related_link[i][-1][0])
            if 'line' not in related_link[i][-1][1]:
                con.append(related_link[i][-1][1])
        link = [cor,name,con]
        new_links.append(link)
    return new_links

def getLinks(new_lines_h,new_lines_v):
    len_h = len(new_lines_h)
    len_v = len(new_lines_v)
    links = []
    leave = []
    result = []
    for i in range(len_h):
        new_line_h = new_lines_h[i]
        connection = new_line_h[-1]
        if 'line' not in connection[0] and 'line' not in connection[1]:
            links.append(new_line_h)
        else:
            leave.append(new_line_h)
    for j in range(len_v):
        new_line_v = new_lines_v[j]
        connection= new_line_v[-1]
        if 'line' not in connection[0] and 'line' not in connection[1]:
            links.append(new_line_v)
        else:
            leave.append(new_line_v)
    for i in range(len(links)):
        cor = [(links[i][0],links[i][1],links[i][2],links[i][3])]
        name = [links[i][4]]
        con = links[i][-1]
        link = [cor,name,con]
        result.append(link)
    related_lines = extract_related_lines(leave)
    related_lines = merge(related_lines)
    # related_lines = filter_longer_arrays(related_lines)
    print("related_lines:", related_lines)
    test = mergeLinks(related_lines)
    for i in range(len(test)):
        result.append(test[i])

    #     处理线条单连接情况
    final_result = []
    for i in range(len(result)):
        if '' in result[i][-1]:
            newlink = []
            newlink.append(result[i][0])
            newlink.append(result[i][1])
            cor = []
            for a in result[i][-1]:
                if a != '':
                    cor.append(a)
            newlink.append(cor)
            final_result.append(newlink)
        else:
            final_result.append(result[i])

    return  final_result
def extract_digits_from_end(input_string):
    # 使用正则表达式查找字符串末尾的数字
    match = re.search(r'\d+$', input_string)

    # 返回匹配到的数字部分，如果未匹配到数字，返回 None
    return match.group() if match else ''

# 读取图元txt文件数据
def opentxt_element(filename):
    data = []
    file = open(filename,'r')
    file_data = file.readlines()
    for row in file_data:
        x0 = int(row.split(' ')[0])
        x1 = int(row.split(' ')[1])
        x2 = int(row.split(' ')[2])
        x3 = int(row.split(' ')[3])
        x4 = int(row.split(' ')[4])
        data.append([x0,x1,x2,x3,x4])
    return data

# 读取线条图
img = cv.imread('35lines.jpg',cv.IMREAD_GRAYSCALE)
# 细化线条
ret, binary_image = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
cv.imwrite('test1.png',binary_image)
# thinned_image = cv.ximgproc.thinning(binary_image)
# cv.imwrite('test2.png',thinned_image)
# 获取垂直线
structure = cv.getStructuringElement(1,(1,12))
eroded = cv.erode(binary_image, structure, iterations=1)
ver = cv.dilate(eroded, structure, iterations=1)
# 获取水平线
structure = cv.getStructuringElement(1,(12,1))
eroded = cv.erode(binary_image, structure, iterations=1)
hor = cv.dilate(eroded, structure, iterations=1)
# cv.imwrite('binary.png',binary_image)
# cv.imwrite('ver.png',ver)
# cv.imwrite('hor.png',hor)
# cv.waitKey(0)

vertical_line = []
horizontal_line = []

contours, hierarchy = cv.findContours(hor, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 遍历轮廓，获取水平线段的端点
for contour in contours:
    left_point = tuple(contour[contour[:,:,0].argmin()][0])   # 最左面的点
    right_point = tuple(contour[contour[:,:,0].argmax()][0])   # 最右面的点
    top_point = tuple(contour[contour[:,:,1].argmin()][0])   # 最上面的点
    bottom_point = tuple(contour[contour[:,:,1].argmax()][0])   # 最下面的点
    point = [left_point[0],(top_point[1]+bottom_point[1])//2,right_point[0],(top_point[1]+bottom_point[1])//2]
    # if top_point[1] != bottom_point[1]:
    #     point = [top_point[0], top_point[1], bottom_point[0], top_point[1]]
    # else:
    #     point = [top_point[0],top_point[1],bottom_point[0],top_point[1]]
    horizontal_line.append(point)

contours, hierarchy = cv.findContours(ver, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 遍历轮廓，获取垂直线段的端点
for contour in contours:
    left_point = tuple(contour[contour[:,:,0].argmin()][0])   # 最左面的点
    right_point = tuple(contour[contour[:,:,0].argmax()][0])   # 最右面的点
    top_point = tuple(contour[contour[:,:,1].argmin()][0])   # 最上面的点
    bottom_point = tuple(contour[contour[:,:,1].argmax()][0])   # 最下面的点
    point = [(left_point[0]+right_point[0])//2,top_point[1],(left_point[0]+right_point[0])//2,bottom_point[1]]
    # if top_point[0] != bottom_point[0]:
    #     point = [top_point[0], top_point[1], top_point[0], bottom_point[1]]
    # else:
    #     point = [top_point[0],top_point[1],bottom_point[0],bottom_point[1]]
    vertical_line.append(point)

lines_h,buslines_h = getBusLine(binary_image,horizontal_line,'horizontal')
lines_v,buslines_v = getBusLine(binary_image,vertical_line,'vertical')
# print(lines_h)
# print(buslines_h)
# print(lines_v)
# print(buslines_v)

elements = opentxt_element('elements2.txt')
# print('elements:',elements)
new_elements = []
for i,element in enumerate(elements):
    icon = [element[1],element[2],element[3],element[4]]
    if element[0] == 0:
        icon.append('CBreaker'+str(i))
    elif element[0] == 1:
        # icon.append('Capacitor'+str(i))
        icon.append('Disconnector' + str(i))
    elif element[0] == 2:
        # icon.append('Inductance'+str(i))
        icon.append('CBreaker' + str(i))
    elif element[0] == 3:
        icon.append('Transformer'+str(i))
    elif element[0] == 4:
        icon.append('Generator'+str(i))
    elif element[0] == 5:
        icon.append('Disconnector'+str(i))
    elif element[0] == 6:
        # icon.append('Grounding'+str(i))
        icon.append('Disconnector' + str(i))
    elif element[0] == 7:
        icon.append('EarthingSwitch'+str(i))
    new_elements.append(icon)
# print('new_elements:',new_elements)
# 增加母线
buslines = buslines_h+buslines_v
n_elements = len(new_elements)
for i,busline in enumerate(buslines_h):
    busline[-1] = 'BusLine'+str(i+n_elements)
    new_elements.append(busline)
# print('new_elements:',new_elements)

# def draw_rectangle_with_label( elements):
#     # 创建白色背景图像
#     # image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
#     image = cv.imread('line.png')
#
#     for element in elements:
#         top_left = (element[0],element[1])
#         bottom_right = (element[2],element[3])
#         # 将浮点坐标转换为整数
#         top_left = tuple(map(int, top_left))
#         bottom_right = tuple(map(int, bottom_right))
#
#         # 在图像上画矩形
#         cv.rectangle(image, top_left, bottom_right, (0, 0, 0), 1)
#
#         # 计算标签的位置（取矩形左上角）
#         label_position = bottom_right
#         label = element[-1]
#         # 在图像上标上矩形的名字
#         cv.putText(image, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#     cv.imwrite('line.png',image)
# draw_rectangle_with_label(new_elements)

new_lines_h, new_lines_v = function(new_elements, lines_h, lines_v, buslines_h, buslines_v, threshold=15)

# print('new_lines_h:',new_lines_h)
# print('new_lines_v:',new_lines_v)

H_lines = []
for line_h in new_lines_h:
    if line_h[-1] != ['', '']:
        H_lines.append(line_h)
V_lines = []
for line_v in new_lines_v:
    if line_v[-1] != ['', '']:
        V_lines.append(line_v)

# print('H_lines:',H_lines)
# print('V_lines:',V_lines)

Links = getLinks(H_lines,V_lines)
# print('Links:',Links)

dom = minidom.getDOMImplementation().createDocument(None,'wiring_diagram',None)
root = dom.documentElement
root.setAttribute('wdith',str(5847))
root.setAttribute('height',str(4133))

# 写入元件信息
icons = dom.createElement('icons')
root.appendChild(icons)
for i,element in enumerate(new_elements):
    icon = dom.createElement('icon')
    icon.setAttribute('id',str(i))
    if 'Disconnector' in element[-1]:
        icon.setAttribute('type',"Disconnector")
    if 'Capacitor' in element[-1]:
        icon.setAttribute('type',"Capacitor")
    if 'Inductance' in element[-1]:
        icon.setAttribute('type',"Inductance")
    if 'Transformer' in element[-1]:
        icon.setAttribute('type',"Transformer")
    if 'Generator' in element[-1]:
        icon.setAttribute('type',"Generator")
    if 'CBreaker' in element[-1]:
        icon.setAttribute('type',"CBreaker")
    if 'Grounding' in element[-1]:
        icon.setAttribute('type',"Grounding")
    if 'EarthingSwitch' in element[-1]:
        icon.setAttribute('type',"EarthingSwitch")
    if 'BusLine' in element[-1]:
        icon.setAttribute('type',"bus")
    box = f"{element[0]},{element[1]};{element[2]},{element[3]}"
    icon.setAttribute('box',box)
    icons.appendChild(icon)
    icon.setAttribute('textId', str(0))

# 写入文本信息
texts = dom.createElement('texts')
root.appendChild(texts)
text = dom.createElement('text')
text.setAttribute('id',str(0))
text.setAttribute('content','')
box = f"{0},{0};{0},{0}"
text.setAttribute('box',box)
texts.appendChild(text)

# #写入连接线关系
links_xml = dom.createElement('links')
root.appendChild(links_xml)
for i,link_i in enumerate(Links):
    link = dom.createElement('link')
    id = f"{i}"
    link.setAttribute('id',id)
    link_position = link_i[0]
    position = ""
    for link_p in link_position:
        position = position + f"{link_p[0]},{link_p[1]};{link_p[2]},{link_p[3]};"
    link.setAttribute('position',position)
#
    link_iconId = link_i[-1]
    iconId = ""
    for link_icon in link_iconId:
        number = extract_digits_from_end(link_icon)
        iconId = iconId + f"{number},"
    link.setAttribute('iconId',iconId)
    links_xml.appendChild(link)


with open('test.xml', 'w', encoding='utf-8') as f:
    dom.writexml(f, addindent='\t', newl='\n',encoding='utf-8')

