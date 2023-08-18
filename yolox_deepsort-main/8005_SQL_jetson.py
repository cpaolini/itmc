import sys
sys.path.insert(0, './YOLOX')
import torch
import cv2
from yolox.utils import vis
import time
from yolox.exp import get_exp
import numpy as np
from collections import deque
from collections import Counter
import matplotlib.pyplot as plt
import csv
import datetime
from datetime import datetime, timedelta
import os
import logging


# importing Detector
from yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Predictor

# Importing Deepsort
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Importing Visuals
from visuals import *

from intersect_ import *
import pymysql
import gst
import glob

# A Dictionary to keep data of tracking
data_deque = {}

class_names = COCO_CLASSES

#database connection
connection = pymysql.connect(host="********", user="********", passwd="********", database="********")
cursor = connection.cursor()
connection.close()



lines  = [
     {'Title': 'Line1', 'Cords': [(272,248), (360,248)]},
     {'Title': 'Line2', 'Cords': [(360,248), (341,280)]},
     {'Title': 'Line3', 'Cords': [(341,280), (239,280)]},
     {'Title': 'Line4', 'Cords': [(239,280), (272,248)]},
     {'Title': 'Line5', 'Cords': [(341,280), (320,316)]},
     {'Title': 'Line6', 'Cords': [(320,316), (203,316)]},
     {'Title': 'Line7', 'Cords': [(203,316), (239,280)]},
     {'Title': 'Line8', 'Cords': [(320,316), (298,353)]},
     {'Title': 'Line9', 'Cords': [(298,353), (165,353)]},
     {'Title': 'Line10', 'Cords': [(165,353), (203,316)]},
     {'Title' : 'Line11', 'Cords': [(298,353), (274,395)]},
     {'Title' : 'Line12', 'Cords': [(274,395), (122,395)]},
     {'Title' : 'Line13', 'Cords': [(122,395), (165,353)]},
     {'Title' : 'Line14', 'Cords': [(274,395), (245,445)]},
     {'Title' : 'Line15', 'Cords': [(245,445), (71,445)]},
     {'Title' : 'Line16', 'Cords': [(71,445), (122,395)]},
     {'Title' : 'Line17', 'Cords': [(360,248), (485,248)]},
     {'Title' : 'Line18', 'Cords': [(485,248), (478,280)]},
     {'Title' : 'Line19', 'Cords': [(478,280), (341,280)]},
     {'Title' : 'Line20', 'Cords': [(478,280), (470,316)]},
     {'Title' : 'Line21', 'Cords': [(470,316), (320,316)]},
     {'Title' : 'Line22', 'Cords': [(470,316), (462,353)]},
     {'Title' : 'Line23', 'Cords': [(462,353), (298,353)]},
     {'Title' : 'Line24', 'Cords': [(462,353), (452,395)]},
     {'Title' : 'Line25', 'Cords': [(452,395), (274,395)]},
     {'Title' : 'Line26', 'Cords': [(452,395), (441,445)]},
     {'Title' : 'Line27', 'Cords': [(441,445), (245,445)]},
     {'Title' : 'Line28', 'Cords' : [(485,248), (600,245)]},
     {'Title' : 'Line29', 'Cords' : [(600,245), (608,280)]},
     {'Title' : 'Line30', 'Cords' : [(608,280), (478,280)]},
     {'Title' : 'Line31', 'Cords' : [(608,280), (616,316)]},
     {'Title' : 'Line32', 'Cords' : [(616,316), (470,316)]},
     {'Title' : 'Line33', 'Cords' : [(616,316), (625,353)]},
     {'Title' : 'Line34', 'Cords' : [(625,353), (462,353)]},
     {'Title' : 'Line35', 'Cords' : [(625,353), (634,395)]},
     {'Title' : 'Line36', 'Cords' : [(634,395), (452,395)]},
     {'Title' : 'Line37', 'Cords' : [(634,395), (645,445)]},
     {'Title' : 'Line38', 'Cords' : [(645,445), (441,445)]},
     {'Title' : 'Line39', 'Cords' : [(600,245), (705,243)]},
     {'Title' : 'Line40', 'Cords' : [(705,243), (725,280)]},
     {'Title' : 'Line41', 'Cords' : [(725,280), (608,280)]},
     {'Title' : 'Line42', 'Cords' : [(725,280), (744,316)]},
     {'Title' : 'Line43', 'Cords' : [(744,316), (616,316)]},
     {'Title' : 'Line44', 'Cords' : [(744,316), (764,353)]},
     {'Title' : 'Line45', 'Cords' : [(764,353), (625,353)]},
     {'Title' : 'Line46', 'Cords' : [(764,353), (787,395)]},
     {'Title' : 'Line47', 'Cords' : [(787,395), (634,395)]},
     {'Title' : 'Line48', 'Cords' : [(787,395), (811,445)]},
     {'Title' : 'Line49', 'Cords' : [(811,445), (645,445)]},
     {'Title' : 'Line50', 'Cords' : [(705,243), (810,243)]},
     {'Title': 'Line51', 'Cords': [(810,243), (838, 280)]},
     {'Title': 'Line52', 'Cords': [(838, 280), (725, 280)]},
     {'Title': 'Line53', 'Cords': [(838, 280), (865, 316)]},
     {'Title': 'Line54', 'Cords': [(865, 316), (744, 316)]},
     {'Title': 'Line55', 'Cords': [(865, 316), (893, 353)]},
     {'Title': 'Line56', 'Cords': [(893, 353), (764, 353)]},
     {'Title': 'Line57', 'Cords': [(893, 353), (925, 395)]},
     {'Title': 'Line58', 'Cords': [(925, 395), (787, 395)]},
     {'Title': 'Line59', 'Cords': [(925, 395), (962, 445)]},
     {'Title': 'Line60', 'Cords': [(962, 445), (811, 445)]},
     {'Title': 'Line61', 'Cords': [(810, 243), (905, 243)]},
     {'Title': 'Line62', 'Cords': [(905, 243), (942, 280)]},
     {'Title': 'Line63', 'Cords': [(942, 280), (838, 280)]},
     {'Title': 'Line64', 'Cords': [(942, 280), (979, 316)]},
     {'Title': 'Line65', 'Cords': [(979, 316), (865, 316)]},
     {'Title': 'Line66', 'Cords': [(979, 316), (1017, 353)]},
     {'Title': 'Line67', 'Cords': [(1017, 353), (893, 353)]},
     {'Title': 'Line68', 'Cords': [(1017, 353), (1060, 395)]},
     {'Title': 'Line69', 'Cords': [(1060, 395), (925, 395)]},
     {'Title': 'Line70', 'Cords': [(1060, 395), (1111, 445)]},
     {'Title': 'Line71', 'Cords': [(1111, 445), (962, 445)]},
     {'Title': 'Line72', 'Cords': [(905, 243), (1010, 243)]},
     {'Title': 'Line73', 'Cords': [(1010, 243), (1060, 280)]},
     {'Title': 'Line74', 'Cords': [(1060, 280), (942, 280)]},
     {'Title': 'Line75', 'Cords': [(1060, 280), (1108, 316)]},
     {'Title': 'Line76', 'Cords': [(1108, 316), (979, 316)]},
     {'Title': 'Line77', 'Cords': [(1108, 316), (1157, 353)]},
     {'Title': 'Line78', 'Cords': [(1157, 353), (1017, 353)]},
     {'Title': 'Line79', 'Cords': [(1157, 353), (1213, 395)]},
     {'Title': 'Line80', 'Cords': [(1213, 395), (1060, 395)]},
     {'Title': 'Line81', 'Cords': [(1213, 395), (1280, 445)]},
     {'Title': 'Line82', 'Cords': [(1280, 445), (1111, 445)]},
     {'Title': 'Line83', 'Cords': [(1060, 280), (1235, 280)]},
     {'Title': 'Line84', 'Cords': [(1235, 280), (1010, 243)]},
     {'Title': 'Line85', 'Cords': [(239, 280), (58, 280)]},
     {'Title': 'Line86', 'Cords': [(58, 280), (272, 248)]}


]





# filepath = "/mnt/beegfs/home/salehipo/Yolox_deepsort/8006"
#
# st = sys.argv[1]
# st_split = st.split("/")
#
# split_end = st_split[-1].split(".")
#
# end_path_pet = "PET"+split_end[0]+".csv"
#
# path_pet = os.path.join(filepath, end_path_pet)

#Draw the Lines
def draw_lines(lines, img):
    for line in lines:
        img = cv2.line(img, line['Cords'][0], line['Cords'][1], (255,255,255), 3)
    return img

# Update the Counter

# f = open(path_pet, 'w')
# writer = csv.writer(f)
Block_id=0
camera = 8005
def update_counter(centerpoints, obj_name, id):
    for line in lines:
        Block_id = ""
        p1 = Point(*centerpoints[0])
        q1 = Point(*centerpoints[1])
        p2 = Point(*line['Cords'][0])
        q2 = Point(*line['Cords'][1])
        if doIntersect(p1, q1, p2, q2):
            #object_counter[line['Title']].update([obj_name])

            #with open(path_pet, 'w') as f:
                #writer = csv.writer(f)

            if line['Title'] == 'Line1' or line['Title'] == 'Line2' or line['Title'] == 'Line3' or line['Title'] == 'Line4':
                Block_id = 1
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert1 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert1)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line3':
                    Block_id = 2
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert2 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert2)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line2':
                    Block_id = 11
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert3 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert3)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line4':
                    Block_id = 81
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert70 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert70)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line5' or line['Title'] == 'Line6' or line['Title'] == 'Line7':
                Block_id = 2
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert4 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert4)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line6':
                    Block_id = 3
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert5 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert5)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line5':
                    Block_id = 12
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])

                    insert6 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert6)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line8' or line['Title'] == 'Line9' or line['Title'] == 'Line10':
                Block_id = 3
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), int(cap.get(cv2.CAP_PROP_POS_FRAMES))])

                insert7 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert7)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line8':
                    Block_id = 13
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])

                    insert9 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert9)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line9':
                    Block_id = 4
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])

                    insert71 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert71)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line11' or line['Title'] == 'Line12' or line['Title'] == 'Line13':
                Block_id = 4
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])

                insert10 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert10)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line12':
                    Block_id = 5
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert11 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert11)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line11':
                    Block_id = 14
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert65 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert65)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line14' or line['Title'] == 'Line15' or line['Title'] == 'Line16':
                Block_id = 5
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert12 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert12)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line14':
                    Block_id = 15
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert13 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert13)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line17' or line['Title'] == 'Line18' or line['Title'] == 'Line19':
                Block_id = 11
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert15 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert15)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line18':
                    Block_id = 21
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert16 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert16)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line19':
                    Block_id = 12
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert72= "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert72)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line20' or line['Title'] == 'Line21':
                Block_id = 12
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert18 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert18)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line20':
                    Block_id = 22
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert19 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert19)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line21':
                    Block_id = 13
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert20 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert20)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line22' or line['Title'] == 'Line23':
                Block_id = 13
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert21 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert21)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line22':
                    Block_id = 23
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert22 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert22)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line23':
                    Block_id = 14
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert66 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert66)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line24' or line['Title'] == 'Line25':
                Block_id = 14
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert23 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert23)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line24':
                    Block_id = 24
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert24 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert24)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line25':
                    Block_id = 15
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert73 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert73)
                    # commiting the connection then closing it.
                    connection.commit()




            if line['Title'] == 'Line26' or line['Title'] == 'Line27':
                Block_id = 15
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert26 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert26)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line26':
                    Block_id = 25
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert27 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert27)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line28' or line['Title'] == 'Line29' or line['Title'] == 'Line30':
                Block_id = 21
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert29 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert29)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line29':
                    Block_id = 31
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert30 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert30)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line30':
                    Block_id = 22
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert31 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert31)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line31' or line['Title'] == 'Line32':
                Block_id = 22
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert32 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert32)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line31':
                    Block_id = 32
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert33 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert33)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line32':
                    Block_id = 23
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert74 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert74)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line33' or line['Title'] == 'Line34':
                Block_id = 23
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert34 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert34)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line34':
                    Block_id = 24
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert35 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert35)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line33':
                    Block_id = 33
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert36 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert36)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line35' or line['Title'] == 'Line36':
                Block_id = 24
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert37 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert37)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line36':
                    Block_id = 25
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert38 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert38)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line35':
                    Block_id = 34
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert39 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert39)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line37' or line['Title'] == 'Line38':
                Block_id = 25
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert40 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert40)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line37':
                    Block_id = 35
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert42 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert42)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line39' or line['Title'] == 'Line40' or line['Title'] == 'Line41':
                Block_id = 31
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert43 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert43)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line40':
                    Block_id = 41
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert44 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert44)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line41':
                    Block_id = 32
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert67 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert67)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line42' or line['Title'] == 'Line43':
                Block_id = 32
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert45 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert45)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line43':
                    Block_id = 33
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert46 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert46)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line42':
                    Block_id = 42
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert75 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert75)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line44' or line['Title'] == 'Line45':
                Block_id = 33
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert48 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert48)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line44':
                    Block_id = 43
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert76 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert76)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line45':
                    Block_id = 34
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert77 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert77)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line46' or line['Title'] == 'Line47':
                Block_id = 34
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert78 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert78)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line46':
                    Block_id = 44
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert79 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert79)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line47':
                    Block_id = 35
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert80 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert80)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line48' or line['Title'] == 'Line49':
                Block_id = 35
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert81 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert81)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line48':
                    Block_id = 45
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert82 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert82)
                    # commiting the connection then closing it.
                    connection.commit()




            if line['Title'] == 'Line50' or line['Title'] == 'Line51' or line['Title'] == 'Line52':
                Block_id = 41
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert83 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert83)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line51':
                    Block_id = 51
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert84 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert84)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line52':
                    Block_id = 42
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert85 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert85)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line53' or line['Title'] == 'Line54':
                Block_id = 42
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert86 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert86)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line53':
                    Block_id = 52
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert87 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert87)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line54':
                    Block_id = 43
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert88 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert88)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line55' or line['Title'] == 'Line56':
                Block_id = 43
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert89 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert89)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line55':
                    Block_id = 53
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert90 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert90)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line56':
                    Block_id = 44
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert91 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert91)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line57' or line['Title'] == 'Line58':
                Block_id = 44
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert92 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert92)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line57':
                    Block_id = 54
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert93 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert93)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line58':
                    Block_id = 45
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert94 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert94)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line59' or line['Title'] == 'Line60':
                Block_id = 45
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert95 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert95)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line59':
                    Block_id = 55
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert96 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert96)
                    # commiting the connection then closing it.
                    connection.commit()



            if line['Title'] == 'Line61' or line['Title'] == 'Line62' or line['Title'] == 'Line63':
                Block_id = 51
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert97 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert97)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line62':
                    Block_id = 61
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert98 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert98)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line63':
                    Block_id = 52
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert99 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert99)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line64' or line['Title'] == 'Line65':
                Block_id = 52
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert100 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert100)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line64':
                    Block_id = 62
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert101 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert101)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line65':
                    Block_id = 53
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert103 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert103)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line66' or line['Title'] == 'Line67':
                Block_id = 53
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert104 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert104)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line66':
                    Block_id = 63
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert105 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert105)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line67':
                    Block_id = 54
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert106 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert106)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line68' or line['Title'] == 'Line69':
                Block_id = 54
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert107 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert107)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line68':
                    Block_id = 64
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert108 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert108)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line69':
                    Block_id = 55
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert109 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert109)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line70' or line['Title'] == 'Line71':
                Block_id = 55
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert110 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert110)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line70':
                    Block_id = 65
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert111 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert111)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line72' or line['Title'] == 'Line73' or line['Title'] == 'Line74':
                Block_id = 61
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert112 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert112)
                # commiting the connection then closing it.
                connection.commit()


                if line['Title'] == 'Line73':
                    Block_id = 80
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert113 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert113)
                    # commiting the connection then closing it.
                    connection.commit()

                if line['Title'] == 'Line74':
                    Block_id = 62
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert114 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert114)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line75' or line['Title'] == 'Line76':
                Block_id = 62
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert115 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert115)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line76':
                    Block_id = 63
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert116 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert116)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line77' or line['Title'] == 'Line78':
                Block_id = 63
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert117 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert117)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line78':
                    Block_id = 64
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert118 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert118)
                    # commiting the connection then closing it.
                    connection.commit()

            if line['Title'] == 'Line79' or line['Title'] == 'Line80':
                Block_id = 64
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert119 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert119)
                # commiting the connection then closing it.
                connection.commit()

                if line['Title'] == 'Line80':
                    Block_id = 65
                    writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                     datetime_object + timedelta(
                                         seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                     int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                    insert120 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                        int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                        datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                    # executing the quires
                    cursor.execute(insert120)
                    # commiting the connection then closing it.
                    connection.commit()


            if line['Title'] == 'Line81' or line['Title'] == 'Line82':
                Block_id = 65
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert121 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert121)
                # commiting the connection then closing it.
                connection.commit()

            if line['Title'] == 'Line83' or line['Title'] == 'Line84':
                Block_id = 80
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert122 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert122)
                # commiting the connection then closing it.
                connection.commit()

            if line['Title'] == 'Line85' or line['Title'] == 'Line86':
                Block_id = 81
                writer.writerow([int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                                 datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                                 int(cap.get(cv2.CAP_PROP_POS_FRAMES))])
                insert123 = "INSERT INTO `PET_RawData_Jetson` (`ObjectID`, `BlockID`, `Mode`, `LineID`, `Time`, `FarameNo`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    int(id), Block_id, obj_name, int(line['Title'].split("Line")[1]),
                    datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1 / fpst),
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)), camera)

                # executing the quires
                cursor.execute(insert123)
                # commiting the connection then closing it.
                connection.commit()




            return True
    return False



borders  = [
     {'Title' : 'Line13', 'Cords' : [(365,244), (820,240)]},
     {'Title' : 'Line14', 'Cords' : [(30,555), (1263,555)]},
     {'Title' : 'Line15', 'Cords' : [(280,380), (193,508)]},
     {'Title' : 'Line16', 'Cords' : [(969,293), (1263,508)]},
     {'Title': 'Line17', 'Cords': [(260,210), (385,210)]},
     {'Title': 'Line18', 'Cords' : [(170,150), (170,350)]},

     {'Title' : 'Line19', 'Cords' : [(443,508), (400,720)]},
     {'Title': 'Line20', 'Cords': [(647, 508), (690, 720)]},
     {'Title': 'Line21', 'Cords': [(1082, 379), (1283, 379)]},
     {'Title': 'Line22', 'Cords': [(1180, 220), (1250, 350)]},
    {'Title': 'Line29', 'Cords': [(820, 240), (1050, 240)]},
     {'Title': 'Line30', 'Cords': [(170,350), (30,508)]}
]


def draw_borders(borders, img):
    for line in borders:
        img = cv2.line(img, line['Cords'][0], line['Cords'][1], (255,0,0), 3)
    return img

#dic1={}
def movement_counter(centerpoints, obj_name, id):
    connection = pymysql.connect(host="********", user="********", passwd="********", database="********")
    cursor = connection.cursor()
    for border in borders:
        p1 = Point(*centerpoints[0])
        q1 = Point(*centerpoints[1])
        p2 = Point(*border['Cords'][0])
        q2 = Point(*border['Cords'][1])
        if doIntersect(p1, q1, p2, q2):
            insert60 = "INSERT INTO `Movement_RawData_Jetson` (`ObjectID`, `Mode`, `LineID`, `Time`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}');".format(
                int(id), obj_name, int(border['Title'].split("Line")[1]),
                datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), camera)

            # executing the quires
            cursor.execute(insert60)
            # commiting the connection then closing it.
            connection.commit()


            if (border['Title'] == 'Line13' and Point(*centerpoints[0]).y > 240) or (border['Title'] == 'Line17'):
                insertcount = "INSERT INTO `Count_RawData` (`ObjectID`, `Mode`, `LineID`, `Time`, `Camera`) VALUES ('{}', '{}', '{}', '{}', '{}');".format(
                    int(id), obj_name, int(border['Title'].split("Line")[1]), datetime_object + timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) * 1/fpst), camera
                )

                # executing the quires
                cursor.execute(insertcount)
                # commiting the connection then closing it.
                connection.commit()


            return True
    return False





# end_path_mov = "MOVEMENT"+split_end[0]+".csv"
#
# path_mov = os.path.join(filepath, end_path_mov)
#
#
#
# f = open(path_mov, 'w')
# writerf = csv.writer(f)


# def movement_write():
#     for item in dic1.items():
#         label = ""
#         street = ""
#
# #H St - West
#
#         if item[1][0]== 'Line1' and item[1][-2]== 'Line2':
#             label = "Through-Movement"
#             From = "H St-West"
#             To = "H St-Eest"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line5' and item[1][-2]== 'Line6':
#             label = "Right Turn"
#             From = "H St-West"
#             To = "Broadway-South"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line1' and item[1][-2]== 'Line4':
#             label = "Left Turn"
#             From = "H St-West"
#             To = "Broadway-North"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0] == 'Line1' and item[1][-2] == 'Line9':
#             label = "Left Turn"
#             From = "H St-West"
#             To = "Broadway-North"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
# #H St - East
#
#         if item[1][0]== 'Line2' and item[1][-2]== 'Line1':
#             label = "Through-Movement"
#             From = "H St-East"
#             To = "H St-West"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line9' and item[1][-2]== 'Line1':
#             label = "Through-Movement"
#             From = "H St-East"
#             To = "H St-West"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#
#
#         if item[1][0]== 'Line2' and item[1][-2]== 'Line6':
#             label = "Left Turn"
#             From = "H St-East"
#             To = "Broadway-South"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line2' and item[1][-2]== 'Line4':
#             label = "Right Turn"
#             From = "H St-East"
#             To = "Broadway-North"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
# #Broadway-South
#
#         if item[1][0]== 'Line7' and item[1][-2]== 'Line8':
#             label = "Through-Movement"
#             From = "Broadway-South"
#             To = "Broadway-North"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line3' and item[1][-2]== 'Line1':
#             label = "Left Turn"
#             From = "Broadway-South"
#             To = "H St-West"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line3' and item[1][-2]== 'Line2':
#             label = "Right Turn"
#             From = "Broadway-South"
#             To = "H St-East"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
# #Broadway-North
#         if item[1][0]== 'Line4' and item[1][-2]== 'Line6':
#             label = "Through-Movement"
#             From = "Broadway-North"
#             To = "Broadway-South"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line10' and item[1][-2]== 'Line6':
#             label = "Through-Movement"
#             From = "Broadway-North"
#             To = "Broadway-South"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line4' and item[1][-2]== 'Line2':
#             label = "Left Turn"
#             From = "Broadway-North"
#             To = "H St-East"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line4' and item[1][-2]== 'Line7':
#             label = "Left Turn"
#             From = "Broadway-North"
#             To = "H St-East"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#         if item[1][0]== 'Line4' and item[1][-2]== 'Line8':
#             label = "Left Turn"
#             From = "Broadway-North"
#             To = "H St-East"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#
#
#
#         if item[1][0]== 'Line10' and item[1][-2]== 'Line1':
#             label = "Right Turn"
#             From = "Broadway-North"
#             To = "H St-West"
#             writerf.writerow([item[0], item[1][1], item[1][0], item[1][-2], label, From, To])
#

# Draw the Final Results
#def draw_results(img):
#    x = 100
#    y = 100
#    offset = 50
#    for line_name, line_counter in object_counter.items():
#        Text = line_name + " : " + ' '.join([f"{label}={count}" for label, count in line_counter.items()])
#        cv2.putText(img, Text, (x,y), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
#        y = y+offset
#    return img



#def draw_blockid(img):

        #img=cv2.putText(img, '1', (375,290), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '2', (525,290), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '3', (675,290), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '4', (825,290), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '5', (975,290), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '6', (225,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '7', (375,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '8', (525,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '9', (675,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '10', (825,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '11', (975,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '12', (1125,375), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '13', (225,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '14', (375,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '15', (525,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '16', (675,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '17', (825,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '18', (975,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        #img=cv2.putText(img, '19', (1125,525), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)

        #return img


# Function to calculate delta time for FPS when using cuda
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#centeroid=[]


# Draw the boxes having tracking indentities 
def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape 
    # Cleaning any previous Enteries
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) +offset[0]  for i in box]  
        box_height = (y2-y1)
        #center = (int((x2+x1)/ 2), int((y1+y2)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in set(data_deque):  
          data_deque[id] = deque(maxlen= 100)

        color = compute_color_for_labels(object_id[i])
        obj_name = class_names[object_id[i]]
        if obj_name == 'person':
            center = (int((x2+x1)/ 2), int((y2+y2)/2))
        else:
            center = (int((x2 + x1) / 2), int((y1 + y2) / 2))
        label = '%s' % (obj_name)
        #centeroid.append(center)
        
        data_deque[id].appendleft(center) #appending left to speed up the check we will check the latest map
        UI_box(box, img, label=label + str(id), color=color, line_thickness=3, boundingbox=True)

        if len(data_deque[id]) >=2:
            update_counter(centerpoints = data_deque[id], obj_name = obj_name, id=id)
            movement_counter(centerpoints = data_deque[id], obj_name = obj_name, id=id)
            

    return img


# Tracking class to integrate Deepsort tracking with our detector
class Tracker():
    def __init__(self, filter_classes=None, model='yolox-s', ckpt='wieghts/yolox_s.pth'):
        self.detector = Predictor(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_classes = filter_classes
    def update(self, image, visual = True, logger_=True):
        height, width, _ = image.shape 
        _,info = self.detector.inference(image, visual=False, logger_=logger_)
        outputs = []
        
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            objectids = []
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_classes:
                    if class_names[class_id] not in set(filter_classes):
                        continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])  
                objectids.append(info['class_ids'])             
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, info['class_ids'],image)
            data = []
            if len(outputs) > 0:
                if visual:
                    if len(outputs) > 0:
                        bbox_xyxy =outputs[:, :4]
                        identities =outputs[:, -2]
                        object_id =outputs[:, -1]
                        image = draw_boxes(image, bbox_xyxy, object_id,identities)
            return image, outputs



if __name__=='__main__':

    # Set up logging
    log_dir = '/mnt/beegfs/home/salehipo/Yolox_deepsort/jetson/8005/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'processed_files.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    #tracker = Tracker(filter_classes=None, model='yolox-s', ckpt='weights/yolox_s.pth')    # instantiate Tracker
    directory = "/mnt/beegfs/itmc/jetson/8005"
    files = glob.glob(os.path.join(directory, '*'))
    files = sorted(files, key=os.path.getmtime)

    #start_index = files.index('/mnt/beegfs/itmc/8005/2023_04_01_06h-07h.avi')
    #end_index = files.index('/mnt/beegfs/itmc/8005/2023_04_01_11h-12h.avi')

    #list = []
    #list1=[]
    for file in (files):
        try:
            tracker = Tracker(filter_classes=None, model='yolox-s', ckpt='weights/yolox_s.pth')  # instantiate Tracker
            connection = pymysql.connect(host="********", user="********", passwd="********", database="********")
            cursor = connection.cursor()

            with open(log_file, 'r') as f:
                if file in f.read():
                    logging.info('Skipping file %s (already processed)', file)
                    continue

            first = file.split("/")
            second = first[6]
            # Splitting nmae of file
            time_st = second.split("_")
            year = time_st[0]  # time_st[0] = 2022
            date = year + "-" + time_st[1] + "-" + time_st[2]  # 2022-11-05
            # Extracting time from file name
            start = time_st[3]
            start_hour = start[:2]
            time = start_hour + ":" + "00" + ":" + "00"
            h = date + " " + time
            # Convertting date from string to datetime
            datetime_object = datetime.strptime(h, '%Y-%m-%d %H:%M:%S')
            datetime_object = datetime_object - timedelta(seconds=1.5)
            # list.append(files)
            print(file)
            # video = os.path.join(directory, files)
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            fpst = cap.get(cv2.CAP_PROP_FPS)
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)
            length = int(cv2.VideoCapture.get(cap, property_id))

            filepath = "/mnt/beegfs/home/salehipo/Yolox_deepsort/jetson/8005"
            split_end = second.split(".")
            end_path_pet = "PET" + split_end[0] + ".csv"
            path_pet = os.path.join(filepath, end_path_pet)
            f = open(path_pet, 'w')
            writer = csv.writer(f)

            #video_path = files[i]
            path_video = os.path.join(filepath,second)

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            vid_writer = cv2.VideoWriter(path_video, fourcc, fps, (int(width), int(height)))
            print(fps)

            frame_count = 0
            fps = 0.0
            logging.info('Processing file %s', file)
            while True:
                ret_val, frame = cap.read()  # read frame from video
                #t1 = time_synchronized()
                if ret_val:
                    frame, bbox = tracker.update(frame, visual=True, logger_=False)  # feed one frame and get result
                    frame = draw_lines(lines, img=frame)
                    frame = draw_borders(borders, img=frame)
                    # movement_write()
                    # frame = draw_results(img= frame)
                    # frame = draw_blockid(img= frame)
                    # cv2.imshow('frame', frame)
                    vid_writer.write(frame)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
                    # fps = cap.get(cv2.CAP_PROP_FPS)
                    #fps = (fps + (1. / (time_synchronized() - t1))) / 2

                    # print(fps)
                else:
                    break
            # print(dic1)

            # Write the name of the processed file to the log file
            with open(log_file, 'a') as f:
                f.write(file + '\n')

        except Exception as e:
            logging.info('Failed Processing Completely file %s', file)
            continue


        #movement_write()
        connection.close()
        cap.release()
        vid_writer.release()
        cv2.destroyAllWindows()

    # for key, value in data_deque.items():
    # print(key, ' : ', value)

    print("Complete!")

#for key, value in data_deque.items(): 
    #print(key, ' : ', value)

    
