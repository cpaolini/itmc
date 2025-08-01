import pymysql
from datetime import datetime
import time

connection = pymysql.connect(
    host="notos.sdsu.edu",
    user="itmc",
    passwd="TiUSiLNXRGASND2@",
    database="itmc"
)

while not connection.open():
    print("Connection not established...waiting 5 seconds")
    time.sleep(5000)

cursor = connection.cursor()

time = datetime.now()
bounds = str([(100, 150), (200, 250)])
cameraID = 1
vehicleID = 101
classID = 3

query = """
INSERT INTO vehicle_instance (time, bounds, cameraID, vehicleID, classID)
VALUES (%s, %s, %s, %s, %s)
"""

n = cursor.execute(query, (time, bounds, cameraID, vehicleID, classID))
print("the value of n is ", n)

connection.commit()
print("connection.commit() has been run")

cursor.close()
print("cursor.close() has been run")

connection.close()
print("connection.close() has been run")