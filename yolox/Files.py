# import required module
import os

# assign directory
directory = "/mnt/beegfs/paolini-notos/ITMC/Hourly_Intersection_Recordings/8005"
list = []
for filename in os.listdir(directory):
    time = filename.split("_")
    if int(time[1]) >=12:
        list.append(filename)

print(list)
print(len(list))