#
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md
#
# sudo systemctl status track_detect.service
# sudo systemctl restart track_detect.service
# sudo systemctl disable track_detect.service
# sudo systemctl enable track_detect.service
# sudo systemctl stop track_detect.service
# sudo systemctl start track_detect.service
# tail -f /tmp/jetson.log
#
# Importing all the necessary modules
from typing import Counter
import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 
from datetime import datetime
from tracker import *
from support_functions import *
import pandas as pd
import paho.mqtt.client as mqtt
import sys;
import logging;
import json;

def on_connect(client, userdata, flags, rc):
   if rc==0:
          client.connected_flag=True #set flag
          logging.debug("paho mqtt client connected ok")
   elif rc==5:
          logging.debug("paho mqtt client not connected, authentication failure")
          client.bad_connection_flag=True
   else:
          logging.debug("paho mqtt client not connected, returned code=%d",rc)
          client.bad_connection_flag=True

logging.basicConfig(filename='/tmp/jetson.log', level=logging.DEBUG)

client_name='Jetson'
client = mqtt.Client(client_name)
host='130.191.161.21' # broker address
client.connected_flag=False
client.bad_connection_flag=False
client.on_connect=on_connect  # bind callback function
client.username_pw_set(username="starlab",password="starlab!")
client.connect(host, port=1883, keepalive=60, bind_address="")

client.loop_start()  #Start loop

while not client.connected_flag and client.bad_connection_flag: #wait in loop
    logging.debug("In wait loop")
    time.sleep(1)

logging.debug('client.bad_connection_flag: %r',client.bad_connection_flag)
logging.debug('client.connected_flag: %r\n\n',client.connected_flag)

msg = f"started"
topic = f"pelco/jetson"
result = client.publish(topic, msg)
status = result[0]
if status == 0:
   logging.debug(f"Send `{msg}` to topic `{topic}`")
else:
   logging.debug(f"Failed to send message to topic {topic}")
   sys.exit()

tracker = EuclideanDistTracker()

# For FPS text need time
timeStamp=time.time()
fpsFilt=0

# Model 
#net = jetson.inference.detectNet(argv=["--model=/media/jetson/UGUR_USB_C/models/epoch_max/ssd-mobilenet.onnx", "--labels=/media/jetson/UGUR_USB_C/models/april_model/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)
net = jetson.inference.detectNet(argv=["--model=/home/iot/jetson-inference/python/training/detection/ssd/models/last_model/ssd-mobilenet.onnx", "--labels=/home/iot/jetson-inference/python/training/detection/ssd/models/last_model/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.5)

# Picture Size for display
dispW=1280
dispH=720
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX  # Font for the texts

# Video Settings
#cap=cv2.VideoCapture('file:///home/jetson/Desktop/Sample_Video/suv_truck.mp4') #can be changed with any other video source or a file


cap=cv2.VideoCapture('rtsp://ued:uU8xwmin@sunray.sdsu.edu/stream2')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cap.set(cv2.CAP_PROP_FPS, int(3))

result = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'),10, (1280,720))
detected_objects = []
while True:

    ret, img = cap.read()

    if ret == False:
        condition = False
        break
    
    # Image Info to OpenCV
    height=img.shape[0]
    width=img.shape[1]

    # Changing the color to Needed format for detection for Nvidia Jetson
    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    # Converting the image from Numpy to CUDA
    frame=jetson.utils.cudaFromNumpy(frame)

    # Getting Detections for each Frame
    detections=net.Detect(frame, width, height)

    
    detection_list = []
    
    for detect in detections:
        # TODO: Uncomment this for all the info for each detection
        print(detect)
        
        # Get all the information from the detection class
        
        ID=detect.ClassID
        top=detect.Top
        left=detect.Left
        bottom=detect.Bottom
        right=detect.Right
        width_d = detect.Width
        height_d = detect.Height
        item=net.GetClassDesc(ID)
        center = detect.Center
        center_x = int(center[0])
        center_y = int(center[1])
        confidence = detect.Confidence
        
        # Getting the center locations in integer format in a tuple
        center = (center_x,center_y)
        
        # Rendering the image
        # * Boundary Box
        img = cv2.rectangle(img,(int(left),int(top)),(int(right),int(bottom)),(0,255,0),2)
        
        # * Center Circles
        img = cv2.circle(img, center, 1, (255,255,255), 10)
        # * Class ID and Confidence Text
        cv2.putText(img,str(item)+" "+str(round(detect.Confidence,2)),(int(left)+75,int(top)-15),font,1,(255,255,0),2)

        
        detection_list.append([int(left),int(top),int(width_d),int(height_d),item])
        
        currentTime = datetime.now()
        
        print(item)

        with open("/home/iot/Desktop/app/Detections.txt", "a") as f:
            f.write("The current timestamp is: " + str(datetime.now()))
            f.write("\n")
            f.write("The detection Details are: " + str(detect))
            f.write("\n")
            f.write("Detected is: " + str(item))
            f.write("\n")
            f.close()
    
        data = {
           "id": format(detect.ClassID),
           "top": "{:.2f}".format(detect.Top),
           "left": "{:.2f}".format(detect.Left),
           "bottom": "{:.2f}".format(detect.Bottom),
           "right": "{:.2f}".format(detect.Right),
           "width_d": "{:.2f}".format(detect.Width),
           "height_d": "{:.2f}".format(detect.Height),
           "class": format(net.GetClassDesc(ID)),
           #"center": format(detect.Center),
           "center_x": format(int(center[0])),
           "center_y": format(int(center[1])),
           "confidence": "{:.2f}".format(detect.Confidence),
           #"class": format(category_index[classes+1]['name']),
           #"box": [format(x_min_disp), format(y_min_disp), format(x_max_disp), format(y_max_disp)],
           #"date": format(t.month) + '/' + format(t.day) + '/' + format(t.year),
           "time": format(currentTime.hour) + ':' + format(currentTime.minute) + ':' + format(currentTime.second)
           #"frame": ['height:'+format(height), 'width:'+format(width)],
           #"score": "{:.2f}".format(score),
           #"inference_time": "{:.4f}".format(inference_time)
        }
        msg = json.dumps(data)
        topic = f"pelco/jetson"
        result = client.publish(topic, msg)
        status = result[0]
        if status == 0:
            logging.debug(f"Send `{msg}` to topic `{topic}`")
        else:
            logging.debug(f"Failed to send message to topic {topic}")

            
    boxes_ids = tracker.update(detection_list)
    
    for box_id in boxes_ids:
        x, y, w, h, id, vehicle_type  = box_id
        
        temp = next((obj for obj in detected_objects if obj.id == id), None)
        
        if temp == None:
            detected_objects.append(Detected(x, y, w, h, id, vehicle_type))

        else:
            prev_cx, prev_cy = temp.cx, temp.cy

            print(prev_cx,prev_cy)
            
            temp.update(x, y, w, h)

            print(temp.cx,temp.cy)

            img = cv2.circle(img, (prev_cx,prev_cy), 1, (255,0,0), 10)
            img = cv2.arrowedLine(img,(prev_cx,prev_cy),(temp.cx,temp.cy),(0,255,0),2)
            #cv2.putText(img,"Motion Vector",(prev_cx+10,prev_cy),font,1,(255,255,0),2)
            temp.get_ingress(datetime.now().strftime("%H:%M:%S:%f"))
            temp.get_exgress(datetime.now().strftime("%H:%M:%S:%f"))
        
        
        cv2.putText(img,"ID:"+str(id),(x,y -15),font,1,(2550,0),2)

    exgress_times = {}
    ingress_times = {}
    pet_calc = {}
    if len(boxes_ids)>1:
        for box_id in boxes_ids:
            _,_,_,_,id,_ = box_id
            temp = next((obj for obj in detected_objects if obj.id == id), None)
            if temp != None:
                exgress_times[id] = temp.exgress
                ingress_times[id] = temp.ingress
        
        # ! Just Showing some stuff will delete later
        print("This is exgress times = ")
        print(exgress_times)
        print("This is ingress times = ")
        print(ingress_times)
        
        print(datetime.now())
        tp2 = 650
        for (k1,v1) in exgress_times.items():
            for (k2,v2) in ingress_times.items():
                if k1 != k2:
                    if v1 != None and v2 != None:
                        pet_calc[k2] = v2 - v1
                        pet_text = f"PET: ID{k2}->ID{k1}={abs(round((v2-v1).total_seconds(),2))}s"
                        print(pet_text)
                        cv2.putText(img,pet_text,(900,tp2),font,1,(0,0,255),2)
                        tp2 += 30

    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps

    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(255,255,0),2)

    # Yellow Box
    overlay = img.copy()
    cv2.rectangle(overlay,(150,110),(900,680),(0,255,255),-1)
    alpha = 0.4
    img = image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Uncomment for saving the output file 
    #result.write(img)
    #cv2.imshow("Frame",img)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    
cap.release()
cv2.destroyAllWindows()
client.loop_stop()

df = pd.DataFrame(columns=['id','vehicle_type','ingress_time','egress_time'])

for x in detected_objects:
    if x.ingress != None or x.exgress != None:

        new_row = {'id':x.id,'vehicle_type':x.vehicle_type, 'ingress_time':x.ingress.strftime("%H:%M:%S:%f"),'egress_time':x.exgress.strftime("%H:%M:%S:%f")}
        df = df.append(new_row,ignore_index=True)
        #print(f"Object ID {x.id}, Object Type = {x.vehicle_type}, Ingress = {x.ingress.strftime("%H:%M:%S:%f")}, Egress = {x.exgress.strftime("%H:%M:%S:%f")}")

df.to_csv("results.csv",index=False)
print(df.set_index('id').dropna())
