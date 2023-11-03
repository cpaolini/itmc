
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import CustomObjectScope
from matplotlib import pyplot as plt
import os
import time


video_path = "D:/Computer__Vision/DataSET/test___tilt__pan___dataSET/tilt__pan___test__NEW.mp4"
#model_path = "D:/Computer__Vision/DataSET/test___tilt__pan___dataSET/models___/best_model__same__retrain__20PER__ACCur__TILT__PAN__DeepLabV3__directMasking__Corrected/best_model.h5"
#output_path = 'D:/Computer__Vision/DataSET/test___tilt__pan___dataSET/models___/best_model__same__retrain__20PER__ACCur__TILT__PAN__DeepLabV3__directMasking__Corrected/output_video__8008_512__GRID.mp4'
model_path = "C:/Users/dheer/Downloads/model (5).h5"

output_folder = "C:/Users/dheer/Downloads/outputs_2/"

output_path = output_folder + "outputGRID.mp4"

#output_path = "C:/Users/dheer/Downloads/outputs/outputGRID.mp4"


###########################
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
###########################


#################
# init variables
IMG_SIZE = 512
#################


video_capture = cv2.VideoCapture(video_path)
# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("DEBUG: fps --> ", fps)
print("DEBUG: width --> ", width)
print("DEBUG: height --> ", height)

# Load your road segmentation model (replace with your actual model loading code)
def load_model____():
    # Compile the model with the same optimizer and loss-function
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(model_path)

    return model

def weighted_img(img, initial_img, alpha=0.5, beta=0.5, gamma=0):
    # print("DEBUG :: (weighted_img) img shape dtype", img.shape, img.dtype)
    # print("DEBUG :: (weighted_img) initial_img shape dtype", initial_img.shape, initial_img.dtype)
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
def segment_road(image, model):

    """ Prediction """
    x =image
    y_pred = model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)  ## this makes the dimension (512, 512, 1)  --> (512, 512)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)  ## (512, 512, 1)
    zero_image = np.zeros_like(y_pred)
    y_pred = np.concatenate([y_pred, zero_image, zero_image], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    x = np.squeeze(x, axis=0)

    return y_pred


def preProcess__(mask, inputImage, frame_count):
    KERNEL_SIZE = 20

    # Load the road segmentation mask image
    # mask is predicted in the above code

    mask__gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # print("DEBUG :: gray mask SHAPE --> ",mask__gray.shape)

    # Apply morphological openings to remove all small raod openings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask_changed = cv2.morphologyEx(mask__gray, cv2.MORPH_OPEN, kernel)

    # Apply morphological closing to fill gaps between the road segments
    mask_changed = cv2.morphologyEx(mask__gray, cv2.MORPH_CLOSE, kernel)
    # print("DEBUG :: gray mask_changed SHAPE --> ",mask_changed.shape)

    # Remove non-road objects using thresholding and connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_changed)

    ## print("\n>>>>>> DEBUG :: <<<<<<<< \n", num_labels)
    sizes = stats[:, -1]
    mask__output = np.zeros_like(mask_changed)
    for i in range(1, num_labels):
        if sizes[i] > 1000:  # set the minimum size of a road segment
            mask__output[labels == i] = 255


    # Smooth the edges of the road segments
    maskFinal = cv2.medianBlur(mask__output, 9)


    # Convert binary mask to color mask
    # maskFinal is GrayScale mask of single color channel
    # mask_color is 3-channel color mask
    mask_color = cv2.cvtColor(maskFinal, cv2.COLOR_GRAY2BGR)
    mask_color[np.where((mask_color == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

    # resize the mask_color to (1280, 720)
    # resize the maskFinal to (1280, 720)
    final_mask = cv2.resize(mask_color, (1280, 720))
    final_mask_Gray = cv2.resize(maskFinal, (1280, 720))
    output = weighted_img(final_mask, inputImage)
    """ [DHEERAJ]:: output contains the predicted mask overlapped on the inputImage at 1280 x 720 """
    

    # Detect the contours from this mask
    contours, hierarchy = cv2.findContours(final_mask_Gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect the hull using the cv2.convexHull()
    hull_img = inputImage.copy()
    # cnt_img = inputImage.copy()

    hull = cv2.convexHull(contours[0])

    cv2.drawContours(hull_img, [hull], 0, (0, 0, 220), 3)
    """ [DHEERAJ]:: hull_img will contain the road boundary detected from the contours"""
    
    
    """DEBUG+++++++++++++++++++++++++++++++++++++++++++++++++++"""
    # Construct the output file path (e.g., frame_001.jpg, frame_002.jpg, etc.)
    #frame_filename = os.path.join(output_folder, f'frame_{frame_count:03d}.jpg')

    # Save the frame as an image
    #cv2.imwrite(frame_filename, hull_img)
    
    """DEBUG+++++++++++++++++++++++++++++++++++++++++++++++++++"""
    

    # cv2.drawContours(cnt_img, [contours[0]], 0, (0,0,220), 3)
    # Draw Bounding rectangle on the contour detected

    imgCPY = inputImage.copy()

    x, y, w, h = cv2.boundingRect(contours[0])

    # Used in the next cell as a tuple
    bounding_rect = (x, y, w, h)

    # Draw the rectangle around the contour
    cv2.rectangle(imgCPY, (x, y), (x + w, y + h), (0, 255, 0), 3)

    ############################################
    # THIS if HULL implementation
    cv2.drawContours(imgCPY, [hull], 0, (0, 0, 220), 3)
    ###############################################



    # try to find the intersection points of the HULL and the bounding box

    x, y, w, h = cv2.boundingRect(contours[0])
    point1 = (x, y)
    point2 = (x + w, y + h)

    # print(point1, point2)

    points = np.array([point1, point2])

    rect_points = cv2.minAreaRect(points)
    box_points = cv2.boxPoints(rect_points).astype(int)


    polygon1 = np.array(box_points, dtype=np.float32)
    polygon2 = np.array(hull, dtype=np.float32)
    # Find the intersection points
    retval, intersection_pts = cv2.intersectConvexConvex(polygon1, polygon2)


    # Find the intersection point in the top part of the image
    intersection_pts = intersection_pts.reshape(-1, 2)  # Reshape the array

    sorted_points = sorted(intersection_pts, key=lambda x: (x[0], x[1]))

    ## print(sorted_points)

    first_point = min(sorted_points, key=lambda x: x[1])

    # print("First point from the upper boundary:", first_point)

    upper_point = ()
    upper_point = upper_point + (first_point[0],)
    upper_point = upper_point + (first_point[1],)


    point3 = first_point
    point4 = (528.0, 720)

    # Create a blank image
    imageBlank = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Set x-coordinate for all points
    # x_coordinate = int(528.0)
    x_coordinate = upper_point[0]

    # Generate y-coordinates of equally spaced points
    # y_coordinates = np.linspace(60.0, 720.0, 10)
    y_coordinates = np.linspace(upper_point[1], 720.0, 10)

    # store all these scatter points
    points_scatter = []

    # Draw scatter points on the image
    for y in y_coordinates:
        point = (int(x_coordinate), int(y))
        points_scatter.append(point)
        # cv2.circle(imageBlank, point, 5, (0, 0, 255), -1)


    """
    for point in points_scatter:
        print(point)
    """

    result = inputImage.copy()
    # Plot the points on the image
    for point in points_scatter:
        x, y = point
        cv2.circle(result, (x, y), 5, (255, 0, 255),
                   -1)  # Draw a filled circle with radius 3 and color red (BGR format)




    ####################################
    ####################################
    #### code for post-processing the mask into exact polygon shape
    ####################################
    ####################################

    # Find the contours of the mask
    contours, hierarchy = cv2.findContours(maskFinal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the polygon
    poly_mask = np.zeros_like(maskFinal)

    # Iterate over each contour and approximate it as a polygon
    for i in range(len(contours)):
        # Approximate the contour as a polygon with a suitable epsilon value
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)

        # Draw the polygon on the blank image
        cv2.drawContours(poly_mask, [approx], 0, (255), -1)

    # plt.imshow(poly_mask, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Convert POLY mask to color mask
    mask_color_poly_mask = cv2.cvtColor(poly_mask, cv2.COLOR_GRAY2BGR)
    mask_color_poly_mask[np.where((mask_color_poly_mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

    ### overlap this mask on the inmage
    ### Using the finalmask variable AGAIN !!!!!!
    final_mask = cv2.resize(mask_color_poly_mask, (1280, 720))
    output = weighted_img(final_mask, inputImage)
    
    """DEBUG+++++++++++++++++++++++++++++++++++++++++++++++++++"""
    # Construct the output file path (e.g., frame_001.jpg, frame_002.jpg, etc.)
    # frame_filename = os.path.join(output_folder, f'frame_{frame_count:03d}.jpg')

    # Save the frame as an image
    # cv2.imwrite(frame_filename, output)
    
    """DEBUG+++++++++++++++++++++++++++++++++++++++++++++++++++"""



    result = inputImage.copy()
    # Plot the points on the image
    for point in points_scatter:
        x, y = point
        cv2.circle(output, (x, y), 5, (255, 0, 255),
                   -1)  # Draw a filled circle with radius 3 and color red (BGR format)

    # plt.figure(figsize=[10,10])
    # plt.imshow(output)
    # plt.axis('off')
    # plt.show()

    # create the copy of the inputImage

    imageTest = inputImage.copy()


    """
    # Draw horizontal lines passing through each point on the image
    for point in points_scatter:
        x, y = point
        cv2.line(imageTest, (0, y), (imageTest.shape[1], y), (0, 0, 255), 5)
    """

    # Store the intersection points between the hull and each horizontal line
    intersection_points = []
    for point in points_scatter:
        x, y = point
        line = np.array([[0, y], [imageTest.shape[1], y]])
        _, intersection = cv2.intersectConvexConvex(hull.reshape(-1, 2), line.reshape(-1, 2))
        # _, intersection = cv2.intersectConvexConvex(contours[0].reshape(-1, 2), line.reshape(-1, 2))
        if intersection is not None:
            intersection_points.append(intersection)

    """
    # Print the intersection points
    for i, points in enumerate(intersection_points):
        # print(f"Intersection points for line {i + 1}:")
        for point in points:
            # print(point)
    """


    finalLST = []
    # points_catch = []
    for i, points in enumerate(intersection_points):
        ## print(f"Intersection points for line {i + 1}:")
        ## print('=======================================\n')
        ## print('DEBUG enum*** :: len(points)', len(points))
        points_catch = []
        for j in range(len(points)):
            ## print('DEBUG points*** :: len(points)', len(points[j]))
            for k in range(len(points[j])):
                ## print('DEBUG coRD*** :: len(points)', len(points[j][k]))
                cord_catch = ()
                # points_catch = []
                for l in range(len(points[j][k])):
                    ## print('DEBUG ele*** :: points[j][k][l] --> ', points[j][k][l])
                    cord_catch = cord_catch + (points[j][k][l],)
                    ## print(cord_catch)
                points_catch.append(cord_catch)
                ## print('i,j,k,l  --- >> ', i,j,k,l)
                ## print('DEBUG:: points_catch --> ',points_catch)
        finalLST.append(points_catch)
        ## print(finalLST)

        # cord_catch = (points[j][k][0], points[j][k][1])
        # point_catch[k].append(cord_catch)
        ## print(point_catch[k])

    # print('DEBUG :: len(finalLST) --> ', len(finalLST))
    # print('DEBUG :: len(finalLST[0]) --> ', len(finalLST[0]))
    # print('DEBUG :: finalLST[0][0] --> ', finalLST[0][0])
    # print('DEBUG :: finalLST[0][1] --> ', finalLST[0][1])

    # Draw lines between intersection points if the length of each intersection is > 1
    for i in range(len(finalLST)):
        if len(finalLST[i]) == 2:
            # print('YES !!')
            cv2.line(imageTest, (int(finalLST[i][0][0]), int(finalLST[i][0][1])),
                     (int(finalLST[i][1][0]), int(finalLST[i][1][1])), (255, 0, 0), 5)
            

    #################################
    # Scatter 8 horizontal points
    ##################################

    left_point = (0, 720)
    right_point = (1280, 720)

    # Create a blank image
    imageBlank = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Set y-coordinate for all points
    # x_coordinate = int(528.0)
    y_coordinate = left_point[1]

    # Generate x-coordinates of equally spaced points
    # x_coordinates = np.linspace(60.0, 720.0, 10)
    x_coordinates = np.linspace(left_point[0], 1280, 13)

    # store all these scatter points
    points_scatter_hor = []

    # Draw scatter points on the image
    for x in x_coordinates:
        point = (int(x), int(y_coordinate))
        points_scatter_hor.append(point)
        cv2.circle(imageBlank, point, 5, (0, 0, 255), -1)
        
    """
    for point in points_scatter_hor:
        print(point)
    """

    result_hor = inputImage.copy()
    # Plot the points on the image
    for point in points_scatter_hor:
        x, y = point
        cv2.circle(result_hor, (x, y), 5, (255, 0, 255),
                   -1)  # Draw a filled circle with radius 3 and color red (BGR format)

    # plt.figure(figsize=[10,10])
    # plt.imshow(result_hor)
    # plt.axis('off')
    # plt.show()

    # Plot the points on the image
    for point in points_scatter_hor:
        x, y = point
        cv2.circle(output, (x, y), 5, (255, 0, 255),
                   -1)  # Draw a filled circle with radius 3 and color red (BGR format)

    # plt.figure(figsize=[10,10])
    # plt.imshow(output)
    # plt.axis('off')
    # plt.show()

    # create the copy of the inputImage

    imageTest_hor = inputImage.copy()

    # Draw the hull on the image
    # cv2.drawContours(imageTest, [hull], 0, (0, 255, 0), 5)
    # cv2.drawContours(imageTest, [contours[0]], 0, (0, 255, 0), 5)

    # Store the intersection points between the hull and each vertical line
    intersection_points_hor = []
    for point in points_scatter_hor:
        x, y = point
        line = np.array([[x, 0], [x, imageTest.shape[0]]])
        _, intersection = cv2.intersectConvexConvex(hull.reshape(-1, 2), line.reshape(-1, 2))
        if intersection is not None:
            intersection_points_hor.append(intersection)

    """
    # Print the intersection points
    for i, points in enumerate(intersection_points_hor):
        print(f"Intersection points for line {i + 1}:")
        for point in points:
            print(point)
    """

    finalLST_hor = []
    # points_catch = []
    for i, points in enumerate(intersection_points_hor):
        # print(f"Intersection points for line {i + 1}:")
        # print('=======================================\n')
        # print('DEBUG enum*** :: len(points)', len(points))
        points_catch = []
        for j in range(len(points)):
            # print('DEBUG points*** :: len(points)', len(points[j]))
            for k in range(len(points[j])):
                # print('DEBUG coRD*** :: len(points)', len(points[j][k]))
                cord_catch = ()
                # points_catch = []
                for l in range(len(points[j][k])):
                    # print('DEBUG ele*** :: points[j][k][l] --> ', points[j][k][l])
                    cord_catch = cord_catch + (points[j][k][l],)
                    # print(cord_catch)
                points_catch.append(cord_catch)
                # print('i,j,k,l  --- >> ', i,j,k,l)
                # print('DEBUG:: points_catch --> ',points_catch)
        finalLST_hor.append(points_catch)
        ## print(finalLST_hor)

    # Draw lines between intersection points if the length of each intersection is > 1
    for i in range(len(finalLST_hor)):
        if len(finalLST_hor[i]) == 2:
            # print('YES !!')
            cv2.line(imageTest, (int(finalLST_hor[i][0][0]), int(finalLST_hor[i][0][1])),
                     (int(finalLST_hor[i][1][0]), int(finalLST_hor[i][1][1])), (255, 255, 0), 5)

    ###############################################################################################################

    ################################
    Lines = []
    ################################

    # print("DEBUG :: TESTING finalLST traversing =============================")
    finalLST_line_info  = []
    for idx in range(len(finalLST)):
        if len(finalLST[idx]) == 2:
            # print("DEBUG :: ==>  cord --> ", int(finalLST[idx][0][0]), int(finalLST[idx][0][1]))
            # print("DEBUG :: ==>  cord --> ", int(finalLST[idx][1][0]), int(finalLST[idx][1][1]))
            line_info = {
                 "Title": f"Horizontal Line {idx}",
                 "Cords": [(int(finalLST[idx][0][0]), int(finalLST[idx][0][1])), (int(finalLST[idx][1][0]), int(finalLST[idx][1][1]))]
             }
            finalLST_line_info.append(line_info)
            Lines.append(line_info)

    # print("Horizontal  Lines in line_info format:")
    # for line_info in finalLST_line_info:
    #     print(line_info)
    ###############################################################################################################

    ###############################################################################################################
    finalLST_hor_line_info = []
    # print("DEBUG :: TESTING finalLST_hor traversing =============================")
    for idx in range(len(finalLST_hor)):
        if len(finalLST_hor[idx]) == 2:
            # print("DEBUG :: ==>  cord --> ", int(finalLST_hor[idx][0][0]), int(finalLST_hor[idx][0][1]))
            # print("DEBUG :: ==>  cord --> ", int(finalLST_hor[idx][1][0]), int(finalLST_hor[idx][1][1]))
            line_info = {
                 "Title": f"Vertical Line {idx}",
                 "Cords": [(int(finalLST_hor[idx][0][0]), int(finalLST_hor[idx][0][1])), (int(finalLST_hor[idx][1][0]), int(finalLST_hor[idx][1][1]))]
             }
            finalLST_hor_line_info.append(line_info)
            Lines.append(line_info)

    # print("Vertical Lines in line_info format:")
    # for line_info in finalLST_hor_line_info:
    #     print(line_info)

    ###############################################################################################################
    # print("Lines Printing ========")
    # for line_info in Lines:
    #     print(line_info)

    redGRID = inputImage.copy()
    # Draw lines on the image
    for line_info in Lines:
        coords = line_info["Cords"]
        cv2.line(redGRID, coords[0], coords[1], (0, 0, 255), 2)  # Red color for lines

    # # Display the image with lines
    # cv2.imshow('Image with Lines', redGRID)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("DEBUG :: redGRID shape ==> ", redGRID.shape)

    # plt.figure(figsize=[10,10])
    # plt.imshow(imageTest)
    # plt.axis('off')
    # plt.show()

    # print("DEBUG :: =====================================\n")
    # print(len(finalLST))
    # print(len(finalLST_hor))

    return final_mask, final_mask_Gray, output, imageTest, redGRID, Lines  # returns the final_mask resized color mask (1280, 720)


# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load your road segmentation model
model = load_model____()


import csv
#csv_filename = "C:/Users/dheer/Downloads/outputs/inference_times.csv"
csv_filename = output_folder + "inference_times.csv"

with open(csv_filename, mode="w", newline="") as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(["frame_count", "Inference Time (seconds)"])
    
    
    frame_count = 0
    while True:
        
        # Start a timer at the begining of each step
        start_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        """Reading the image"""
        x = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        x = x / 255.0
        # print(x.shape)
        x = np.expand_dims(x, axis=0)  ## this expands the dimension and makes it into a batch/list of images
        # print(x.shape)



        # Perform road segmentation on the frame
        road_mask = segment_road(x, model)

        road_mask = cv2.resize(road_mask, (1280, 720))

        #frame = np.squeeze(frame, axis=0)
        # print("DEBUG :: (for weighted image !!) x shape, dtype", frame.shape, frame.dtype)
        # Apply the road mask to the frame
        processed_frame = weighted_img(road_mask, frame)


        maskFinal, maskFinal_Gray, segmented_frame, imageTest, redGRID, Lines = preProcess__(road_mask, frame, frame_count)

        inference_time = time.time() - start_time
        print(f"frame_count {frame_count}: Inference time: {inference_time} seconds")
        # Write the step and inference time to the CSV file
        csv_writer.writerow([frame_count, inference_time])
        
        
        # Write the processed frame to the output video file
        output_video.write(redGRID)

        # Increment the frame count
        frame_count += 1
        print('DEBUG: frame= ', frame_count)

        # Break the loop if all desired frames are processed
        # if frame_count >= num_frames:
        #    break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and output video writer
    video_capture.release()
    output_video.release()

    # Destroy all windows
    cv2.destroyAllWindows()