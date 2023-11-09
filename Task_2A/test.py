import numpy as np
import cv2
from cv2 import aruco
import math
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################

def detect_ArUco_details(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns two dictionaries where one
    contains details regarding the center coordinates and orientation of the marker
    and the second dictionary contains values of the 4 corner coordinates of the marker. 
    
    First output: The dictionary ArUco_details_dict should should have the id of the marker 
    as the key and the value corresponding to that id should be a list containing the following details
    in this order: [[center_x, center_y], angle from the vertical]     
    This order should be strictly maintained in the output
    Datatypes:
    1. id - int
    2. center coordinates - int
    3. angle - int, x and y coordinates should be combined as a list for each corner

    Second output: The dictionary ArUco_corners should contain the id of the marker as key and the
    corresponding value should be an array of the coordinates of 4 corner points of the markers
    Datatypes:
    1. id - int
    2. corner coordinates - each coordinate value should be float, x and y coordinates should 
    be combined as a list for each corner

    Input Arguments:
    ---
    image :	[ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    ArUco_details_dict : { dictionary }
            dictionary containing the details regarding the ArUco marker

    ArUco_corners : { dictionary }
            dictionary containing the details regarding the corner coordinates of the ArUco marker
    
    Example call:
    ---
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(image)

    Example output for 2 markers in an image:
    ---
    * ArUco_details_dict = {9: [[311, 490], 0], 3: [[158, 175], -22]}
    * ArUco_corners = 
       {9: array([[211., 389.],
       [412., 389.],
       [412., 592.],
       [211., 592.]], dtype=float32), 
       3: array([[109.,  46.],
       [284., 118.],
       [207., 304.],
       [ 33., 232.]], dtype=float32)}
    """    
    ArUco_details_dict = {}
    ArUco_corners = {}
    
    ##############	ADD YOUR CODE HERE	##############
    #marker_size = 6
    #total_markers = 250
    #image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    #key = getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    print(parameters)
    #parameters = cv2.aruco.DetectorParameters_create()
    print(aruco_dict)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict,parameters= cv2.aruco.DetectorParameters())
    print(cv2.aruco.detectMarkers(gray, aruco_dict,parameters=parameters))
    print("chill")
    if ids is None:
        print("Nahh")
    else:
        print(corners)
    for i in range(len(ids)):
        marker_id = ids[i][0]
        print(corners[i][0,:,0])

        # Extract center coordinates and angle
        center_x = int(np.mean(corners[i][0, :, 0]))
        center_y = int(np.mean(corners[i][0, :, 1]))
        print(center_x)
        print(center_y)

        # Calculate angle from the vertical (assuming a vertical marker)
        angle = 90 - np.degrees(np.arctan2(corners[i][0, 3, 1] - corners[i][0, 0, 1],
                                      corners[i][0, 3, 0] - corners[i][0, 0, 0])) 

        # Store details in the dictionary
        ArUco_details_dict[marker_id] = [[center_x, center_y], angle]

        # Store corner coordinates in the dictionary
        ArUco_corners[marker_id] = corners[i][0].tolist()
    '''
            center, rotation, _, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size)

            ArUco_details_dict[ids[i][0]] = {'center': center[0], 'rotation': rotation[0]}

            corners_list = corners[i][0].tolist()

            ArUco_corners[ids[i][0]] = {'top_left': corners_list[0], 'top_right': corners_list[1],
                                       'bottom_right': corners_list[2], 'bottom_left': corners_list[3]}

            '''

    ##################################################
    
    return ArUco_details_dict, ArUco_corners 

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE CODE BELOW #########	

def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):

    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0,0,255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        cv2.line(image,center,(tl_tr_center_x, tl_tr_center_y),(255,0,0),5)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        angle = details[1]
        cv2.putText(image,str(angle),(center[0]-display_offset,center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return image

if __name__ == "__main__":

    # path directory of images in test_images folder
    img_dir_path = "public_test_cases/"

    marker = 'aruco'

    for file_num in range(0,2):
        img_file_path = img_dir_path +  marker + '_' + str(file_num) + '.png'

        # read image using opencv
        img = cv2.imread(img_file_path)

        print('\n============================================')
        print('\nFor '+ marker  +  str(file_num) + '.png')
   
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        print("Detected details of ArUco: " , ArUco_details_dict)

        #displaying the marked image
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners) 
        cv2.imshow("Marked Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()