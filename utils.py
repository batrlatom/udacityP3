import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



######################################################################################
#   splitting dataset directly in the pandas
######################################################################################
def pandas_split(X, y):
    msk = np.random.rand(len(X)) < 0.9

    X_train = X[msk]
    X_valid = X[~msk]

    y_train = y[msk]
    y_valid = y[~msk]

    return X_train, X_valid, y_train, y_valid


######################################################################################
#   Load RGB images from a file
######################################################################################
def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


######################################################################################
#   Convert the image from RGB to YUV (This is what the NVIDIA model does)
######################################################################################
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


######################################################################################
#   provide  TF lambda layer
######################################################################################
def resize_normalize(image):
    import tensorflow as tf
    print("net lambda layers")
    print(image.shape)

    # resize image to size as mentioned in nvidia paper
    resized = tf.image.resize_images(image, (66, 200))
    #normalize
    resized = resized/255.0 - 0.5

    return resized



######################################################################################
######################################################################################

def get_image(data_dir, center, left, right, steering_angle):
    """
    will randomly choose and load image from one of three cameras and provide adjusted steering angle
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


######################################################################################
#   flip image from left to right and change steering angle accordingly
######################################################################################
def flip(image, steering_angle):
   
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


######################################################################################
#   translate image and change steering angle according the distortion
######################################################################################
def translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle



######################################################################################
#   Augumented image and adjust steering angle accordingly. Steering angle is associated with the image from central camera
######################################################################################
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = get_image(data_dir, center, left, right, steering_angle) 
    image, steering_angle = flip(image, steering_angle)
    image, steering_angle = translate(image, steering_angle, range_x, range_y)
    

    return image, steering_angle

######################################################################################
#    Generate training image give image paths and associated steering angles
######################################################################################
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

    #print("generating")
    while True:
        i = 0
        print(image_paths.shape[0])
        for index in np.random.permutation(image_paths.shape[0]):

            # randomly choose time point from dataset and get associated camera views and angle
            center, left, right = image_paths[index]        
            steering_angle = steering_angles[index]


            # add some image augumentation
            if is_training and np.random.rand() < 0.5:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 

            # convert choosen image to yuv representatiom
            images[i] = rgb2yuv(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
