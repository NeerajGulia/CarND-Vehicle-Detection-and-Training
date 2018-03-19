import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
from random import random, shuffle
from skimage.feature import hog
from scipy.ndimage.measurements import label

def change_color(image, color_space):
    """apply color conversion if other than 'RGB'"""
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(image)       
    return feature_image

def data_look(car_list, notcar_list):
    """gets the information about the cars and not cars data"""
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    image = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = image.dtype
    # Return data_dict
    return data_dict

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """Get the hog features"""
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    """Get channel spatial"""
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    """ Computes and returns thhe color histogram of channels"""
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
        
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """ Extract the features vector from the given image"""
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        # Read in each one by one
        feature_image = change_color(image, color_space) 

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features    

# Define a function to draw bounding boxes
def draw_boxes(img, box_set, color=(0, 0, 255), thick=6):
    """Draw the boxes on the given image"""
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bboxes in box_set:
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    """ Add += 1 for all pixes inside each box"""
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes 

def apply_threshold(heatmap, threshold):
    """ Zero out pixels below the threshold"""
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, colors):
    """Draw the boxes from the given labels on the given image"""
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], colors[car_number - 1], 6)
    # Return the image
    return img

def get_images(files):
    """Get images for the given files list, it returns image and flipped version of same image as well"""
    images = []
    for file in files:
        img = mpimg.imread(file)
        img1 = np.copy(img)
        img1 = cv2.flip(img1, 1)
        images.append(img)
        images.append(img1)
    return images


def get_labels(image, boxes_set, threshold):
    """ Get the labels after applying heat and threshold"""
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    for boxes in boxes_set:
        heat = add_heat(heat, boxes)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying    
    heatmap = (np.clip(heat, 0, 255))
    return label(heatmap), heatmap
    
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, 
              cell_per_block, spatial_size, hist_bins, color_space, show_all = False,
             spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel=0):
    """Find the boxes around the car which is detected in this method"""
    draw_img = np.copy(img)
    # these imagea are in jpg, while traning is done on PNG. Scale of PNG is from 0 to 1, 
    # while for jpg it is from 0 to 255
    img = img.astype(np.float32)/255 
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = change_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    if hog_feat:
        if hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        else:
            ch1 = ctrans_tosearch[:,:,hog_channel]
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            file_features = []
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                file_features.append(spatial_features)
                
            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
                file_features.append(hist_features)
            
            # Extract HOG for this patch
            if hog_feat:
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1))
                file_features.append(hog_features)
            features = np.concatenate(file_features).reshape(1, -1)
            test_features = X_scaler.transform(features)
            # Scale features and make a prediction
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all:
#                 decision = svc.decision_function(test_features)
#                 print('decision: ', decision)
#                 if decision > .4:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                boxes.append(box)
    return boxes


class Average:
    """store list by adding to item and return Average"""
    def __init__(self, period):
        self.total = 0
        self.queue = []
        self.period = period
    
    def append(self, item):
        """add item to the list"""
        #print("append item: ", item)
        self.queue.append(item)
        #print("append total new ", self.total)
        if len(self.queue) > self.period:
            self.queue.pop(0)
        else:
            self.total += item
        
    def getAverage(self):
        """get the mean of the list"""
        length = len(self.queue)
        if length == 0:
            length = 1
        #print("get avg, total: ", self.total, ", queue length: ", len(self.queue), "mean: ", self.total/length)
        return self.total/length
    
    def clear(self):
        """clear the average"""
        self.total = 0
        self.queue.clear()

class Store:
    """store list by adding to item and return Average"""
    def __init__(self, period):
        self.queue = []
        self.period = period
    
    def append(self, item):
        """add item to the list"""
        self.queue.append(item)
        if len(self.queue) > self.period:
            self.queue.pop(0)
    
    def getAll(self):
        return self.queue
    
    def getLength(self):
        """get the mean of the list"""
        return len(self.queue)
    
    def clear(self):
        """clear the average"""
        self.queue.clear()
                
        
def show_images(images, titles = None):
    """Method to plot the images array. The images is the array of images array"""
    images_count = len(images) * len(images[0])
    index = 1;
    cols = len(images)
    if images_count < cols:
        cols = images_count
    rows = images_count//cols
    if rows == 0:
        rows = 1
    if rows * cols < images_count:
        rows += 1
    print('images_count: {}, rows: {}, cols: {}'.format(images_count, rows, cols))
    figsize = (8 * cols, 5 * rows)
    fontsize = cols * 10
    fig = plt.figure(figsize=figsize)
    
    for image_index in range(len(images[0])):
        for array_index in range(len(images)):
            ax = fig.add_subplot(rows, cols, index)
            if titles is not None:
                ax.set_title(titles[array_index], fontsize = fontsize)
            plt.imshow(images[array_index][image_index])
            index += 1
    plt.tight_layout()
    plt.show()


def random_color():
    rgb=[255,0,0]
    shuffle(rgb)
    return tuple(rgb)

def inset_image(resize_image, to_image, resize_factor = .2, left_offset = 20, convert2RGB = False, text = None):
    """insert the given resize_image to the to_image, puts image inplace"""
    if convert2RGB:
        resize_image = np.array(cv2.merge((resize_image, resize_image, resize_image)),np.uint8)
    resize = cv2.resize(resize_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    to_image[20: 20 + resize.shape[0], left_offset: left_offset + resize.shape[1]] = resize
    if text:
        cv2.putText(to_image, text, (left_offset, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
    return to_image    
