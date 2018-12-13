#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 2
theta = 1
threshold = 15
min_line_len = 20
max_line_gap = 40
columns = 6
#fig=plt.figure(figsize=(30, 50))
ignore_mask_color = 255   

for i, img_name in enumerate(os.listdir("test_images/"), 1):
    img = mpimg.imread("test_images/" + img_name)
    img_copy = np.copy(img)
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]),(480, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    
    gray = grayscale(img) 
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    edges = region_of_interest(edges, vertices)
    
    line_img = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print (type((lines[])))
    left_segments = [[x1,y1,x2,y2] for line in lines for x1,y1,x2,y2 in line if (y2-y1)/(x2-x1) < 0]
    right_segments = [[x1,y1,x2,y2] for line in lines for x1,y1,x2,y2 in line if (y2-y1)/(x2-x1) > 0]
    left_points = []
    right_points = []
    for x1,y1,x2,y2 in left_segments:
        left_points.append((x1,y1))
        left_points.append((x2,y2))
    left_points.sort(key = lambda point: point[1])
    for x1,y1,x2,y2 in right_segments:
        right_points.append((x1,y1))
        right_points.append((x2,y2))
    right_points.sort(key = lambda point: point[1])
    left_slope = np.mean([(y2-y1)/(x2-x1) for [x1,y1,x2,y2] in left_segments])
    right_slope = np.mean([(y2-y1)/(x2-x1) for [x1,y1,x2,y2] in right_segments])
    bottom_y = imshape[0]
    #left_b = int(left_points[0][1] - left_slope*left_points[0][0])
    #bottom_x = int((bottom_y - left_b) / left_slope)
    bottom_x_left = 170
    bottom_x_right = 870
    #left_line = [(bottom_x, bottom_y, left_points[0][0], left_points[0][1])]
    #print ((bottom_x_left, bottom_y, left_points[0][0], left_points[0][1]))
    left_line = np.array([[[bottom_x_left, bottom_y, left_points[0][0], left_points[0][1]]]])
    right_line = np.array([[[bottom_x_right, bottom_y, right_points[0][0], right_points[0][1]]]])
    line_image = np.copy(img)*0
    draw_lines(line_image, left_line, thickness = 15)
    draw_lines(line_image, right_line, thickness = 15)
        
    #final_img = weighted_img(line_img, img, α=0.8, β=1., γ=0.)
    final_img = weighted_img(img, line_image, α=0.8, β=1., γ=0.)
    #fig.add_subplot(6, 2, i*2 - 1)
    #plt.imshow(final_img)
    #fig.add_subplot(6, 2, i*2)
    #plt.imshow(line_img)
    
    out_path = "test_images_output/"
    file_name, file_ext = img_name.split('.')
    out_file_name = file_name + '_out.' + file_ext
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, out_file_name)
    plt.imsave(save_path, final_img)
    
#plt.show()

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    #img = mpimg.imread(image)
    img = np.copy(image)
    img_copy = np.copy(img)
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]),(480, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    
    gray = grayscale(img) 
    blur_gray = gaussian_blur(gray, kernel_size)
    edges = canny(blur_gray, low_threshold, high_threshold)
    edges = region_of_interest(edges, vertices)
    
    line_img = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print (type((lines[])))
    left_segments = [[x1,y1,x2,y2] for line in lines for x1,y1,x2,y2 in line if (y2-y1)/(x2-x1) < 0]
    right_segments = [[x1,y1,x2,y2] for line in lines for x1,y1,x2,y2 in line if (y2-y1)/(x2-x1) > 0]
    left_points = []
    right_points = []
    for x1,y1,x2,y2 in left_segments:
        left_points.append((x1,y1))
        left_points.append((x2,y2))
    left_points.sort(key = lambda point: point[1])
    for x1,y1,x2,y2 in right_segments:
        right_points.append((x1,y1))
        right_points.append((x2,y2))
    right_points.sort(key = lambda point: point[1])
    left_slope = np.mean([(y2-y1)/(x2-x1) for [x1,y1,x2,y2] in left_segments])
    right_slope = np.mean([(y2-y1)/(x2-x1) for [x1,y1,x2,y2] in right_segments])
    bottom_y = imshape[0]
    #left_b = int(left_points[0][1] - left_slope*left_points[0][0])
    #bottom_x = int((bottom_y - left_b) / left_slope)
    bottom_x_left = 170
    bottom_x_right = 870
    #left_line = [(bottom_x, bottom_y, left_points[0][0], left_points[0][1])]
    #print ((bottom_x_left, bottom_y, left_points[0][0], left_points[0][1]))
    left_line = np.array([[[bottom_x_left, bottom_y, left_points[0][0], left_points[0][1]]]])
    right_line = np.array([[[bottom_x_right, bottom_y, right_points[0][0], right_points[0][1]]]])
    line_image = np.copy(img)*0
    draw_lines(line_image, left_line, thickness = 15)
    draw_lines(line_image, right_line, thickness = 15)
        
    #final_img = weighted_img(line_img, img, α=0.8, β=1., γ=0.)
    result = weighted_img(img, line_image, α=0.8, β=1., γ=0.)
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)