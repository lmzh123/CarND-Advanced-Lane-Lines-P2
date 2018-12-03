# CarND-Advanced-Lane-Lines-P2
Second project of the Udacity's Self Driving Cars Nano Degree

# **Finding Lane Lines on the Road** 

## Luis Miguel Zapata

---

**Finding Lane Lines on the Road**

This projects aims to develop a Computer Vision pipeline in order to the detect the lane lines seen by a camera located in a car's bumper.  

[image1]: ./output_images/distorted.jpg "Distorted"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/original.jpg "Original"
[image4]: ./output_images/undist.jpg "Undist"
[image5]: ./output_images/thresholded.png "Thresholded"
[image6]: ./output_images/warped.png "Warped"
[image7]: ./output_images/sliding_window.png "Sliding Window"
[image8]: ./output_images/search_polygon.png "Search Polygon"
[image9]: ./output_images/unwarped_img_undist.jpg "Unwarped"
[image10]: ./output_images/undist_lines.jpg "Lane"
[image11]: ./output_images/undist_lines_rads.jpg "Results"


### 1. Camera Calibration.

In order to correct the distortion from the images the calibration matrix as well as the distortion coefficients are calculated using a set chessboard images. Firs step is to find the chessboard corners of each image and stack these points for furter computations.

```
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```
Once all the corners are located and stored in a list called `imgpoints` the calibration matrix and the distortion coefficients are calculated using the following OpenCV built in function where `objpoints` is just an enumeration for further corners correspondence between images. 

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```
Using these found parameters any image can be corrected as the images below.

Original image             |  Corrected image 
:-------------------------:|:-------------------------:
![][image1]                |  ![][image2]

### 2. Distortion correction.

Using the found calibration parameters every incoming image from the camera is corrected using the following function.

```
undist = cv2.undistort(img, mtx, dist, None, mtx) # Undistore the image
```
This procedure will ensure that the calculations performed will correspond to real world measurements.


Original image             |  Corrected image 
:-------------------------:|:-------------------------:
![][image3]                |  ![][image4]

### 3. Thresholding.
Next step is to obtain the edges of the image. For this the approached that suited the best was two combine the S channel thresholding from HLS color space along with the magnitud of the gradient of RGB images. 

```
# S channel thresholding
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
```
```
# Gradient thresholding
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
abs_sobelx = np.absolute(sobelx) 
abs_sobely = np.absolute(sobely)
# Threshold by magnitud
mag = np.sqrt(np.square(sobelx)+np.square(sobely))
scaled_mag = np.uint8(255*mag/np.max(mag))
mag_binary = np.zeros_like(scaled_mag)
mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
```

![alt text][image5]

### 4. Bird-eye view.

The idea now is to warp the binary image to transform the projected lines on the road as if they were seen from the top. For this a set of source and destination points are defined.

Source points:

| Point | X        | Y        |
| ----- |:--------:| --------:|
| 1     | (w/2)-75 | (h/2)+100|
| 2     | (w/2)+75 | (h/2)+100|
| 3     | w-125    | h        |
| 4     | 125      | h        |

Destination points:

| Point | X        | Y        |
| ----- |:--------:| --------:|
| 1     | 100      | 50       |
| 2     | w-100    | 50       |
| 3     | w-100    | h-50     |
| 4     | 100      | h-50     |

```
 # Given src and dst points, calculate the perspective transform matrix
 M = cv2.getPerspectiveTransform(src, dst)
 # Warp the image using OpenCV warpPerspective()
 warped = cv2.warpPerspective(img, M, (w,h))
 ```

Binary                     |  Binary warped
:-------------------------:|:-------------------------:
![][image5]                |  ![][image6]

### 5. Lane pixels and curve fitting

There are two different approaches in order to find the pixels corresponding to each line. The first one is a sliding window approach where rectangular regions are used to bound the search region for white pixels. As a starting point the binary warped image is split into two from the middle hoping to separete each line. The pixels along every column are summed up for both the left and right side and the column with the maximum values while corresponde to the beggining of each line.

```
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
The image will be splitted vertically and the windows will bound the area to determine the pixels that correspond to each line, if the number of found pixels are above a certain threshold the search area will be moved corresponding to the center of mass of those pixels. After this the found pixels of each line will be fitted into a secod degree polynomial.

```
# Fit a second order polynomial to each using `np.polyfit`
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
![alt text][image7]

The second method to find the pixels corresponding to each line is to use the previously obtain polynomial to look around thos curves using a certain margin and hopefully the lines did not change abruptly.

![alt text][image8]

### 6. Radius of curvature and offset

First thing to determine the radius of curvature is to define how much a pixel corresponds to meters in this case 720 pixels correspond to 30 meters vertically and 700 pixels correspond to 3.7 meter horizontally.

```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
The calucalations to determine the radius of curvature will be performed on the bottom of the image. 

```
# Define y-value where we want radius of curvature
# We'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

Furthermore the difference between the center of the two lines measured again on the bottom of the image and the image's center will determine how far is the car from the center of the lane.

```
left_bottomx = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
right_bottomx = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
curv_center = (left_bottomx + right_bottomx)/2
offset = (curv_center - img_size[0]/2)*xm_per_pix
```

### 7. Results

Finally the resulting curves can be unwarped and plotted into the original undistorted image.

![][image9]

For a better representation the curves that bound the lane and the radius of curvature and offset are also displayed in the image.

![][image11]

### 8. Potential shortcomings

This algorithm relies highly in a good segmentation of the lane lines and even though the Saturation channel from HLS color space is more robust to light changing conditions, it is not certain that good lines are going to be obtained and that will not be affected by shades or other facts.

### 9. Possible improvements

In my opinion a better segmentation of the lane lines has to be done and possibly Convolutional Neural Networks for this task could be used. 
