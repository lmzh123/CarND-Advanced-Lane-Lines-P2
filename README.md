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


### 2. Lines extrapolation

Using the slope of a line equation it can be determined which of the remaining lines correspond to both the left and right line using an if statement within the `draw_lines()` function. For instance if the slope of a line is positive it corresponds to the right line otherwise it belongs to the left line.

![](https://latex.codecogs.com/gif.latex?m%20%3D%20%5Cfrac%7By_%7B2%7D-y_%7B1%7D%7D%7Bx_%7B2%7D-x_%7B1%7D%7D)

Once the left and right lines are grouped the average of such lines are obtained using the function `np.average()`. From there the equations of these lines are calculated using the `np.polyfit()` function. Because of this first order equation it is possible to extrapolate this mean line to the same extents of the region of interest used before.

 Hough's lines             |  Mean extrapolated lines
:-------------------------:|:-------------------------:
![][image8]                |  ![][image9]

For matters of displaying the thickness of the line drawn is increased to 10.

### 3. Optional challenge

Trying to improve the performance of this pipeline with the challenge video a small change was made that even tought was not a complete solution to the problem, it improved the results obtained. It can be noticed that the car's bumper can be seen by the camera and that bigger barriers are located at the side of the road.

To tackle this problem the lines were filtered according to their slope, in this case the lines that where close to be completely horizontal were discarded and this corresponds to avoid lines whith slopes between -0.1 and 0.1.

```
if ((y2-y1)/(x2-x1)) > 0.1:
    right = np.vstack((right, np.array([x1, y1, x2, y2])))
elif ((y2-y1)/(x2-x1)) < -0.1:
    left = np.vstack((left, np.array([x1, y1, x2, y2])))
```
### 4. Potential shortcomings

This pipeline is highly tuned for this particular setup and as shown by the challenge video it could have problems with the following.

* Overfitted: This algorithm is not robust enough to handle well when there are other lines in our region of interes besides the lane lines.
* Rotations or traslations of the camera: It is highly dependant of getting the same kind of images everytime and any change in the region of interes could be a problem.
* Light conditions: Edge detection can be a problem depending of the amount of light reflected by the road and also by the shadows that could be seen in or region of interest.

### 5. Possible improvements

For improvements of this pipeline the following is proposed:

* After the canny edge detection a set of morphological operators could be applied with a kernel that looks for diagonal lines. This could help to discard any other kind of lines that we are not interested in having.
* After obtaining the gray scale image it could be helpfull to apply an histogram equalization to it so the sharpnes of the image is improved and it is more inmune to light changes.
* Further tuning can be done for clustering which lines correspond to the left and right lines. For instance an outlier detection algorithm such as RANSAC along with a clustering algorithm like k-nearest neighbor could lead to a better model and more robust when different lines are seen within our region of interest. 
