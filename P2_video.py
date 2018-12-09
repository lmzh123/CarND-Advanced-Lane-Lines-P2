import cv2
from line import Line
from img_proccesing import *

# Get the calibaration parameters
mtx, dist = get_calib_params()

# Define the video to process
file_name = "project_video.mp4"
cap = cv2.VideoCapture(file_name)
# Define the video to be written
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'x264')
writer = cv2.VideoWriter("test_videos_outputs/"+file_name,
                            fourcc, fps, (w,h))

left_line = Line()
right_line = Line()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Undistore the image
        undist = cv2.undistort(frame, mtx, dist, None, mtx)
        # Threshold the image
        binary = get_binary_image(undist)
        # Warp the image for eagle eye view
        binary_warped = warp_image(binary)
        
        if(not(left_line.detected) or not(right_line.detected)):
            # Determine the pixels belonging to each image using sliding window
            leftx, lefty, rightx, righty, lane_pixels = find_lane_pixels(
                                                            binary_warped)
            # Fit polynomial using those pixels
            left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
                                            (w,h), leftx, lefty, rightx, righty)
            left_line.detected = True
            right_line.detected = True
            left_line.current_fit = left_fit
            right_line.current_fit = right_fit
            left_line.allx = left_fitx
            left_line.ally = ploty
            right_line.allx = right_fitx
            right_line.ally = ploty
            left_line.stack_x_vals(left_fitx, left_fit)
            right_line.stack_x_vals(right_fitx, right_fit)

        else:
            # Fit new polynomials using previous polynomials
            left_fit, right_fit, left_fitx, right_fitx, ploty, result = search_around_poly(
                                                binary_warped, left_line.current_fit, 
                                                right_line.current_fit)
            # Get the difference between previous and new coefficients
            left_line.diffs = left_line.current_fit - left_fit
            right_line.diffs = right_line.current_fit - right_fit
            # Store the new information
            left_line.current_fit = left_fit
            right_line.current_fit = right_fit
            left_line.allx = left_fitx
            left_line.ally = ploty
            right_line.allx = right_fitx
            right_line.ally = ploty
            left_line.stack_x_vals(left_fitx, left_fit)
            right_line.stack_x_vals(right_fitx, right_fit)

        # Measure the radius of curvature
        left_curverad, right_curverad, offset = measure_curvature_real(
                                        left_line.current_fit, right_line.current_fit, 
                                        left_line.ally,(w,h))
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad
        left_line.line_base_pos = offset
        right_line.line_base_pos = offset
        # Draw the detected lanes
        undist_lines = draw_lines(binary_warped, undist, left_line.allx, 
                                right_line.allx, left_line.ally)
        # Display results in the image
        undist_lines_rad = display_curv(undist_lines, left_line.radius_of_curvature, 
                            right_line.radius_of_curvature, left_line.line_base_pos)
        writer.write(undist_lines_rad)

        # Display the resulting frame
        if 'result' in locals():
            cv2.imshow('warped',result)
        cv2.imshow('final',undist_lines_rad)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
