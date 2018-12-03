import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #History of the n fitted coefficients
        self.fit_historical = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def stack_x_vals(self, x, fitted_coeff, n = 10):
        if len(self.recent_xfitted) >= n:
            self.recent_xfitted.append(x)
            self.recent_xfitted.pop(0)
            self.fit_historical.append(fitted_coeff)
            self.fit_historical.pop(0)
            # Get the average
            self.bestx = np.sum(self.recent_xfitted, axis = 0)
            self.best_fit = np.sum(self.fit_historical, axis = 0)
        else:
            self.recent_xfitted.append(x)
            self.fit_historical.append(fitted_coeff)



