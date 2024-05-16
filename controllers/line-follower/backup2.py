import cv2
import numpy as np
from controller import Robot
import math

class EPuckRobot:
    def __init__(self, timestep=64):
        self.robot = Robot()
        self.timestep = timestep
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Motor parameters
        self.max_velocity = 6.28  # Maximum velocity for e-puck motors
        
        # PID control parameters (fine-tuned)
        self.kp = 0.04  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.01  # Derivative gain
        self.integral = 0
        self.previous_error = 0

    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.process_camera_image()

    def process_camera_image(self):
        image = self.camera.getImage()
        image = np.frombuffer(image, np.uint8).reshape((self.height, self.width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Draw a vertical line in the middle of the entire image
        cv2.line(image, (self.width // 2, 0), (self.width // 2, self.height), (0, 0, 255), 1)

        # Define the region of interest (ROI) for line detection
        second_roi_width = 320  # Reduced width
        second_roi_height = 20  # Reduced height
        
        second_vertical_start = 0
        second_vertical_end = second_roi_width
        second_horizontal_start = (self.height // 2) - 10
        second_horizontal_end = second_horizontal_start + second_roi_height
        second_roi = image[second_horizontal_start:second_horizontal_end, second_vertical_start:second_vertical_end]

        # Process the second ROI
        second_gray_roi = cv2.cvtColor(second_roi, cv2.COLOR_BGR2GRAY)
        _, second_thresholded_roi = cv2.threshold(second_gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

        # Calculate the moments of the binary image
        second_moments = cv2.moments(second_thresholded_roi)
        second_detected = False
        if second_moments['m00'] != 0:
            # Calculate the centroid of the black line in the ROI
            second_cX = int(second_moments['m10'] / second_moments['m00'])
            second_cY = int(second_moments['m01'] / second_moments['m00'])

            # Draw the centroid on the ROI
            cv2.circle(second_roi, (second_cX, second_cY), 5, (0, 255, 0), -1)
            second_detected = True

        # Replace the ROI in the original image
        image[second_horizontal_start:second_horizontal_end, second_vertical_start:second_vertical_end] = second_roi

        # Define the region of interest (ROI) for line detection
        first_roi_width = 55  # Reduced width
        first_roi_height = 20  # Reduced height
        first_vertical_start = (self.width - first_roi_width) // 2
        first_vertical_end = first_vertical_start + first_roi_width
        first_horizontal_start = self.height - first_roi_height
        first_horizontal_end = self.height
        first_roi = image[first_horizontal_start:first_horizontal_end, first_vertical_start:first_vertical_end]

        # Process the first ROI
        first_gray_roi = cv2.cvtColor(first_roi, cv2.COLOR_BGR2GRAY)
        _, first_thresholded_roi = cv2.threshold(first_gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

        # Calculate the moments of the binary image
        first_moments = cv2.moments(first_thresholded_roi)
        first_detected = False
        if first_moments['m00'] != 0:
            # Calculate the centroid of the black line in the ROI
            cX = int(first_moments['m10'] / first_moments['m00'])
            cY = int(first_moments['m01'] / first_moments['m00'])

            # Draw the centroid on the ROI
            cv2.circle(first_roi, (cX, cY), 5, (255, 0, 0), -1)
            first_detected = True

            # Calculate the error: distance from the center of the ROI
            error = cX - (first_roi_width // 2)

            # Adjust the robot's steering based on the error
            self.adjust_robot_steering(error)

        # Replace the ROI in the original image
        image[first_horizontal_start:first_horizontal_end, first_vertical_start:first_vertical_end] = first_roi

        # Draw line between centroids if both are detected
        if first_detected and second_detected:
            first_centroid = (first_vertical_start + cX, first_horizontal_start + cY)
            second_centroid = (second_vertical_start + second_cX, second_horizontal_start + second_cY)
            cv2.line(image, first_centroid, second_centroid, (255, 255, 0), 2)

            # Calculate the angle between the line and the vertical
            delta_x = second_centroid[0] - first_centroid[0]
            delta_y = second_centroid[1] - first_centroid[1]
            angle = math.degrees(math.atan2(delta_y, delta_x))

            # Adjust angle to be relative to the vertical line
            if angle < -90:
                angle += 90
            elif angle > -90:
                angle -= -90

            # Display the angle on the image
            angle_text = f"Angle: {angle:.2f} degrees"
            cv2.putText(image, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the image
        resized_image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", resized_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def adjust_robot_steering(self, error):
        # Use PID control to calculate the steering adjustment
        steering_adjustment = self.pid_control(error)
        
        # Base speed for the motors
        base_speed = 2.0

        # Adjust the speed of the motors based on the steering adjustment
        left_speed = base_speed + steering_adjustment
        right_speed = base_speed - steering_adjustment

        # Cap the speeds to the max velocity
        left_speed = max(min(left_speed, self.max_velocity), -self.max_velocity)
        right_speed = max(min(right_speed, self.max_velocity), -self.max_velocity)

        # Set the motor speeds
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        # Print the steering adjustment and motor speeds with two decimal places, aligning positive and negative values
        print(f"Steering Adjustment: {steering_adjustment:+.2f}, Left Speed: {left_speed:+.2f}, Right Speed: {right_speed:+.2f}")

    def pid_control(self, error):
        # Calculate integral and derivative terms
        self.integral += error
        derivative = error - self.previous_error

        # PID control equation
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Update previous error
        self.previous_error = error

        return output

    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)  # Optionally add this to quit the simulation

if __name__ == "__main__":
    epuck_robot = EPuckRobot()
    epuck_robot.run()
