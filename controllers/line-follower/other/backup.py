import cv2
import numpy as np
from controller import Robot

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
        self.kp = 0.05  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.05  # Derivative gain
        self.integral = 0
        self.previous_error = 0

    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.process_camera_image()

    def process_camera_image(self):
        image = self.camera.getImage()
        image = np.frombuffer(image, np.uint8).reshape((self.height, self.width, 4))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Define the region of interest (ROI) for line detection
        roi_width = 55  # Reduced width
        roi_height = 20  # Reduced height
        vertical_start = (self.width - roi_width) // 2
        vertical_end = vertical_start + roi_width
        horizontal_start = self.height - roi_height
        horizontal_end = self.height
        roi = image[horizontal_start:horizontal_end, vertical_start:vertical_end]

        # Draw a green rectangle around the ROI on the original image
        cv2.rectangle(image, (vertical_start, horizontal_start), (vertical_end, horizontal_end), (0, 255, 0), 2)

        # Process the ROI
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

        # Calculate the moments of the binary image
        moments = cv2.moments(thresholded_roi)
        if moments['m00'] != 0:
            # Calculate the centroid of the black line in the ROI
            cX = int(moments['m10'] / moments['m00'])
            cY = int(moments['m01'] / moments['m00'])

            # Draw the centroid on the ROI
            cv2.circle(roi, (cX, cY), 5, (255, 0, 0), -1)

            # Calculate the error: distance from the center of the ROI
            error = cX - (roi_width // 2)

            # Adjust the robot's steering based on the error
            self.adjust_robot_steering(error)

        # Draw a vertical line in the middle of the entire image
        cv2.line(image, (self.width // 2, 0), (self.width // 2, self.height), (0, 0, 255), 1)

        # Replace the ROI in the original image
        image[horizontal_start:horizontal_end, vertical_start:vertical_end] = roi

        # Display the image
        resized_image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", resized_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def adjust_robot_steering(self, error):
        # Use PID control to calculate the steering adjustment
        steering_adjustment = self.pid_control(error)
        
        # Base speed for the motors
        base_speed = 2.5

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