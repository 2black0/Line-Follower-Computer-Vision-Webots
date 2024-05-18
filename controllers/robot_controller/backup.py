import cv2
import numpy as np
from controllers import Robot

class LineFollower:
    def __init__(self, timestep=64):
        self.Robot = Robot()
        self.TimeStep = timestep
        self.Camera = self.robot.getDevice('camera')
        self.Camera.enable(self.timestep)
        self.CameraWidth = self.camera.getWidth()
        self.CameraHeight = self.camera.getHeight()
        
        # Initialize motors
        self.LeftMotor = self.robot.getDevice('left wheel motor')
        self.RightMotor = self.robot.getDevice('right wheel motor')
        self.LeftMotor.setPosition(float('inf'))
        self.RightMotor.setPosition(float('inf'))
        self.LeftMotor.setVelocity(0.0)
        self.RightMotor.setVelocity(0.0)
        
        # Motor parameters
        self.MaxVelocity = 6.28  # Maximum velocity for e-puck motors
        
        # PID control parameters (fine-tuned)
        self.KpFollow = 0.06  # Proportional gain
        self.KiFollow = 0.015  # Integral gain
        self.KdFollow = 0.02  # Derivative gain
        self.IntegralFollow = 0
        self.PreviousErrorFollow = 0
        
        # PID control parameters (fine-tuned)
        self.KpSpeed = 0.05  # Proportional gain
        self.KdSpeed = 0.005  # Derivative gain
        self.PreviousErrorSpeed = 0


    def run(self):
        while self.robot.step(self.TimeStep) != -1:
            CameraImage = self.ReadCamera()
            self.ShowCamera(CameraImage)


    def ReadCamera(self):
        CameraImage = self.camera.getImage()
        CameraImage = np.frombuffer(CameraImage, np.uint8).reshape((self.height, self.width, 4))
        CameraImage = cv2.cvtColor(CameraImage, cv2.COLOR_BGRA2BGR)
        
        return CameraImage

    def ShowCamera(self, CameraImage):
        ImageResize = cv2.resize(CameraImage, (320, 240), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", ImageResize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()