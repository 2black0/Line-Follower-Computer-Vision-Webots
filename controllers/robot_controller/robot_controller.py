import cv2
import math
import numpy as np
from controller import Robot

class LineFollower:
    def __init__(self, timestep=64):
        self.Robot = Robot()
        self.TimeStep = timestep
        self.Camera = self.Robot.getDevice('camera')
        self.Camera.enable(self.TimeStep)
        self.CameraWidth = self.Camera.getWidth()
        self.CameraHeight = self.Camera.getHeight()
        
        # Initialize motors
        self.LeftMotor = self.Robot.getDevice('left wheel motor')
        self.RightMotor = self.Robot.getDevice('right wheel motor')
        self.LeftMotor.setPosition(float('inf'))
        self.RightMotor.setPosition(float('inf'))
        self.LeftMotor.setVelocity(0.0)
        self.RightMotor.setVelocity(0.0)
        
        # Motor parameters
        self.MaxVelocity = 6.28  # Maximum velocity for e-puck motors
        
        # PID control parameters (fine-tuned)
        self.KpFollow = 0.02  # Proportional gain
        self.KiFollow = 0.025  # Integral gain
        self.KdFollow = 0.01  # Derivative gain
        self.IntegralFollow = 0
        self.PreviousErrorFollow = 0
        
        # PID control parameters (fine-tuned)
        self.KpSpeed = 0.2  # Proportional gain
        self.KdSpeed = 0.02  # Derivative gain
        self.PreviousErrorSpeed = 0

    def ReadCamera(self):
        CameraImage = self.Camera.getImage()
        CameraImage = np.frombuffer(CameraImage, np.uint8).reshape((self.CameraHeight, self.CameraWidth, 4))
        CameraImage = cv2.cvtColor(CameraImage, cv2.COLOR_BGRA2BGR)
        
        return CameraImage

    def GetReference(self, CameraImage, VerticalStart, VerticalEnd, HorizontalStart, HorizontalEnd):
        # Process the ROI
        Roi = CameraImage[HorizontalStart:HorizontalEnd, VerticalStart:VerticalEnd]
        RoiGray = cv2.cvtColor(Roi, cv2.COLOR_BGR2GRAY)
        _, RoiThreshold = cv2.threshold(RoiGray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate the moments of the binary image
        RoiMoments = cv2.moments(RoiThreshold)
        RoiDetected = False
        if RoiMoments['m00'] != 0:
            # Calculate the centroid of the black line in the ROI
            RoiCx = int(RoiMoments['m10'] / RoiMoments['m00'])
            RoiCy = int(RoiMoments['m01'] / RoiMoments['m00'])

            # Draw the centroid on the ROI
            cv2.circle(Roi, (RoiCx, RoiCy), 5, (0, 255, 0), -1)
            RoiDetected = True
            
            CameraImage[HorizontalStart:HorizontalEnd, VerticalStart:VerticalEnd] = Roi
            
            return CameraImage, RoiDetected, RoiCx, RoiCy
        return CameraImage, RoiDetected, 0, 0

    def GetErrorFollow(self, CameraImage, RoiCxFollow):
        cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        ErrorFollow = (RoiCxFollow - (55 // 2))
        return ErrorFollow

    def GetAngleSpeed(self, CameraImage, RoiDetectedSpeed, RoiDetectedFollow, VerticalStartSpeed, VerticalStartFollow, RoiCxSpeed, RoiCxFollow, HorizontalStartSpeed, HorizontalStartFollow, RoiCySpeed, RoiCyFollow):
        AngleSpeed = 85
        if RoiDetectedSpeed and RoiDetectedFollow:
            CentroidFollow = (VerticalStartFollow + RoiCxFollow, HorizontalStartFollow + RoiCySpeed)
            CentroidSpeed = (VerticalStartSpeed + RoiCxSpeed, HorizontalStartSpeed + RoiCyFollow)
            cv2.line(CameraImage, CentroidFollow, CentroidSpeed, (255, 255, 0), 2)

            # Calculate the angle between the line and the vertical
            DeltaX = CentroidSpeed[0] - CentroidFollow[0]
            DeltaY = CentroidSpeed[1] - CentroidFollow[1]
            AngleSpeed = math.degrees(math.atan2(DeltaY, DeltaX))

            # Adjust angle to be relative to the vertical line
            if AngleSpeed < -90:
                AngleSpeed += 90
            elif AngleSpeed > -90:
                AngleSpeed -= -90
            AngleSpeed = abs(AngleSpeed)
            
            if AngleSpeed < 15 and AngleSpeed > -15:
                AngleSpeed = 0
            else:
                AngleSpeed = AngleSpeed
        return AngleSpeed

    def ShowCamera(self, CameraImage):
        ImageResize = cv2.resize(CameraImage, (320, 240), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", ImageResize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def CalculateBaseSpeed(self, AngleSpeed):
        DerivativeSpeed = AngleSpeed - self.PreviousErrorSpeed
        BaseSpeed = 6.0
        # PID control equation
        BaseSpeed = BaseSpeed - (self.KpSpeed * AngleSpeed) - (self.KdSpeed * DerivativeSpeed)
        # Update previous error
        self.PreviousErrorSpeed = AngleSpeed
        return BaseSpeed

    def CalculateSteeringFollow(self, ErrorFollow):
        # Calculate integral and derivative terms
        self.IntegralFollow += ErrorFollow
        DerivativeFollow = ErrorFollow - self.PreviousErrorFollow
        # PID control equation
        SteeringFollow = (self.KpFollow * ErrorFollow) + (self.KiFollow * self.IntegralFollow) + (self.KdFollow * DerivativeFollow)
        # Update previous error
        self.PreviousErrorFollow = ErrorFollow
        
        SteeringFollow = max(min(SteeringFollow, 1.0), -1.0)            
        return SteeringFollow
  
    def MotorAction(self, BaseSpeed, SteeringFollow):
        if BaseSpeed < 2.0 and BaseSpeed >= 0:
            BaseSpeed = 2.0
        elif BaseSpeed > -2.0 and BaseSpeed < 0:
            BaseSpeed = -2.0
        else:
            BaseSpeed = BaseSpeed

        # Adjust the speed of the motors based on the steering adjustment
        #LeftSpeed = BaseSpeed + SteeringFollow
        #RightSpeed = BaseSpeed - SteeringFollow

        # Cap the speeds to the max velocity
        LeftSpeed = max(min(BaseSpeed + SteeringFollow, self.MaxVelocity), -self.MaxVelocity)
        RightSpeed = max(min(BaseSpeed - SteeringFollow, self.MaxVelocity), -self.MaxVelocity)

        # Set the motor speeds
        self.LeftMotor.setVelocity(LeftSpeed)
        self.RightMotor.setVelocity(RightSpeed)
        
        print(f"Base Speed: {BaseSpeed:+.2f}, Steering: {SteeringFollow:+.2f}, Left Speed: {LeftSpeed:+.2f}, Right Speed: {RightSpeed:+.2f}")
    
    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)    
    
    def run(self):
        while self.Robot.step(self.TimeStep) != -1:
            CameraImage = self.ReadCamera()
            CameraImage, RoiDetectedSpeed, RoiCxSpeed, RoiCySpeed = self.GetReference(CameraImage, 0, 320, 40, 60)
            CameraImage, RoiDetectedFollow, RoiCxFollow, RoiCyFollow = self.GetReference(CameraImage, 132, 185, 195, 215) #VerticalStart, VerticalEnd, HorizontalStart, HorizontalEnd

            if RoiDetectedFollow: 
                ErrorFollow = self.GetErrorFollow(CameraImage, RoiCxFollow)
            AngleSpeed = self.GetAngleSpeed(CameraImage, RoiDetectedSpeed, RoiDetectedFollow, 0, 130, RoiCxSpeed, RoiCxFollow, 40, 185, RoiCySpeed, RoiCyFollow)

            BaseSpeed = self.CalculateBaseSpeed(AngleSpeed)
            #BaseSpeed = 2.0
            SteeringFollow = self.CalculateSteeringFollow(ErrorFollow)

            #print(AngleSpeed, ErrorFollow, BaseSpeed, SteeringFollow)
            
            self.MotorAction(BaseSpeed, SteeringFollow)
            self.ShowCamera(CameraImage)


if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()