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
        
        self.RowTotal = 12
        self.RowHeight = int(self.CameraHeight / self.RowTotal)
        
        self.LeftMotor = self.Robot.getDevice('left wheel motor')
        self.RightMotor = self.Robot.getDevice('right wheel motor')
        self.LeftMotor.setPosition(float('inf'))
        self.RightMotor.setPosition(float('inf'))
        self.LeftMotor.setVelocity(0.0)
        self.RightMotor.setVelocity(0.0)
        
        self.MaxVelocity = 6.28
        
        self.KpFollow = 0.025
        self.KiFollow = 0.0006
        self.KdFollow = 0.006
        self.IntegralFollow = 0
        self.PreviousErrorFollow = 0
        
        self.KpSpeed = 0.2
        self.KdSpeed = 0.02
        self.PreviousErrorSpeed = 0

    def ReadCamera(self):
        CameraImage = self.Camera.getImage()
        CameraImage = np.frombuffer(CameraImage, np.uint8).reshape((self.CameraHeight, self.CameraWidth, 4))
        CameraImage = cv2.cvtColor(CameraImage, cv2.COLOR_BGRA2BGR)
        
        return CameraImage

    def GetReference(self, CameraImage, SensorRow, SensorWidth, drawDot=False, drawBox=False):
        VerticalStart = int(self.CameraWidth * ((1-SensorWidth)/2))
        VerticalEnd = int(self.CameraWidth - self.CameraWidth * ((1-SensorWidth)/2))
        HorizontalStart = self.RowHeight * SensorRow
        HorizontalEnd = HorizontalStart + self.RowHeight
        
        LeftTopPoint = (VerticalStart, HorizontalStart)
        RightBottomPoint = (VerticalEnd, HorizontalEnd)
            
        Roi = CameraImage[int(HorizontalStart):int(HorizontalEnd), int(VerticalStart):int(VerticalEnd)]
        RoiGray = cv2.cvtColor(Roi, cv2.COLOR_BGR2GRAY)
        _, RoiThreshold = cv2.threshold(RoiGray, 50, 255, cv2.THRESH_BINARY_INV)
        
        RoiMoments = cv2.moments(RoiThreshold)
        RoiDetected = False
        if RoiMoments['m00'] != 0:
            RoiCx = int(RoiMoments['m10'] / RoiMoments['m00'])
            RoiCy = int(RoiMoments['m01'] / RoiMoments['m00'])
            RoiDetected = True
            if drawDot:
                cv2.circle(Roi, (RoiCx, RoiCy), 3, (0, 0, 255), -1)
            if drawBox:
                cv2.rectangle(CameraImage, LeftTopPoint, RightBottomPoint, (255, 0, 0), 2)
        else:
            RoiCx = 0
            RoiCy = 0
        return CameraImage, RoiDetected, RoiCx, RoiCy    

    def GetErrorFollow(self, CameraImage, RoiCxFollow, SensorFollowWidth, drawLine=False):
        if drawLine:
            cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        ErrorFollow = (RoiCxFollow - (self.CameraWidth * SensorFollowWidth) // 2)
        return ErrorFollow

    def GetAngleSpeed(self, CameraImage, 
                      RoiDetectedSpeed, RoiDetectedFollow, 
                      SensorSpeedWidth, SensorFollowWidth,
                      RoiCxSpeed, RoiCxFollow, 
                      SensorSpeedRow, SensorFollowRow, 
                      RoiCySpeed, RoiCyFollow, 
                      drawDot=False, drawLine=False):
        AngleSpeed = 45        
        if RoiDetectedSpeed and RoiDetectedFollow:
            FirstPointCx = int((RoiCxFollow - (self.CameraWidth * SensorFollowWidth) // 2) + (self.CameraWidth / 2))
            FirstPointCy = int(self.RowHeight * SensorSpeedRow + RoiCySpeed)
            
            SecondPointCx = int((RoiCxFollow - (self.CameraWidth * SensorFollowWidth) // 2) + (self.CameraWidth / 2))
            SecondPointCy = int(self.RowHeight * SensorFollowRow + RoiCyFollow)
            
            ThirdPointCx = int((RoiCxSpeed - (self.CameraWidth * SensorSpeedWidth) // 2) + (self.CameraWidth / 2))
            ThirdPointCy = int(self.RowHeight * SensorSpeedRow + RoiCySpeed)
            
            PointA = np.array([FirstPointCx, FirstPointCy])
            PointB = np.array([SecondPointCx, SecondPointCy])
            PointC = np.array([ThirdPointCx, ThirdPointCy])
            
            PointBA = PointA - PointB
            PointBC = PointC - PointB
        
            CosineAngle = np.dot(PointBA, PointBC) / (np.linalg.norm(PointBA) * np.linalg.norm(PointBC))
            AngleRadian = np.arccos(CosineAngle)
            AngleSpeed = np.degrees(AngleRadian)
        
            if drawDot:
                cv2.circle(CameraImage, PointA, 3, (0, 0, 255), -1)
                cv2.circle(CameraImage, PointB, 3, (0, 0, 255), -1)
                cv2.circle(CameraImage, PointC, 3, (0, 0, 255), -1)
            if drawLine:
                cv2.line(CameraImage, PointA, PointB, (0, 255, 0), 2)
                cv2.line(CameraImage, PointB, PointC, (0, 255, 0), 2)

        return AngleSpeed

    def ShowCamera(self, CameraImage):
        ImageResize = cv2.resize(CameraImage, (320, 240), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", ImageResize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def CalculateBaseSpeed(self, AngleSpeed):
        DerivativeSpeed = AngleSpeed - self.PreviousErrorSpeed
        BaseSpeed = 6.0
        BaseSpeed = BaseSpeed - (self.KpSpeed * AngleSpeed) - (self.KdSpeed * DerivativeSpeed)
        self.PreviousErrorSpeed = AngleSpeed
        return BaseSpeed

    def CalculateSteeringFollow(self, ErrorFollow):
        self.IntegralFollow += ErrorFollow
        DerivativeFollow = ErrorFollow - self.PreviousErrorFollow
        SteeringFollow = (self.KpFollow * ErrorFollow) + (self.KiFollow * self.IntegralFollow) + (self.KdFollow * DerivativeFollow)
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

        LeftSpeed = max(min(BaseSpeed + SteeringFollow, self.MaxVelocity), -self.MaxVelocity)
        RightSpeed = max(min(BaseSpeed - SteeringFollow, self.MaxVelocity), -self.MaxVelocity)

        self.LeftMotor.setVelocity(LeftSpeed)
        self.RightMotor.setVelocity(RightSpeed)
    
    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)    
    
    def run(self):
        SensorSpeedRow = 2
        SensorSpeedWidth = 0.8
        SensorFollowRow = 10
        SensorFollowWidth = 0.3
        while self.Robot.step(self.TimeStep) != -1:
            CameraImage = self.ReadCamera()
            CameraImage, RoiDetectedSpeed, RoiCxSpeed, RoiCySpeed = self.GetReference(CameraImage, SensorSpeedRow, SensorSpeedWidth, drawDot=True, drawBox=True)
            CameraImage, RoiDetectedFollow, RoiCxFollow, RoiCyFollow = self.GetReference(CameraImage, SensorFollowRow, SensorFollowWidth, drawDot=True, drawBox=True)

            if RoiDetectedFollow: 
                ErrorFollow = self.GetErrorFollow(CameraImage, RoiCxFollow, SensorFollowWidth, drawLine=True)
            AngleSpeed = self.GetAngleSpeed(CameraImage, RoiDetectedSpeed, RoiDetectedFollow, SensorSpeedWidth, SensorFollowWidth, RoiCxSpeed, RoiCxFollow, SensorSpeedRow, SensorFollowRow, RoiCySpeed, RoiCyFollow, drawDot=True, drawLine=True)

            #BaseSpeed = self.CalculateBaseSpeed(AngleSpeed)
            BaseSpeed = 2.0
            SteeringFollow = self.CalculateSteeringFollow(ErrorFollow)

            #print(AngleSpeed, ErrorFollow BaseSpeed, SteeringFollow)
            self.MotorAction(BaseSpeed, SteeringFollow)
            print(f"Base Speed: {BaseSpeed:+.2f}, Steering: {SteeringFollow:+.2f}, Left Speed: {BaseSpeed+SteeringFollow:+.2f}, Right Speed: {BaseSpeed-SteeringFollow:+.2f}")
            
            self.ShowCamera(CameraImage)


if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()