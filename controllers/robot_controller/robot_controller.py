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
        self.KiFollow = 0.015
        self.KdFollow = 0.009
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

    def GetReference(self, CameraImage, SensorRow, SensorWidth, draw=False):
        VerticalStart = int(self.CameraWidth * ((1-SensorWidth)/2))
        VerticalEnd = int(self.CameraWidth - self.CameraWidth * ((1-SensorWidth)/2))
        HorizontalStart = self.RowHeight * SensorRow
        HorizontalEnd = HorizontalStart + self.RowHeight
                
        Roi = CameraImage[int(HorizontalStart):int(HorizontalEnd), int(VerticalStart):int(VerticalEnd)]
        RoiGray = cv2.cvtColor(Roi, cv2.COLOR_BGR2GRAY)
        _, RoiThreshold = cv2.threshold(RoiGray, 50, 255, cv2.THRESH_BINARY_INV)
        
        RoiMoments = cv2.moments(RoiThreshold)
        RoiDetected = False
        if RoiMoments['m00'] != 0:
            RoiCx = int(RoiMoments['m10'] / RoiMoments['m00'])
            RoiCy = int(RoiMoments['m01'] / RoiMoments['m00'])
            RoiDetected = True
            CameraImage[HorizontalStart:HorizontalEnd, VerticalStart:VerticalEnd] = Roi
            if draw == True:
                cv2.circle(Roi, (RoiCx, RoiCy), 3, (0, 255, 0), -1)
        else:
            RoiCx = 0
            RoiCy = 0
            
        #print(RoiDetected, SensorRow, RoiCx, RoiCy)
        return CameraImage, RoiDetected, RoiCx, RoiCy

    def GetReference2(self, CameraImage, Row, draw=False):
        #VerticalStart = 0
        #VerticalEnd = self.CameraWidth
        #HorizontalStart = self.RowHeight * Row
        #HorizontalEnd = HorizontalStart + self.RowHeight
        
        #Roi = CameraImage[int(HorizontalStart):int(HorizontalEnd), int(VerticalStart):int(VerticalEnd)]
        RoiGray = cv2.cvtColor(CameraImage, cv2.COLOR_BGR2GRAY)
        RoiBlurred = cv2.GaussianBlur(RoiGray, (5, 5), 0)
        _, RoiThreshold = cv2.threshold(RoiBlurred, 50, 255, cv2.THRESH_BINARY_INV)
        RoiEdges = cv2.Canny(RoiThreshold, 50, 150)
        RoiContours, _ = cv2.findContours(RoiEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        RoiDetected = False
        
        if RoiContours:
            RoiC = max(RoiContours, key=cv2.contourArea)
            RoiMoments = cv2.moments(RoiC)
            if RoiMoments["m00"] != 0:
                RoiCx = int(RoiMoments["m10"] / RoiMoments["m00"])
                RoiCy = int(RoiMoments["m01"] / RoiMoments["m00"])
                if draw == True:
                    # Gambar kontur dan centroid pada frame asli
                    cv2.drawContours(CameraImage, [RoiC], -1, (0, 255, 0), 2)
                    cv2.circle(CameraImage, (RoiCx, RoiCy), 7, (255, 0, 0), -1)
                RoiDetected = True
            else:
                RoiCx, RoiCy = 0, 0
            
            #print(RoiDetected, Row, RoiCx, RoiCy)
            return CameraImage, RoiDetected, RoiCx, RoiCy
    

    def GetErrorFollow(self, CameraImage, RoiCxFollow, SensorFollowWidth):
        cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        #ErrorFollow = (RoiCxFollow - (55 // 2))
        VerticalStart = int(self.CameraWidth * ((1-SensorFollowWidth)/2))
        VerticalEnd = int(self.CameraWidth - self.CameraWidth * ((1-SensorFollowWidth)/2))
        ErrorFollow = (RoiCxFollow - 56)
        return ErrorFollow

    def GetAngleSpeed(self, CameraImage, 
                      RoiDetectedSpeed, RoiDetectedFollow, 
                      SensorSpeedWidth, SensorFollowWidth,
                      RoiCxSpeed, RoiCxFollow, 
                      SensorSpeedRow, SensorFollowRow, 
                      RoiCySpeed, RoiCyFollow, 
                      draw=False):
        AngleSpeed = 45
        VerticalStartFollow = int(self.CameraWidth * ((1-SensorSpeedWidth)/2))
        VerticalStartSpeed = int(self.CameraWidth * ((1-SensorFollowWidth)/2))
        HorizontalStartFollow = self.RowHeight * SensorFollowRow
        HorizontalStartSpeed = self.RowHeight * SensorSpeedRow
        if RoiDetectedSpeed and RoiDetectedFollow:
            CentroidFollow = (VerticalStartFollow + RoiCxFollow, HorizontalStartFollow + RoiCySpeed)
            CentroidSpeed = (VerticalStartSpeed + RoiCxSpeed, HorizontalStartSpeed + RoiCyFollow)
            if draw:
                cv2.line(CameraImage, CentroidFollow, CentroidSpeed, (255, 255, 0), 2)

            DeltaX = CentroidSpeed[0] - CentroidFollow[0]
            DeltaY = CentroidSpeed[1] - CentroidFollow[1]
            AngleSpeed = math.degrees(math.atan2(DeltaY, DeltaX))

            if AngleSpeed < -90:
                AngleSpeed += 90
            elif AngleSpeed > -90:
                AngleSpeed -= -90
            AngleSpeed = abs(AngleSpeed)
            
            if AngleSpeed < 10 and AngleSpeed > -10:
                AngleSpeed = 0
            else:
                AngleSpeed = AngleSpeed
        return AngleSpeed

    def GetAngleSpeed2(self, CameraImage, RoiDetectedSpeed, RoiDetectedFollow, RoiCxSpeed, RoiCxFollow, RoiCySpeed, RoiCyFollow, draw=False):
        AngleSpeed = 60
        if RoiDetectedSpeed and RoiDetectedFollow:
            '''FirstPoint = (RoiCxSpeed, RoiCySpeed)
            SecondPoint = (RoiCxFollow, RoiCyFollow)
            ThirdPoint = (RoiCxFollow, 0)
            
            FirstPoint = np.array(FirstPoint)
            SecondPoint = np.array(SecondPoint)
            ThirdPoint = np.array(ThirdPoint)
            
            FirstVector = FirstPoint - SecondPoint
            SecondVector = ThirdPoint - SecondPoint
            
            DotProduct = np.dot(FirstVector, SecondVector)
            FirstMagnitude = np.linalg.norm(FirstVector)
            SecondMagnitude = np.linalg.norm(SecondVector)
            
            CosAngle = DotProduct / (FirstMagnitude * SecondMagnitude)
            
            AngleSpeed = np.arccos(CosAngle)
            AngleSpeed = np.degrees(AngleSpeed)'''
            
            if draw:
                cv2.circle(CameraImage, (RoiCxSpeed, RoiCySpeed), 5, (0, 255, 0), -1)
                cv2.circle(CameraImage, (RoiCxFollow, RoiCyFollow), 5, (0, 255, 0), -1)
                cv2.circle(CameraImage, (RoiCxFollow, 10), 5, (0, 255, 0), -1)
                #cv2.line(CameraImage, FirstPoint, SecondPoint, (0, 255, 255), 2)
                
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
        
        #print(f"Base Speed: {BaseSpeed:+.2f}, Steering: {SteeringFollow:+.2f}, Left Speed: {LeftSpeed:+.2f}, Right Speed: {RightSpeed:+.2f}")
    
    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)    
    
    def run(self):
        SensorSpeedRow = 2
        SensorSpeedWidth = 0.8
        SensorFollowRow = 9
        SensorFollowWidth = 0.3
        while self.Robot.step(self.TimeStep) != -1:
            CameraImage = self.ReadCamera()
            CameraImage, RoiDetectedSpeed, RoiCxSpeed, RoiCySpeed = self.GetReference(CameraImage, SensorSpeedRow, SensorSpeedWidth, draw=True)
            CameraImage, RoiDetectedFollow, RoiCxFollow, RoiCyFollow = self.GetReference(CameraImage, SensorFollowRow, SensorFollowWidth, draw=True)

            if RoiDetectedFollow: 
                ErrorFollow = self.GetErrorFollow(CameraImage, RoiCxFollow, SensorFollowWidth)
                print(RoiDetectedFollow, RoiCxFollow, RoiCyFollow, ErrorFollow)
            #AngleSpeed = self.GetAngleSpeed(CameraImage, RoiDetectedSpeed, RoiDetectedFollow, SensorSpeedWidth, SensorFollowWidth, RoiCxSpeed, RoiCxFollow, SensorSpeedRow, SensorFollowRow, RoiCySpeed, RoiCyFollow, draw=True)

            #BaseSpeed = self.CalculateBaseSpeed(AngleSpeed)
            BaseSpeed = 2.0
            SteeringFollow = self.CalculateSteeringFollow(ErrorFollow)

            #print(AngleSpeed, ErrorFollow)#, BaseSpeed, SteeringFollow)
            
            self.MotorAction(BaseSpeed, SteeringFollow)
            self.ShowCamera(CameraImage)


if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()