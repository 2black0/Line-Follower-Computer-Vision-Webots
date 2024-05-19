import cv2
import os
import math
import csv
import numpy as np
import skfuzzy as fuzz
from controller import Robot
from skfuzzy import control as ctrl

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
        
        # PID Speed Low
        #self.KpFollow = 0.025
        #self.KiFollow = 0.0006
        #self.KdFollow = 0.006
        
        # PID Speed High
        self.KpFollow = 0.05
        self.KiFollow = 0.01
        self.KdFollow = 0.00001
        self.IntegralFollow = 0
        self.PreviousErrorFollow = 0
        
        self.KpSpeed = 0.01
        self.KdSpeed = 0.00
        self.PreviousErrorSpeed = 0
        
        self.Error = ctrl.Antecedent(np.arange(-200, 201, 1), 'Error')
        #self.DerivativeError = ctrl.Antecedent(np.arange(-30, 30.1, 0.1), 'DerivativeError')
        self.DeltaSpeed = ctrl.Consequent(np.arange(-5.0, 5.01, 0.01), 'DeltaSpeed')

        #self.Error.automf(3, names=['negative', 'zero', 'positive'])
        self.Error['negative'] = fuzz.trimf(self.Error.universe, [-200, -100, -10])
        self.Error['zero'] = fuzz.trimf(self.Error.universe, [-30, 0, 30])
        self.Error['positive'] = fuzz.trimf(self.Error.universe, [10, 100, 200])

        #self.DerivativeError.automf(3, names=['negative', 'zero', 'positive'])
        #self.DerivativeError['negative'] = fuzz.trimf(self.DerivativeError.universe, [-30, -15, 0])
        #self.DerivativeError['zero'] = fuzz.trimf(self.DerivativeError.universe, [-6, 0, 6])
        #self.DerivativeError['positive'] = fuzz.trimf(self.DerivativeError.universe, [0, 15, 30])
        
        self.DeltaSpeed.automf(3, names=['negative', 'zero', 'positive'])
             
        '''self.Rule1 = ctrl.Rule(self.Error['negative'] & self.DerivativeError['negative'], self.DeltaSpeed['negative'])
        self.Rule2 = ctrl.Rule(self.Error['negative'] & self.DerivativeError['zero'], self.DeltaSpeed['negative'])
        self.Rule3 = ctrl.Rule(self.Error['negative'] & self.DerivativeError['positive'], self.DeltaSpeed['zero'])
        self.Rule4 = ctrl.Rule(self.Error['zero'] & self.DerivativeError['negative'], self.DeltaSpeed['negative'])
        self.Rule5 = ctrl.Rule(self.Error['zero'] & self.DerivativeError['zero'], self.DeltaSpeed['zero'])
        self.Rule6 = ctrl.Rule(self.Error['zero'] & self.DerivativeError['positive'], self.DeltaSpeed['positive'])
        self.Rule7 = ctrl.Rule(self.Error['positive'] & self.DerivativeError['negative'], self.DeltaSpeed['zero'])
        self.Rule8 = ctrl.Rule(self.Error['positive'] & self.DerivativeError['zero'], self.DeltaSpeed['positive'])
        self.Rule9 = ctrl.Rule(self.Error['positive'] & self.DerivativeError['positive'], self.DeltaSpeed['positive'])'''
             
        self.Rule1 = ctrl.Rule(self.Error['negative'], self.DeltaSpeed['negative'])
        self.Rule2 = ctrl.Rule(self.Error['zero'], self.DeltaSpeed['zero'])
        self.Rule3 = ctrl.Rule(self.Error['positive'], self.DeltaSpeed['positive'])
             
        #self.DeltaSpeedControl = ctrl.ControlSystem([self.Rule1, self.Rule2, self.Rule3, self.Rule4, self.Rule5, self.Rule6, self.Rule7, self.Rule8, self.Rule9])
        self.DeltaSpeedControl = ctrl.ControlSystem([self.Rule1, self.Rule2, self.Rule3])
        self.DeltaSpeedSim = ctrl.ControlSystemSimulation(self.DeltaSpeedControl)

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
        AngleSpeed = 60        
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

    def CalculateBaseSpeedPID(self, AngleSpeed, MaxSpeed):
        DerivativeSpeed = AngleSpeed - self.PreviousErrorSpeed
        BaseSpeed = MaxSpeed - (self.KpSpeed * AngleSpeed) - (self.KdSpeed * DerivativeSpeed)
        self.PreviousErrorSpeed = AngleSpeed
        return BaseSpeed

    def CalculateDeltaSpeedPID(self, ErrorFollow):
        self.IntegralFollow += ErrorFollow
        DerivativeFollow = ErrorFollow - self.PreviousErrorFollow
        DeltaSpeed = (self.KpFollow * ErrorFollow) + (self.KiFollow * self.IntegralFollow) + (self.KdFollow * DerivativeFollow)
        self.PreviousErrorFollow = ErrorFollow
        return DerivativeFollow, DeltaSpeed
  
    def CalculateDeltaSpeedFuzzy(self, ErrorFollow):
        ErrorFollow = max(min(ErrorFollow, 200), -200)
        DerivativeFollow = ErrorFollow - self.PreviousErrorFollow
        DerivativeFollow = max(min(DerivativeFollow, 30), -30)
        self.DeltaSpeedSim.input['Error'] = ErrorFollow
        #self.DeltaSpeedSim.input['DerivativeError'] = DerivativeFollow
        self.DeltaSpeedSim.compute()
        self.PreviousErrorFollow = ErrorFollow
        return DerivativeFollow, self.DeltaSpeedSim.output['DeltaSpeed']
  
    def MotorAction(self, BaseSpeed, SteeringFollow):
        LeftSpeed = max(min(BaseSpeed + SteeringFollow, self.MaxVelocity), -self.MaxVelocity)
        RightSpeed = max(min(BaseSpeed - SteeringFollow, self.MaxVelocity), -self.MaxVelocity)

        self.LeftMotor.setVelocity(LeftSpeed)
        self.RightMotor.setVelocity(RightSpeed)
    
    def cleanup(self):
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)    
    
    def run(self):
        SensorSpeedRow = 0
        SensorSpeedWidth = 1
        SensorFollowRow = 7
        SensorFollowWidth = 1
        while self.Robot.step(self.TimeStep) != -1:
            CameraImage = self.ReadCamera()
            CameraImage, RoiDetectedSpeed, RoiCxSpeed, RoiCySpeed = self.GetReference(CameraImage, SensorSpeedRow, SensorSpeedWidth, drawDot=True, drawBox=True)
            CameraImage, RoiDetectedFollow, RoiCxFollow, RoiCyFollow = self.GetReference(CameraImage, SensorFollowRow, SensorFollowWidth, drawDot=True, drawBox=True)

            if RoiDetectedFollow: 
                ErrorFollow = self.GetErrorFollow(CameraImage, RoiCxFollow, SensorFollowWidth, drawLine=True)
            #else:
            #    ErrorFollow = 0
            #AngleSpeed = self.GetAngleSpeed(CameraImage, RoiDetectedSpeed, RoiDetectedFollow, SensorSpeedWidth, SensorFollowWidth, RoiCxSpeed, RoiCxFollow, SensorSpeedRow, SensorFollowRow, RoiCySpeed, RoiCyFollow, drawDot=True, drawLine=True)

            #BaseSpeed = self.CalculateBaseSpeedPID(AngleSpeed, 6.28)
            BaseSpeed = 2.0
            #DerivativeError, DeltaSpeed = self.CalculateDeltaSpeedPID(ErrorFollow)
            DerivativeError, DeltaSpeed = self.CalculateDeltaSpeedFuzzy(ErrorFollow)
            
            #print(f"Angle: {AngleSpeed:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {ErrorFollow:+06.2f}, DerivativeError: {DerivativeError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            #print(f"Angle: {AngleSpeed:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {ErrorFollow:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            print(f"Error: {ErrorFollow:+06.2f}, DerivativeError: {DerivativeError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            self.MotorAction(BaseSpeed, DeltaSpeed)            
            self.ShowCamera(CameraImage)

if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()