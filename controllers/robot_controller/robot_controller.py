import cv2
import os
import math
import csv
import numpy as np
import skfuzzy as fuzz
from controller import Robot
from skfuzzy import control as ctrl

class LineFollower:
    def __init__(self):
        self.Robot = Robot()
        self.TimeStep = int(self.Robot.getBasicTimeStep())
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
        
        self.SensorSpeedRow = 0
        self.SensorSpeedWidth = 1
        self.SensorFollowRow = 11
        self.SensorFollowWidth = 0.5
        self.SetPoint = (self.CameraWidth * self.SensorFollowWidth) // 2
        
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
        
        Error = ctrl.Antecedent(np.arange(-320, 321, 1), 'Error')
        DeltaError = ctrl.Antecedent(np.arange(-320, 321, 1), 'DeltaError')
        DeltaSpeed = ctrl.Consequent(np.arange(-15, 16, 1), 'DeltaSpeed')

        Error['NL'] = fuzz.trimf(Error.universe, [-320, -320, -160])
        Error['NS'] = fuzz.trimf(Error.universe, [-320, -160, 0])
        Error['Z'] = fuzz.trimf(Error.universe, [-160, 0, 160])
        Error['PS'] = fuzz.trimf(Error.universe, [0, 160, 320])
        Error['PL'] = fuzz.trimf(Error.universe, [160, 320, 320])

        DeltaError['NL'] = fuzz.trimf(DeltaError.universe, [-320, -320, -160])
        DeltaError['NS'] = fuzz.trimf(DeltaError.universe, [-320, -160, 0])
        DeltaError['Z'] = fuzz.trimf(DeltaError.universe, [-160, 0, 160])
        DeltaError['PS'] = fuzz.trimf(DeltaError.universe, [0, 160, 320])
        DeltaError['PL'] = fuzz.trimf(DeltaError.universe, [160, 320, 320])

        DeltaSpeed['DL'] = fuzz.trimf(DeltaSpeed.universe, [-15, -15, -7.5])
        DeltaSpeed['DS'] = fuzz.trimf(DeltaSpeed.universe, [-15, -7.5, 0])
        DeltaSpeed['NC'] = fuzz.trimf(DeltaSpeed.universe, [-7.5, 0, 7.5])
        DeltaSpeed['IS'] = fuzz.trimf(DeltaSpeed.universe, [0, 7.5, 15])
        DeltaSpeed['IL'] = fuzz.trimf(DeltaSpeed.universe, [7.5, 15, 15])

        rule1 = ctrl.Rule(Error['NL'] & DeltaError['NL'], DeltaSpeed['DL'])
        rule2 = ctrl.Rule(Error['NL'] & DeltaError['NS'], DeltaSpeed['DL'])
        rule3 = ctrl.Rule(Error['NL'] & DeltaError['Z'], DeltaSpeed['DL'])
        rule4 = ctrl.Rule(Error['NL'] & DeltaError['PS'], DeltaSpeed['DS'])
        rule5 = ctrl.Rule(Error['NL'] & DeltaError['PL'], DeltaSpeed['NC'])

        rule6 = ctrl.Rule(Error['NS'] & DeltaError['NL'], DeltaSpeed['DL'])
        rule7 = ctrl.Rule(Error['NS'] & DeltaError['NS'], DeltaSpeed['DS'])
        rule8 = ctrl.Rule(Error['NS'] & DeltaError['Z'], DeltaSpeed['DS'])
        rule9 = ctrl.Rule(Error['NS'] & DeltaError['PS'], DeltaSpeed['NC'])
        rule10 = ctrl.Rule(Error['NS'] & DeltaError['PL'], DeltaSpeed['IS'])

        rule11 = ctrl.Rule(Error['Z'] & DeltaError['NL'], DeltaSpeed['DS'])
        rule12 = ctrl.Rule(Error['Z'] & DeltaError['NS'], DeltaSpeed['DS'])
        rule13 = ctrl.Rule(Error['Z'] & DeltaError['Z'], DeltaSpeed['NC'])
        rule14 = ctrl.Rule(Error['Z'] & DeltaError['PS'], DeltaSpeed['IS'])
        rule15 = ctrl.Rule(Error['Z'] & DeltaError['PL'], DeltaSpeed['IS'])

        rule16 = ctrl.Rule(Error['PS'] & DeltaError['NL'], DeltaSpeed['NC'])
        rule17 = ctrl.Rule(Error['PS'] & DeltaError['NS'], DeltaSpeed['NC'])
        rule18 = ctrl.Rule(Error['PS'] & DeltaError['Z'], DeltaSpeed['IS'])
        rule19 = ctrl.Rule(Error['PS'] & DeltaError['PS'], DeltaSpeed['IL'])
        rule20 = ctrl.Rule(Error['PS'] & DeltaError['PL'], DeltaSpeed['IL'])

        rule21 = ctrl.Rule(Error['PL'] & DeltaError['NL'], DeltaSpeed['NC'])
        rule22 = ctrl.Rule(Error['PL'] & DeltaError['NS'], DeltaSpeed['IS'])
        rule23 = ctrl.Rule(Error['PL'] & DeltaError['Z'], DeltaSpeed['IS'])
        rule24 = ctrl.Rule(Error['PL'] & DeltaError['PS'], DeltaSpeed['IL'])
        rule25 = ctrl.Rule(Error['PL'] & DeltaError['PL'], DeltaSpeed['IL'])

        DeltaSpeed_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5,
            rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15,
            rule16, rule17, rule18, rule19, rule20,
            rule21, rule22, rule23, rule24, rule25
        ])
        self.DeltaSpeedSim = ctrl.ControlSystemSimulation(DeltaSpeed_ctrl)
        
        self.FileName = 'output.csv'
        with open(self.FileName, 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['Time', 'SetPoint', 'Error', 'DeltaError', 'DeltaSpeed', 'BaseSpeed', 'LeftSpeed', 'RightSpeed', 'Angle'])

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

    def GetErrorFollow(self, CameraImage, RoiCxFollow, drawLine=False):
        if drawLine:
            cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        ErrorFollow = RoiCxFollow - self.SetPoint
        return ErrorFollow

    def GetAngleSpeed(self, CameraImage, 
                      RoiDetectedSpeed, RoiDetectedFollow, 
                      SensorSpeedWidth,
                      RoiCxSpeed, RoiCxFollow, 
                      SensorSpeedRow, SensorFollowRow, 
                      RoiCySpeed, RoiCyFollow, 
                      drawDot=False, drawLine=False):
        AngleSpeed = 60        
        if RoiDetectedSpeed and RoiDetectedFollow:
            FirstPointCx = int(RoiCxFollow - self.SetPoint + (self.CameraWidth / 2))
            FirstPointCy = int(self.RowHeight * SensorSpeedRow + RoiCySpeed)
            
            SecondPointCx = int(RoiCxFollow - self.SetPoint + (self.CameraWidth / 2))
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
        self.DeltaSpeedSim.input['DeltaError'] = DerivativeFollow
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
    
    def LogData(self, FileName, TimeStep, SetPoint, Error, DeltaError, DeltaSpeed, BaseSpeed, LeftSpeed, RightSpeed, Angle):
        with open(FileName, 'a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow([TimeStep, SetPoint, Error, DeltaError, DeltaSpeed, BaseSpeed, LeftSpeed, RightSpeed, Angle])

    def run(self):
        while self.Robot.step(self.TimeStep) != -1:
            Time = self.Robot.getTime()
            
            CameraImage = self.ReadCamera()
            print(CameraImage)
            #CameraImage, RoiDetectedSpeed, RoiCxSpeed, RoiCySpeed = self.GetReference(CameraImage, self.SensorSpeedRow, self.SensorSpeedWidth, drawDot=True, drawBox=True)
            #CameraImage, RoiDetectedFollow, RoiCxFollow, RoiCyFollow = self.GetReference(CameraImage, self.SensorFollowRow, self.SensorFollowWidth, drawDot=True, drawBox=True)

            #if RoiDetectedFollow: 
            #    Error = self.GetErrorFollow(CameraImage, RoiCxFollow, drawLine=True)
            #else:
            #    Error = 0
            #Angle = self.GetAngleSpeed(CameraImage, RoiDetectedSpeed, RoiDetectedFollow, self.SensorSpeedWidth, RoiCxSpeed, RoiCxFollow, self.SensorSpeedRow, self.SensorFollowRow, RoiCySpeed, RoiCyFollow, drawDot=True, drawLine=True)

            #BaseSpeed = self.CalculateBaseSpeedPID(Angle, 6.28)
            #BaseSpeed = 4.0
            #DeltaError, DeltaSpeed = self.CalculateDeltaSpeedPID(Error)
            #DeltaError, DeltaSpeed = self.CalculateDeltaSpeedFuzzy(Error)
            
            #print(f"Angle: {Angle:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {Error:+06.2f}, DerivativeError: {DeltaError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            #print(f"Angle: {AngleSpeed:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {ErrorFollow:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            #print(f"Error: {Error:+06.2f}, DeltaError: {DeltaError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            
            #self.MotorAction(BaseSpeed, DeltaSpeed)            
            #self.LogData(self.FileName, Time, self.SetPoint, Error, DeltaError, DeltaSpeed, BaseSpeed, BaseSpeed+DeltaSpeed, BaseSpeed-DeltaSpeed, Angle)

            #self.ShowCamera(CameraImage)

if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()