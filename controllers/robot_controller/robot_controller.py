import os
import csv
import cv2
import joblib
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from controller import Robot
from skfuzzy import control as ctrl

class LineFollower:
    def __init__(self): 
        self.Robot = Robot()
        self.TimeStep = 16#int(self.Robot.getBasicTimeStep())
        
        self.Camera = self.Robot.getDevice('camera')
        self.Camera.enable(self.TimeStep)
        self.CameraWidth = self.Camera.getWidth()
        self.CameraHeight = self.Camera.getHeight()
        
        self.GPS = self.Robot.getDevice('gps')
        self.GPS.enable(self.TimeStep)

        self.IMU = self.Robot.getDevice('accelerometer')
        self.IMU.enable(self.TimeStep)

        self.RowTotal = 12
        self.RowHeight = int(self.CameraHeight / self.RowTotal)
        
        self.LeftMotor = self.Robot.getDevice('left wheel motor')
        self.RightMotor = self.Robot.getDevice('right wheel motor')
        self.LeftMotor.setPosition(float('inf'))
        self.RightMotor.setPosition(float('inf'))
        self.LeftMotor.setVelocity(0.0)
        self.RightMotor.setVelocity(0.0)
        
        self.MaxVelocity = 6.28
        
        self.SensorAngleRow = 4
        self.SensorAngleWidth = 1
        self.SetPointAngle = 0
        
        self.SensorErrorRow = 7
        self.SensorErrorWidth = 1
        self.SetPointError = (self.CameraWidth * self.SensorErrorWidth) // 2
        
        self.KpBaseSpeed = 0.05
        self.KdBaseSpeed = 0.001
        self.PreviousAngle = 0
        
        self.KpDeltaSpeed = 0.05
        self.KiDeltaSpeed = 0.005
        self.KdDeltaSpeed = 0.0045
        self.IntegralError = 0
        self.PreviousError = 0
        
        Error = ctrl.Antecedent(np.arange(-320, 321, 1), 'Error')
        DeltaError = ctrl.Antecedent(np.arange(-320, 321, 1), 'DeltaError')
        DeltaSpeed = ctrl.Consequent(np.arange(-15, 16, 1), 'DeltaSpeed')

        Error['NL'] = fuzz.trimf(Error.universe, [-320, -320, -150])
        Error['NS'] = fuzz.trimf(Error.universe, [-320, -150, 0])
        Error['Z'] = fuzz.trimf(Error.universe, [-150, 0, 150])
        Error['PS'] = fuzz.trimf(Error.universe, [0, 150, 320])
        Error['PL'] = fuzz.trimf(Error.universe, [150, 320, 320])

        DeltaError['NL'] = fuzz.trimf(DeltaError.universe, [-320, -320, -150])
        DeltaError['NS'] = fuzz.trimf(DeltaError.universe, [-320, -150, 0])
        DeltaError['Z'] = fuzz.trimf(DeltaError.universe, [-150, 0, 150])
        DeltaError['PS'] = fuzz.trimf(DeltaError.universe, [0, 150, 320])
        DeltaError['PL'] = fuzz.trimf(DeltaError.universe, [150, 320, 320])

        DeltaSpeed['DL'] = fuzz.trimf(DeltaSpeed.universe, [-15, -15, -10])
        DeltaSpeed['DS'] = fuzz.trimf(DeltaSpeed.universe, [-15, -10, 0])
        DeltaSpeed['NC'] = fuzz.trimf(DeltaSpeed.universe, [-10, 0, 10])
        DeltaSpeed['IS'] = fuzz.trimf(DeltaSpeed.universe, [0, 10, 15])
        DeltaSpeed['IL'] = fuzz.trimf(DeltaSpeed.universe, [10, 15, 15])

        Rule1 = ctrl.Rule(Error['NL'] & DeltaError['NL'], DeltaSpeed['DL'])
        Rule2 = ctrl.Rule(Error['NL'] & DeltaError['NS'], DeltaSpeed['DL'])
        Rule3 = ctrl.Rule(Error['NL'] & DeltaError['Z'], DeltaSpeed['DL'])
        Rule4 = ctrl.Rule(Error['NL'] & DeltaError['PS'], DeltaSpeed['DS'])
        Rule5 = ctrl.Rule(Error['NL'] & DeltaError['PL'], DeltaSpeed['NC'])

        Rule6 = ctrl.Rule(Error['NS'] & DeltaError['NL'], DeltaSpeed['DL'])
        Rule7 = ctrl.Rule(Error['NS'] & DeltaError['NS'], DeltaSpeed['DS'])
        Rule8 = ctrl.Rule(Error['NS'] & DeltaError['Z'], DeltaSpeed['DS'])
        Rule9 = ctrl.Rule(Error['NS'] & DeltaError['PS'], DeltaSpeed['NC'])
        Rule10 = ctrl.Rule(Error['NS'] & DeltaError['PL'], DeltaSpeed['IS'])

        Rule11 = ctrl.Rule(Error['Z'] & DeltaError['NL'], DeltaSpeed['DS'])
        Rule12 = ctrl.Rule(Error['Z'] & DeltaError['NS'], DeltaSpeed['NC'])
        Rule13 = ctrl.Rule(Error['Z'] & DeltaError['Z'], DeltaSpeed['NC'])
        Rule14 = ctrl.Rule(Error['Z'] & DeltaError['PS'], DeltaSpeed['IS'])
        Rule15 = ctrl.Rule(Error['Z'] & DeltaError['PL'], DeltaSpeed['IL'])

        Rule16 = ctrl.Rule(Error['PS'] & DeltaError['NL'], DeltaSpeed['NC'])
        Rule17 = ctrl.Rule(Error['PS'] & DeltaError['NS'], DeltaSpeed['NC'])
        Rule18 = ctrl.Rule(Error['PS'] & DeltaError['Z'], DeltaSpeed['IS'])
        Rule19 = ctrl.Rule(Error['PS'] & DeltaError['PS'], DeltaSpeed['IL'])
        Rule20 = ctrl.Rule(Error['PS'] & DeltaError['PL'], DeltaSpeed['IL'])

        Rule21 = ctrl.Rule(Error['PL'] & DeltaError['NL'], DeltaSpeed['NC'])
        Rule22 = ctrl.Rule(Error['PL'] & DeltaError['NS'], DeltaSpeed['IS'])
        Rule23 = ctrl.Rule(Error['PL'] & DeltaError['Z'], DeltaSpeed['IS'])
        Rule24 = ctrl.Rule(Error['PL'] & DeltaError['PS'], DeltaSpeed['IL'])
        Rule25 = ctrl.Rule(Error['PL'] & DeltaError['PL'], DeltaSpeed['IL'])

        DeltaSpeedControl = ctrl.ControlSystem([
            Rule1, Rule2, Rule3, Rule4, Rule5,
            Rule6, Rule7, Rule8, Rule9, Rule10,
            Rule11, Rule12, Rule13, Rule14, Rule15,
            Rule16, Rule17, Rule18, Rule19, Rule20,
            Rule21, Rule22, Rule23, Rule24, Rule25
        ])
        self.DeltaSpeedSim = ctrl.ControlSystemSimulation(DeltaSpeedControl)
        
        self.BaseSpeedModel = joblib.load('BaseSpeedModel.pkl')
        self.DeltaSpeedModel = joblib.load('DeltaSpeedModel.pkl')
        
        self.FileName = 'output.csv'
        if os.path.exists(self.FileName):
            os.remove(self.FileName)
        else:
            print("The file does not exist")
        
        with open(self.FileName, 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['TimeStep', 'SetPointAngle', 'Angle', 'BaseSpeed', 'SetPointError', 'Error', 'DeltaError', 'DeltaSpeed', 'LeftSpeed', 'RightSpeed', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'])

        self.VideoFileName = 'output.avi'
        self.VideoOut = cv2.VideoWriter(self.VideoFileName,  cv2.VideoWriter_fourcc(*'MJPG'), 30, (self.CameraWidth, self.CameraHeight)) 

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

    def GetError(self, CameraImage, RoiDetectedError, RoiCxError, drawLine=False):
        Error = self.PreviousError
        if RoiDetectedError:
            Error = RoiCxError - self.SetPointError
        if drawLine:
            cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        return Error

    def GetAngle(self, CameraImage, 
                 RoiDetectedError, RoiDetectedAngle, 
                 SensorAngleWidth, SensorErrorWidth,
                 RoiCxAngle, RoiCxError, 
                 SensorAngleRow, SensorErrorRow, 
                 RoiCyAngle, RoiCyError, 
                 drawDot=False, drawLine=False):
        Angle = 75   
        if RoiDetectedError and RoiDetectedAngle:
            FirstPointCx = int(RoiCxError - self.SetPointError + (self.CameraWidth / 2))
            FirstPointCy = int(self.RowHeight * SensorAngleRow + RoiCyAngle)
            
            SecondPointCx = int(RoiCxError - self.SetPointError + (self.CameraWidth / 2))
            SecondPointCy = int(self.RowHeight * SensorErrorRow + RoiCyError)
            
            ThirdPointCx = int((RoiCxAngle - (self.CameraWidth * SensorErrorWidth) // 2) + (self.CameraWidth / 2))
            ThirdPointCy = int(self.RowHeight * SensorAngleRow + RoiCyAngle)
            
            PointA = np.array([FirstPointCx, FirstPointCy])
            PointB = np.array([SecondPointCx, SecondPointCy])
            PointC = np.array([ThirdPointCx, ThirdPointCy])
            
            PointBA = PointA - PointB
            PointBC = PointC - PointB
        
            CosineAngle = np.dot(PointBA, PointBC) / (np.linalg.norm(PointBA) * np.linalg.norm(PointBC))
            AngleRadian = np.arccos(CosineAngle)
            Angle = np.degrees(AngleRadian)
        
            if drawDot:
                cv2.circle(CameraImage, PointA, 3, (0, 0, 255), -1)
                cv2.circle(CameraImage, PointB, 3, (0, 0, 255), -1)
                cv2.circle(CameraImage, PointC, 3, (0, 0, 255), -1)
            if drawLine:
                cv2.line(CameraImage, PointA, PointB, (0, 255, 0), 2)
                cv2.line(CameraImage, PointB, PointC, (0, 255, 0), 2)
        return Angle

    def ShowCamera(self, CameraImage, Show=False, Saved=False):
        ImageResize = cv2.resize(CameraImage, (320, 240), interpolation=cv2.INTER_LINEAR)
        if Show:
            cv2.imshow("Camera Image from e-puck", ImageResize)
        if Saved:
            self.VideoOut.write(ImageResize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def CalculateBaseSpeed(self, Angle, MaxSpeed, Control='PID'):
        DeltaAngle = Angle - self.PreviousAngle
        if Control == 'PID':
            BaseSpeed = MaxSpeed - (self.KpBaseSpeed * Angle) - (self.KdBaseSpeed * DeltaAngle)
        elif Control == 'Learning':
            BaseSpeedData = pd.DataFrame({
                'Angle': [Angle]
            })
            BaseSpeed = self.BaseSpeedModel.predict(BaseSpeedData)[0]
        self.PreviousAngle = Angle
        return DeltaAngle, BaseSpeed

    def CalculateDeltaSpeed(self, Error, DeltaSpeed, Control='PID'):
        DeltaError = Error - self.PreviousError
        if Control == 'PID':
            self.IntegralError += Error
            DeltaSpeed = (self.KpDeltaSpeed * Error) + (self.KiDeltaSpeed * self.IntegralError) + (self.KdDeltaSpeed * DeltaError)
        elif Control == "Fuzzy":
            Error = max(min(Error, 320), -320)
            DeltaError = max(min(DeltaError, 320), -320)
            self.DeltaSpeedSim.input['Error'] = Error
            self.DeltaSpeedSim.input['DeltaError'] = DeltaError
            self.DeltaSpeedSim.compute()
        elif Control == 'Learning':
            DeltaSpeedData = pd.DataFrame({
                'Error': [Error],
                'DeltaError': [DeltaError]
            })
            DeltaSpeed = self.DeltaSpeedModel.predict(DeltaSpeedData)[0]
        self.PreviousError = Error
        return DeltaError, DeltaSpeed
  
    def MotorAction(self, BaseSpeed, DeltaSpeed):
        LeftSpeed = max(min(BaseSpeed + DeltaSpeed, self.MaxVelocity), -self.MaxVelocity)
        RightSpeed = max(min(BaseSpeed - DeltaSpeed, self.MaxVelocity), -self.MaxVelocity)
        self.LeftMotor.setVelocity(LeftSpeed)
        self.RightMotor.setVelocity(RightSpeed)
        return LeftSpeed, RightSpeed
    
    def cleanup(self):
        self.VideoOut.release()
        cv2.destroyAllWindows()
        self.robot.simulationQuit(0)    
    
    def ReadGPS(self):
        Position = self.GPS.getValues()
        return Position
    
    def ReadIMU(self):    
        Orientation = self.IMU.getValues()
        return Orientation
        
    def LogData(self, FileName, TimeStep, SetPointAngle, Angle, BaseSpeed, SetPointError, Error, DeltaError, DeltaSpeed, LeftSpeed, RightSpeed, X, Y, Z, Roll, Pitch, Yaw):
        with open(FileName, 'a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow([TimeStep, SetPointAngle, Angle, BaseSpeed, SetPointError, Error, DeltaError, DeltaSpeed, LeftSpeed, RightSpeed, X, Y, Z, Roll, Pitch, Yaw])

    def run(self):
        while self.Robot.step(self.TimeStep) != -1:
            Time = self.Robot.getTime()
            
            Position = self.ReadGPS()
            Orientation = self.ReadIMU()
            
            CameraImage = self.ReadCamera()
            CameraImage, RoiDetectedAngle, RoiCxAngle, RoiCyAngle = self.GetReference(CameraImage, self.SensorAngleRow, self.SensorAngleWidth, drawDot=True, drawBox=True)
            CameraImage, RoiDetectedError, RoiCxError, RoiCyError = self.GetReference(CameraImage, self.SensorErrorRow, self.SensorErrorWidth, drawDot=True, drawBox=True)

            Error = self.GetError(CameraImage, RoiDetectedError, RoiCxError, drawLine=True)
            Angle = self.GetAngle(CameraImage, RoiDetectedError, RoiDetectedAngle, 
                                  self.SensorAngleWidth, self.SensorErrorWidth,
                                  RoiCxAngle, RoiCxError, 
                                  self.SensorAngleRow, self.SensorErrorRow, 
                                  RoiCyAngle, RoiCyError, 
                                  drawDot=True, drawLine=True)
            
            DeltaAngle, BaseSpeed = self.CalculateBaseSpeed(Angle, 6.28, 'Learning')
            #BaseSpeed = 3.0
            DeltaError, DeltaSpeed = self.CalculateDeltaSpeed(Error, 'Learning')
            
            print(f"Angle: {Angle:+06.2f}, DeltaAngle: {DeltaAngle:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {Error:+06.2f}, DeltaError: {DeltaError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")
            
            LeftSpeed, RightSpeed = self.MotorAction(BaseSpeed, DeltaSpeed)            
                        
            self.LogData(self.FileName, Time, 
                         self.SetPointAngle, Angle, BaseSpeed, 
                         self.SetPointError, Error, DeltaError, 
                         DeltaSpeed, LeftSpeed, RightSpeed, 
                         Position[0], Position[1], Position[2],
                         Orientation[0], Orientation[1], Orientation[2])

            self.ShowCamera(CameraImage, Show=True, Saved=False)

if __name__ == "__main__":
    LineFollower = LineFollower()
    LineFollower.run()