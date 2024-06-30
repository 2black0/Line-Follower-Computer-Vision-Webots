import os
import csv
import cv2
import numpy as np
from controller import Robot

class LineFollower:
    def __init__(self, Log=False, Print=False, Camera=False, CameraSaved=False): 
        self.Robot = Robot()
        self.TimeStep = int(self.Robot.getBasicTimeStep())
        
        self.InitSensor()
        self.InitMotor()        
        self.InitPID()
        self.LogStatus = Log
        self.Print = Print
        if self.LogStatus:
            self.InitCSV()
        self.CameraStatus = Camera
        if self.CameraStatus:
            self.InitRecording()
        self.CameraSaved = CameraSaved

    def InitSensor(self):
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
        
        SensorAngleRow = 4
        SensorAngleWidth = 1
        SetPointAngle = 0
        self.SensorAngle = [SetPointAngle, SensorAngleRow, SensorAngleWidth] # SetPoint, Row, Width
        
        SensorErrorRow = 7
        SensorErrorWidth = 1
        SetPointError = (self.CameraWidth * SensorErrorWidth) // 2
        self.SensorError = [SetPointError, SensorErrorRow, SensorErrorWidth] # SetPoint, Row, Width
        
        self.StartPosition = None
        self.MovingStatus = False

    def InitMotor(self):
        self.LeftMotor = self.Robot.getDevice('left wheel motor')
        self.RightMotor = self.Robot.getDevice('right wheel motor')
        self.LeftMotor.setPosition(float('inf'))
        self.RightMotor.setPosition(float('inf'))
        self.LeftMotor.setVelocity(0.0)
        self.RightMotor.setVelocity(0.0)
        self.MaxVelocity = 6.28

    def InitPID(self):
        self.KpBaseSpeed = 0.05
        self.KdBaseSpeed = 0.001
        self.IntegralAngle = 0
        self.PreviousAngle = 0
        
        self.KpDeltaSpeed = 0.05
        self.KiDeltaSpeed = 0.005
        self.KdDeltaSpeed = 0.0045
        self.IntegralError = 0
        self.PreviousError = 0

    def InitCSV(self):
        self.FileName = 'output.csv'
        if os.path.exists(self.FileName):
            os.remove(self.FileName)
        else:
            print("The file does not exist")
        
        with open(self.FileName, 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['Time', 
                                 'SetPointAngle', 'SensorAngleRow', 'SensorAngleWidth', 'Angle', 'IntegralAngle', 'DeltaAngle', 'BaseSpeed', 
                                 'SetPointError', 'SensorErrorRow', 'SensorErrorWidth', 'Error', 'IntegralError', 'DeltaError', 'DeltaSpeed', 
                                 'LeftSpeed', 'RightSpeed', 
                                 'X', 'Y', 'Z', 
                                 'Roll', 'Pitch', 'Yaw'])

    def InitRecording(self):
        self.VideoFileName = 'output.avi'
        self.VideoOut = cv2.VideoWriter(self.VideoFileName,  cv2.VideoWriter_fourcc(*'MJPG'), 30, (self.CameraWidth, self.CameraHeight)) 

    def ReadCamera(self):
        CameraImage = self.Camera.getImage()
        CameraImage = np.frombuffer(CameraImage, np.uint8).reshape((self.CameraHeight, self.CameraWidth, 4))
        CameraImage = cv2.cvtColor(CameraImage, cv2.COLOR_BGRA2BGR)
        return CameraImage

    def GetReference(self, CameraImage, SensorConfig, drawDot=False, drawBox=False):
        SensorRow = SensorConfig[1]
        SensorWidth = SensorConfig[2] 
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
        ReferenceValue = [RoiDetected, RoiCx, RoiCy]
        return CameraImage, ReferenceValue        

    def GetError(self, CameraImage, ReferenceValue, drawLine=False):
        RoiDetectedError = ReferenceValue[0]
        RoiCxError = ReferenceValue[1]
        Error = self.PreviousError
        if RoiDetectedError:
            Error = RoiCxError - self.SensorError[0]
        if drawLine:
            cv2.line(CameraImage, (self.CameraWidth // 2, 0), (self.CameraWidth // 2, self.CameraHeight), (0, 0, 255), 1)
        return Error

    def GetAngle(self, CameraImage, ReferenceValueError, ReferenceValueAngle, drawDot=False, drawLine=False):
        RoiDetectedError = ReferenceValueError[0]
        RoiCxError = ReferenceValueError[1]
        RoiCyError = ReferenceValueError[2]
        
        RoiDetectedAngle = ReferenceValueAngle[0]
        RoiCxAngle = ReferenceValueAngle[1]
        RoiCyAngle = ReferenceValueAngle[2]
        
        Angle = 75   
        if RoiDetectedError and RoiDetectedAngle:
            FirstPointCx = int(RoiCxError - self.SensorError[0] + (self.CameraWidth / 2))
            FirstPointCy = int(self.RowHeight * self.SensorAngle[1] + RoiCyAngle)
            
            SecondPointCx = int(RoiCxError - self.SensorError[0] + (self.CameraWidth / 2))
            SecondPointCy = int(self.RowHeight * self.SensorError[1] + RoiCyError)
            
            ThirdPointCx = int((RoiCxAngle - (self.CameraWidth * self.SensorError[2]) // 2) + (self.CameraWidth / 2))
            ThirdPointCy = int(self.RowHeight * self.SensorAngle[1] + RoiCyAngle)
            
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

    def ShowCamera(self, CameraImage, CameraSaved=False):
        ImageResize = cv2.resize(CameraImage, (320, 240), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Camera Image from e-puck", ImageResize)
        if CameraSaved:
            self.VideoOut.write(ImageResize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()

    def CalculateBaseSpeed(self, Angle, MaxSpeed):
        DeltaAngle = Angle - self.PreviousAngle
        self.IntegralAngle += Angle
        BaseSpeed = MaxSpeed - (self.KpBaseSpeed * Angle) - (self.KdBaseSpeed * DeltaAngle)
        self.PreviousAngle = Angle
        AngleValue = [Angle, self.IntegralAngle, DeltaAngle]
        return AngleValue, BaseSpeed

    def CalculateDeltaSpeed(self, Error):
        DeltaError = Error - self.PreviousError
        self.IntegralError += Error
        DeltaSpeed = (self.KpDeltaSpeed * Error) + (self.KiDeltaSpeed * self.IntegralError) + (self.KdDeltaSpeed * DeltaError)  
        self.PreviousError = Error
        ErrorValue = [Error, self.IntegralError, DeltaError]
        return ErrorValue, DeltaSpeed
  
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
        
    def LogData(self, FileName, Time, SensorAngle, AngleValue, BaseSpeed, SensorError, ErrorValue, DeltaSpeed, LeftSpeed, RightSpeed, Position, Orientation):
        with open(FileName, 'a', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            SetPointAngle = SensorAngle[0]
            SensorAngleRow = SensorAngle[1]
            SensorAngleWidth = SensorAngle[2]
            Angle = AngleValue[0]
            IntegralAngle = AngleValue[1]
            DeltaAngle = AngleValue[2]
            SetPointError = SensorError[0]
            SensorErrorRow = SensorError[1]
            SensorErrorWidth = SensorError[2]
            Error = ErrorValue[0]
            IntegralError = ErrorValue[1]
            DeltaError = ErrorValue[2]
            X = Position[0]
            Y = Position[1]
            Z = Position[2]
            Roll = Orientation[0]
            Pitch = Orientation[1]
            Yaw = Orientation[2]
            log_writer.writerow([Time, 
                                SetPointAngle, SensorAngleRow, SensorAngleWidth, Angle, IntegralAngle, DeltaAngle, BaseSpeed, 
                                SetPointError, SensorErrorRow, SensorErrorWidth, Error, IntegralError, DeltaError, DeltaSpeed, 
                                LeftSpeed, RightSpeed, 
                                X, Y, Z, Roll, Pitch, Yaw])

    def PrintData(self, Time, SensorAngle, AngleValue, BaseSpeed, SensorError, ErrorValue, DeltaSpeed, LeftSpeed, RightSpeed, Position, Orientation):
        SetPointAngle = SensorAngle[0]
        SensorAngleRow = SensorAngle[1]
        SensorAngleWidth = SensorAngle[2]
        Angle = AngleValue[0]
        IntegralAngle = AngleValue[1]
        DeltaAngle = AngleValue[2]
        SetPointError = SensorError[0]
        SensorErrorRow = SensorError[1]
        SensorErrorWidth = SensorError[2]
        Error = ErrorValue[0]
        IntegralError = ErrorValue[1]
        DeltaError = ErrorValue[2]
        X = Position[0]
        Y = Position[1]
        Z = Position[2]
        Roll = Orientation[0]
        Pitch = Orientation[1]
        Yaw = Orientation[2]
        print(f"Angle: {Angle:+06.2f}, DeltaAngle: {DeltaAngle:+06.2f}, BaseSpeed: {BaseSpeed:+06.2f}, Error: {Error:+06.2f}, DeltaError: {DeltaError:+06.2f}, DeltaSpeed: {DeltaSpeed:+06.2f}")        
            
    def run(self):
        while self.Robot.step(self.TimeStep) != -1:
            Time = self.Robot.getTime()
            
            Position = self.ReadGPS()
            Orientation = self.ReadIMU()
            CameraImage = self.ReadCamera()
            
            SensorErrorRow = 7
            SensorErrorWidth = 1
            SetPointError = (self.CameraWidth * SensorErrorWidth) // 2
            self.SensorError = [SetPointError, SensorErrorRow, SensorErrorWidth] # SetPoint, Row, Width
            
            CameraImage, ReferenceValueAngle = self.GetReference(CameraImage, self.SensorAngle, drawDot=True, drawBox=True)
            CameraImage, ReferenceValueError = self.GetReference(CameraImage, self.SensorError, drawDot=True, drawBox=True)
            
            #ReferenceValue = [RoiDetected, RoiCx, RoiCy]
            #ReferenceValue = []
            #for i in range(self.RowTotal):
            #    SensorConfig = [(self.CameraWidth * 1) // 2, i, 1]
            #    CameraImage, ReferenceValueResult = self.GetReference(CameraImage, SensorConfig, drawDot=True, drawBox=True)
            #    ReferenceValue.append(ReferenceValueResult)

            Angle = self.GetAngle(CameraImage, ReferenceValueError, ReferenceValueAngle, drawDot=True, drawLine=True)
            Error = self.GetError(CameraImage, ReferenceValueError, drawLine=True)
            
            AngleValue, BaseSpeed = self.CalculateBaseSpeed(Angle, 6.28) 
            ErrorValue, DeltaSpeed = self.CalculateDeltaSpeed(Error)            
            
            LeftSpeed, RightSpeed = self.MotorAction(BaseSpeed, DeltaSpeed)
            #LeftSpeed, RightSpeed = 0, 0
            
            if self.Print:
                self.PrintData(Time, self.SensorAngle, AngleValue, BaseSpeed, self.SensorError, ErrorValue, DeltaSpeed, LeftSpeed, RightSpeed, Position, Orientation)
            #if self.LogStatus:
            #    self.LogData(self.FileName, Time, self.SensorAngle, AngleValue, BaseSpeed, self.SensorError, ErrorValue, DeltaSpeed, LeftSpeed, RightSpeed, Position, Orientation)
            if self.CameraStatus:
                self.ShowCamera(CameraImage, CameraSaved=self.CameraSaved)

if __name__ == "__main__":
    LineFollower = LineFollower(Log=False, Print=True, Camera=True, CameraSaved=False)
    LineFollower.run()