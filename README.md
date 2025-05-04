# 🤖 Line Follower Robot with PID, Fuzzy Logic, and Machine Learning Controllers 🚗

This project implements a **Line Follower Robot** using a combination of **PID Control**, **Fuzzy Logic**, and **Machine Learning** techniques for autonomous navigation in a **Webots** simulation environment. The robot utilizes various sensors to detect and follow a line, and its behavior can be controlled with multiple algorithms.

## 🌟 Features

* **Line Following**: The robot autonomously follows a line using a combination of **PID control**, **fuzzy logic**, and **machine learning models**.
* **Multiple Control Strategies**: Supports **PID**, **Fuzzy Logic**, and various **Machine Learning** models (e.g., **Decision Trees**, **Neural Networks**, etc.) for controlling base speed and delta speed.
* **Real-Time Camera Feed**: Displays the robot’s camera feed, processed to detect the line and adjust its movement accordingly.
* **Data Logging**: Logs real-time sensor data and control outputs for analysis, including GPS, IMU, and wheel speed data.
* **Learning Capabilities**: Pre-trained machine learning models are used for control, enhancing the robot’s performance over time.
* **Simulation in Webots**: Operates in a **Webots** simulated environment, providing realistic testing of the robot’s navigation and control algorithms.

## 📦 Requirements

### Hardware:

* **Webots** simulation environment
* Robot sensors and actuators defined in Webots (camera, GPS, IMU, wheels)

### Software:

* **Python 3.6+**
* **Webots R2025a** or newer
* Required Python libraries:

  * OpenCV (`cv2`)
  * Scikit-Fuzzy (`skfuzzy`)
  * Pandas (`pandas`)
  * NumPy (`numpy`)
  * Joblib (`joblib`)

To install the necessary dependencies, use the following:

```sh
pip install -r Project/controllers/requirements.txt
```

### 📁 File Structure

```
.
├── LICENSE
├── Project
│   ├── controllers
│   │   ├── line-follower
│   │   │   ├── line-follower.py
│   │   │   └── other
│   │   ├── requirements.txt
│   │   ├── robot_controller
│   │   │   ├── other
│   │   │   └── robot_controller.py
│   │   └── supervisor_controller
│   │       ├── other
│   │       └── supervisor_controller.py
│   ├── protos
│   │   ├── E-puck.proto
│   │   └── icons
│   │       └── E-puck.png
│   └── worlds
│       ├── arena_mod_light.wbt
│       ├── arena.wbt
│       ├── assets
│       │   ├── arena
│       │   ├── arena.png
│       │   ├── arena10.png
│       │   └── arena9.png
│       └── robot
│           └── Robot.urdf
└── README.md
```

### 🔑 Key Files:

* **`robot_controller.py`**: Main script for controlling the robot. It integrates the **PID**, **Fuzzy Logic**, and **Machine Learning** control algorithms to navigate the robot using sensor inputs.
* **`line-follower.py`**: Implements the line-following logic and sets up the robot’s behavior in Webots.
* **`supervisor_controller.py`**: Supervisory control for managing robot interactions with the environment in Webots.
* **`requirements.txt`**: Lists the dependencies required for the project.

## 🚀 Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/2black0/Line-Follower-Robot-PID-Fuzzy-Machine-Learning-Control.git
   cd Line-Follower-Robot-PID-Fuzzy-Machine-Learning-Control
   ```

2. **Set up the virtual environment** (optional but recommended):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install dependencies**:

   ```sh
   pip install -r Project/controllers/requirements.txt
   ```

4. **Start Webots Simulation**:

   * Open **Webots** and load the world files from the **worlds** directory.
   * Run the simulation with the appropriate controller for the robot.

## 🏃 Usage

1. **Run the robot control script**:

   * To use **PID** control:

   ```sh
   python Project/controllers/robot_controller.py
   ```

2. **Control Modes**:

   * You can switch between control modes for base speed and delta speed. The following control modes are available:

     * **PID**: Proportional-Integral-Derivative control.
     * **Fuzzy**: Fuzzy logic control based on predefined fuzzy rules.
     * **Machine Learning**: Pre-trained models (e.g., **Decision Trees**, **Neural Networks**) for controlling the robot.

   You can modify the control modes by updating the variables in the script:

   ```python
   BaseControl = 'PID'  # PID, Fuzzy, DecisionTree, NeuralNetworks, etc.
   DeltaControl = 'PID'  # PID, Fuzzy, DecisionTree, NeuralNetworks, etc.
   ```

3. **Enable Data Logging**:

   * To log sensor data and control parameters, set the `Log` parameter to `True` during robot initialization.

   Example:

   ```python
   LineFollower = LineFollower(Log=True, Camera=True, Learning=False)
   ```

4. **Enable Camera Feed**:

   * To display the camera feed in real-time, set the `Camera` parameter to `True`.

5. **Video Recording**:

   * To save the camera feed as a video file, enable the video recording feature by setting the `CameraSaved` parameter to `True`.

   Example:

   ```python
   LineFollower.run(CameraSaved=True)
   ```

## 🤖 Controller Description

The **robot\_controller.py** script handles the control algorithms:

* **PID Control**: Used for adjusting the robot’s speed and position based on the angle and error.
* **Fuzzy Logic**: A fuzzy control system that uses fuzzy rules to adjust the robot’s base speed and delta speed.
* **Machine Learning Models**: Pre-trained models like Decision Trees, Neural Networks, and Random Forests are used to predict base and delta speed based on sensor inputs.

### ⚙️ Fuzzy Control Rules

* **Angle Control**: Adjusts the base speed based on the robot’s angle relative to the line.
* **Error Control**: Uses fuzzy logic to adjust the robot's movement based on the sensor input and error correction.

### 🤖 Machine Learning Models

* The system uses pre-trained models for both base and delta speed control, improving the robot’s performance over time.

## 📊 Data Logging

The system logs various data (e.g., angle, speed, sensor data) during the simulation into a **CSV** file. This data can be used for analysis or further tuning of the control algorithms.

## 🔚 Conclusion

This project demonstrates the integration of **PID control**, **Fuzzy Logic**, and **Machine Learning** to create an autonomous **line-following robot** in **Webots**. By experimenting with different control strategies, you can optimize the robot’s performance in different environments.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---