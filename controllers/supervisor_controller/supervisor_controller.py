from controller import Supervisor

# Initialize supervisor
supervisor = Supervisor()

# Time step
timestep = int(supervisor.getBasicTimeStep())

# Robot reference
robot_node = supervisor.getFromDef("ROBOT")  # Replace ROBOT_NAME with the DEF name of your robot

# Final Waypoints
waypoints = [
    [0.46488879953276824, 0.5248615537783488, -7.284879693758399e-05],
    [0.4754894656538557, 0.21917057545533003, -7.192309795408906e-05],
    [0.6079301094855897, -0.21426830988797166, -7.730153196513698e-05],
    [-0.0614526298441335, -0.429289680016794, -6.671217632723062e-05],
    [-0.09432875718822165, 0.0040431185951938006, -7.321312532702559e-05],
    [-0.3645235239106699, -0.4880005385506754, -9.506281676057403e-05],
    [-0.8030341069954259, -0.02566443366172788, -6.414136037328697e-05],
    [-0.11715503137128479, 0.5480391376758033, -6.946751218004851e-05],
    [-0.5168625308618092, 0.32126461740941986, -8.27739178609066e-05]
]

# Logging setup
lap = 0
waypoint_index = 0
initial_position = [-0.29070500000000005, 0.1427280000000019, -6.396199575281827e-05]  # Set initial position here
at_initial_position = False  # To check if robot is at the initial position

def log_status(lap, current_time, waypoint_index):
    log_message = f"Lap: {lap}, Time: {current_time:.2f}, Waypoint: {waypoint_index}"
    #print(log_message)

# Log initial status
log_status(lap, 0.0, waypoint_index)

while supervisor.step(timestep) != -1:
    # Get simulation time
    current_time = supervisor.getTime()
    
    # Get robot position
    position = robot_node.getField("translation").getSFVec3f()
    
    # Check if robot is at current waypoint (simple proximity check, can be improved)
    if ((position[0] - waypoints[waypoint_index][0])**2 + 
        (position[1] - waypoints[waypoint_index][1])**2 + 
        (position[2] - waypoints[waypoint_index][2])**2) < 0.01:  # Threshold distance for waypoint
        log_status(lap, current_time, waypoint_index + 1)
        waypoint_index += 1
        
        # Check if all waypoints are reached to complete the lap
        if waypoint_index >= len(waypoints):
            waypoint_index = 0
            at_initial_position = True  # Enable check for initial position

    # Check if robot is close to the initial position to complete the lap
    if at_initial_position and ((position[0] - initial_position[0])**2 + 
                                (position[1] - initial_position[1])**2 + 
                                (position[2] - initial_position[2])**2) < 0.01:  # Threshold distance for start line
        lap += 1
        at_initial_position = False  # Reset the check
        log_status(lap, current_time, 0)  # Log lap completion and reset waypoint to 0
