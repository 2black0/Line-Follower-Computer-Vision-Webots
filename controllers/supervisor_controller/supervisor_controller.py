from controller import Supervisor
import time

# Inisialisasi supervisor
supervisor = Supervisor()

# Waktu langkah dalam milidetik
timestep = int(supervisor.getBasicTimeStep())

# Dapatkan waktu simulasi awal
start_time = supervisor.getTime()

# Fungsi untuk menambahkan robot e-puck
def add_robot():
    # Dapatkan root node
    root = supervisor.getRoot()
    children_field = root.getField("children")
    
    # Definisikan urdf atau proto file robot e-puck yang akan ditambahkan
    robot_def = 'DEF ROBOT E-puck {translation 0 0 0}'
    
    # Tambahkan robot
    children_field.importMFNodeFromString(-1, robot_def)
    print("Robot added.")

# Fungsi untuk menghapus robot e-puck
def remove_robot():
    # Dapatkan robot node
    robot = supervisor.getFromDef("ROBOT")
    if robot is not None:
        robot.remove()
        print("Robot removed.")

# Main loop
while supervisor.step(timestep) != -1:
    # Dapatkan waktu simulasi saat ini
    current_time = supervisor.getTime()
    
    # Periksa apakah 30 detik telah berlalu
    if current_time - start_time >= 30:
        # Hapus robot
        remove_robot()
        
        # Tunggu beberapa detik (misalnya 2 detik) sebelum menambahkan robot kembali
        time.sleep(2)
        
        # Tambahkan robot kembali
        add_robot()
        
        # Reset waktu mulai
        start_time = current_time
