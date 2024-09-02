import numpy as np
import json
from dronehover.bodies.custom_bodies import Custombody
from dronehover.optimization import Hover

def load_drone(path):
    # Load drone
    file = "./drones/ctrl_drone.json"
    with open(file, 'r') as f:
        props = json.load(f)

    drone = Custombody(props)
    drone_properties = Hover(drone)
    drone_properties.compute_hover()
    
    M = drone.mass * np.eye(3)
    Ixx = drone.Ix
    Iyy = drone.Iy
    Izz = drone.Iz
    Ixy = drone.Ixy

    I = np.array([[Ixx, Ixy, 0  ],
                  [Ixy, Iyy, 0  ],
                  [0  , 0  , Izz]])
    
    Bf = drone_properties.Bf
    Bm = drone_properties.Bm
    eta_hat = drone_properties.eta

    return M, I, Bf, Bm, eta_hat