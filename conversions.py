import numpy as np

# q = x,y,z,w
# return [roll,pitch,yaw]
def toEulerAngles(q):
    sinr = 2.0 * (q[3] * q[0] + q[1] * q[2])
    cosr = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1] )
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (q[3] * q[1]  - q[2] * q[0] )

    if(np.abs(sinp) >= 1):
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny = 2.0 * (q[3] * q[0] + q[0] * q[1] )
    cosy = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2] )
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])

def posesFromQuaternionToRPY(poses):
    poses_xyzrpy = []
    for i in range(0,len(poses)):
        pose = np.zeros(6)
        pose[0:3] = poses[i,0:3]
        pose[3:6] = toEulerAngles(poses[i,3:7])
        poses_xyzrpy.append(pose)

    return np.array(poses_xyzrpy)
