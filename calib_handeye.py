import numpy as np
import open3d as o3d
import os
import pathlib
from Pose_Estimation_Class import Batch_Processing,EPS
from helpers import Tools
file_path='./data/Cam2DCali.txt'

# Using readlines()
file1 = open(file_path, 'r')
Lines = file1.readlines()
   
robot_str='Transform Matrix Robot->TCP:'
cam_str='Transform Matrix Chessboard->Cam:'
handeye_str= 'HandEye Matrix (Camera in Robot TCP Coordinate):'
robot_count=0
cam_count=0
def str_2_array(s):
    s=s.translate({ord(i): None for i in '()\n'})
    s=s.split(',')
    return s

# Strips the newline character
robot_poses=[]
cam_poses=[]
B_seq=[]
handeye=np.identity(4)
for i,line in enumerate(Lines): 
    if robot_str in line:  
        pose_value=str_2_array(Lines[i+1]) 
        pose_value.extend(str_2_array(Lines[i+2]))
        pose_value.extend(str_2_array(Lines[i+3]))
        pose_value.extend(str_2_array(Lines[i+4]))
        pose_value=np.asarray(pose_value,dtype='float32').reshape(4,4)
        robot_poses.append(pose_value)
        print(len(robot_poses),pose_value)
    if cam_str in line:  
        pose_value=str_2_array(Lines[i+1]) 
        pose_value.extend(str_2_array(Lines[i+2]))
        pose_value.extend(str_2_array(Lines[i+3]))
        pose_value.extend(str_2_array(Lines[i+4]))
        pose_value=np.asarray(pose_value,dtype='float32').reshape(4,4)
        cam_poses.append(pose_value)
        B_seq.append(np.linalg.inv(pose_value))
        print(len(cam_poses),pose_value)
    if handeye_str in line: 
        pose_value=str_2_array(Lines[i+1]) 
        pose_value.extend(str_2_array(Lines[i+2]))
        pose_value.extend(str_2_array(Lines[i+3]))
        pose_value.extend(str_2_array(Lines[i+4]))
        pose_value=np.asarray(pose_value,dtype='float32').reshape(4,4)
        handeye=pose_value
        print("Handeye=",handeye)

#Display
render_items=[]
render_items.append(o3d.geometry.TriangleMesh.create_coordinate_frame( size=100, origin=[0,0,0]))
#HandEyE Calib 
A_seq = np.stack(robot_poses,axis=2) 
B_seq = np.stack(B_seq,axis=2) 
print(A_seq.shape)
print(B_seq.shape)
#Batch Processing
X_est,Y_est,Y_est_check,ErrorStats=Batch_Processing.pose_estimation(A_seq,B_seq)
print('\n')
print('.....Batch Processing Results')
print('HandEye=\n')
print(X_est)
print('Target Location=\n')
print(Y_est)
handeye=X_est


for i in range(len(robot_poses)):
    robot_pose=robot_poses[i]
    cam_pose=cam_poses[i]
    
    robot_frame=o3d.geometry.TriangleMesh.create_coordinate_frame( size=10, origin=[0,0,0])
    robot_frame.transform(robot_pose)
    cam_frame=o3d.geometry.TriangleMesh.create_coordinate_frame( size=30, origin=[0,0,0])
    cam_frame.transform(np.matmul(robot_pose,handeye)) 

    target_frame=o3d.geometry.TriangleMesh.create_coordinate_frame( size=50, origin=[0,0,0])
    target_frame.transform(np.matmul(np.matmul(robot_pose,handeye),cam_pose)) 

    render_items.extend([robot_frame,cam_frame,target_frame]) 
     
o3d.visualization.draw_geometries(render_items)
