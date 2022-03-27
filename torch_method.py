import torch
import cv2
import numpy as np
import os
from liegroups.numpy import SE3


PATH = os.path.dirname(__file__)
os.chdir(PATH)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")#训练设备

flag = 1

camera_para = np.load(f'camera_mtx{flag}.npz')
mtx=camera_para['mtx']
dist=camera_para['dist']

c_mtx=mtx.copy()
marker_mtx=np.load('marker_mtx.npz')
r_mtx=marker_mtx['r_mtx']
t_mtx=marker_mtx['t_mtx']
d=0.0108


fx=mtx[0][0]
fy=mtx[1][1]
cx=mtx[0][2]
cy=mtx[1][2]


def calc_loss(r,t,corner,c_mtx,d):
    '''计算四个角点的欧氏距离损失'''

    torch.set_default_tensor_type(torch.DoubleTensor)
    c_mtx = torch.from_numpy(c_mtx).to(device)


    theta = torch.norm(r)
    r_norm = r/theta

    r_x = r_norm[0][0]
    r_y = r_norm[0][1]
    r_z = r_norm[0][2]

    temp = torch.tensor([[0 , -r_z , r_y],
                        [r_z , 0 ,-r_x],
                        [-r_y , r_x ,0]]).to(device)
    

    I = torch.eye(3).to(device)

    R = torch.cos(theta)*I+(1-torch.cos(theta))*torch.mm(r_norm.T,r_norm)+torch.sin(theta)*temp

    #R1,_ = cv2.Rodrigues(r.cpu().detach().numpy())

    R_T = torch.cat([R,t.T],dim=1)

    corner1 = torch.tensor([[-d/2,d/2,0,1]]).to(device)
    corner2 = torch.tensor([[d/2,d/2,0,1]]).to(device)
    corner3 = torch.tensor([[d/2,-d/2,0,1]]).to(device)
    corner4 = torch.tensor([[-d/2,-d/2,0,1]]).to(device)

    c_corner1 = torch.mm(R_T,corner1.T)
    c_corner2 = torch.mm(R_T,corner2.T)
    c_corner3 = torch.mm(R_T,corner3.T)
    c_corner4 = torch.mm(R_T,corner4.T)

    u1 = torch.mm(c_mtx,c_corner1)
    u2 = torch.mm(c_mtx,c_corner2)
    u3 = torch.mm(c_mtx,c_corner3)
    u4 = torch.mm(c_mtx,c_corner4)

    u1_norm = u1/u1[2][0]
    u2_norm = u2/u2[2][0]
    u3_norm = u3/u3[2][0]
    u4_norm = u4/u4[2][0]

    correct_u1 = torch.cat([torch.from_numpy(corner[0][0].reshape(1,2)),torch.tensor([[1]])],dim=1).reshape(3,1).to(device)
    correct_u2 = torch.cat([torch.from_numpy(corner[0][1].reshape(1,2)),torch.tensor([[1]])],dim=1).reshape(3,1).to(device)
    correct_u3 = torch.cat([torch.from_numpy(corner[0][2].reshape(1,2)),torch.tensor([[1]])],dim=1).reshape(3,1).to(device)
    correct_u4 = torch.cat([torch.from_numpy(corner[0][3].reshape(1,2)),torch.tensor([[1]])],dim=1).reshape(3,1).to(device)


    loss = torch.norm(correct_u1 - u1_norm)**2+torch.norm(correct_u2 - u2_norm)**2+torch.norm(correct_u3 - u3_norm)**2+torch.norm(correct_u4 - u4_norm)**2
    loss = loss/4.0

    return loss


def calibrate_pose(rvec,tvec,corner,c_mtx,d,r_lr,t_lr,min_loss,iters=100):
    '''矫正初始姿态，提高精度'''
    current_r = torch.from_numpy(rvec).to(device)
    current_t = torch.from_numpy(tvec).to(device)
    current_r.requires_grad=True
    current_t.requires_grad=True



    for i in range(iters):

        if (i+1)%50==0:
            r_lr = r_lr/10
            t_lr = t_lr/10

        loss = calc_loss(current_r,current_t,corner,c_mtx,d)
        loss.backward(retain_graph=True)
        
        if i ==0 : print('loss:  ',loss)
 
        dr = current_r.grad
        dt = current_t.grad

        current_r.requires_grad = False
        current_t.requires_grad = False

        current_r = current_r - r_lr *dr
        current_t = current_t - t_lr *dt

        current_r.requires_grad = True
        current_t.requires_grad = True

        if loss.item()<min_loss :break

    print('loss:  ',loss)
    return current_r.cpu().detach().numpy(),current_t.cpu().detach().numpy()




def calibrate_poses(rvecs,tvecs,corners,c_mtx,d,r_lr,t_lr,min_loss,iters=100):
    '''矫正初始姿态，提高精度'''
    n=np.shape(rvecs)[0]

    for i in range(n):
        rvecs[i] , tvecs[i] = calibrate_pose(rvecs[i],tvecs[i],corners[i],c_mtx,d,r_lr,t_lr,min_loss,iters)

    
    return rvecs , tvecs



def GN(corners, rvec,tvec,d,iters=7,show_loss=False):
    n=np.shape(rvec)[0]
    X=np.array([[[-d/2],[d/2],[0],[1]],[[d/2],[d/2],[0],[1]],[[d/2],[-d/2],[0],[1]],[[-d/2],[-d/2],[0],[1]]])
    rvec2=np.zeros((n,1,3))
    tvec2=np.zeros((n,1,3))
    for j in range(n):

        R,_=cv2.Rodrigues(rvec[j])
        T_esti=SE3.from_matrix(np.vstack([np.hstack([R,tvec[j].T]),np.array([[0,0,0,1]])]))
        cost=0.
        # print(T_esti.as_matrix())
        for iter in range(iters):
            cost=0.
            J=np.zeros((2,6))
            H=np.zeros((6,6))
            b=np.zeros((6,1))
            for i in range(4):
                vector3d = np.dot(T_esti.as_matrix(),X[i])
                x=vector3d[0][0]
                y=vector3d[1][0]
                z=vector3d[2][0]
                p_=np.array([[fx * ( x/z ) + cx],[fy * ( y/z ) + cy]])
                e=np.array([[corners[j][0][i][0]-p_[0][0]],[corners[j][0][i][1]-p_[1][0]]])
                cost+=e[0][0]*e[0][0]+e[1][0]*e[1][0]
                J[0,0] = -(fx/z)
                J[0,1] = 0
                J[0,2] = (fx*x/(z*z))
                J[0,3] = (fx*x*y/(z*z))
                J[0,4] = -(fx*x*x/(z*z)+fx)
                J[0,5] = (fx*y/z)
                J[1,0] = 0
                J[1,1] = -(fy/z)
                J[1,2] = (fy*y/(z*z))
                J[1,3] = (fy*y*y/(z*z)+fy)
                J[1,4] = -(fy*x*y/(z*z))
                J[1,5] = -(fy*x/z)
            
                H+=np.dot(J.T,J)
                b-=np.dot(J.T,e)

            if iter==0 and show_loss:
                print(f"init loss:  {cost/4}")
            if cost>500:
                return rvec,tvec
            dx=0.5*np.dot(np.linalg.inv(H),b)
            T_esti=SE3.exp(dx.T[0]).dot(T_esti)
            #if cost/4<1: break
        tmp,_=cv2.Rodrigues(T_esti.as_matrix()[0:3,0:3])
        rvec2[j]=tmp.T
        tvec2[j]=T_esti.as_matrix()[0:3,3:4].T
        if show_loss:print(f"final loss:  {cost/4}")
        
    return rvec2,tvec2



def calc_error(r,t,corner,c_mtx,ID,d):
    '''计算四个角点的欧氏距离损失'''

    torch.set_default_tensor_type(torch.DoubleTensor)
    c_mtx = torch.from_numpy(c_mtx).to(device)
    theta = torch.norm(r)
    r_norm = r/theta

    r_x = r_norm[0][0]
    r_y = r_norm[0][1]
    r_z = r_norm[0][2]

    temp = torch.tensor([[0 , -r_z , r_y],
                        [r_z , 0 ,-r_x],
                        [-r_y , r_x ,0]]).to(device)
    

    I = torch.eye(3).to(device)

    R = torch.cos(theta)*I+(1-torch.cos(theta))*torch.mm(r_norm.T,r_norm)+torch.sin(theta)*temp

    #R1,_ = cv2.Rodrigues(r.cpu().detach().numpy())
    R=torch.mm(R,torch.linalg.inv(torch.from_numpy(r_mtx[ID]) ))

    R_T = torch.cat([R,t.T-torch.mm(R,torch.from_numpy(t_mtx[ID].T))],dim=1)

    corner1 = torch.tensor([[-d/2,d/2,0,1]]).to(device)
    corner2 = torch.tensor([[d/2,d/2,0,1]]).to(device)
    corner3 = torch.tensor([[d/2,-d/2,0,1]]).to(device)
    corner4 = torch.tensor([[-d/2,-d/2,0,1]]).to(device)

    c_corner1 = torch.mm(R_T,corner1.T)
    c_corner2 = torch.mm(R_T,corner2.T)
    c_corner3 = torch.mm(R_T,corner3.T)
    c_corner4 = torch.mm(R_T,corner4.T)

    u1 = torch.mm(c_mtx,c_corner1)
    u2 = torch.mm(c_mtx,c_corner2)
    u3 = torch.mm(c_mtx,c_corner3)
    u4 = torch.mm(c_mtx,c_corner4)

    u1_norm = u1/u1[2][0]
    u2_norm = u2/u2[2][0]
    u3_norm = u3/u3[2][0]
    u4_norm = u4/u4[2][0]



    
    return [corner[0][0][0]-u1_norm[0][0],corner[0][0][1]-u1_norm[1][0],
            corner[1][0][0]-u2_norm[0][0],corner[1][0][1]-u2_norm[1][0],
            corner[2][0][0]-u3_norm[0][0],corner[2][0][1]-u3_norm[1][0],
            corner[3][0][0]-u4_norm[0][0],corner[3][0][1]-u4_norm[1][0]]
    

def GN3(rvec,tvec,ids,corners,iter=3):
    n=np.shape(corners)[0]
    current_r = torch.from_numpy(rvec).to(device)
    current_t = torch.from_numpy(tvec).to(device)
    
    corners=torch.from_numpy(corners).to(device)
    for i in range(iter):
        E=[]
        J=[]
        H=np.zeros((6,6))
        b=np.zeros((6,1))
        current_r.requires_grad_(True)
        current_t.requires_grad_(True)
        current_r.retain_grad()
        current_t.retain_grad()
        for j in range(n):
            
            e=calc_error(current_r,current_t,corners[j],c_mtx,ids[j],d)
            E.append([tmp.detach().numpy() for tmp in e])
            for k in range(8):
                e[k].backward(retain_graph=True)
                #print(current_r.grad)
                J.append(np.array([current_r.grad.numpy(),current_t.grad.numpy()]).reshape([1,6]))

                current_r.grad.zero_()
                current_t.grad.zero_()

        E=np.array(E).reshape(1,-1)
        J=np.array(J).reshape(-1,6)
        print("iter {}: error={}".format(i,np.linalg.norm(E)))
        for j in range(n*4):
            tmp=J[2*j:2*j+2,]
            H+=np.dot(tmp.T,tmp)
            b-=np.dot(tmp.T,E[:,2*j:2*j+2].T)
        dx=0.01*np.dot(np.linalg.inv(H),b)
        current_r=current_r+torch.from_numpy(dx[0:3,0].reshape(1,3))
        current_t=current_t+torch.from_numpy(dx[3:6,0].reshape(1,3))

    return current_r.detach().numpy(),current_t.detach().numpy()

def GN_L(obj_points,corners, rvec,tvec,iters=7,show_loss=False):
    n=np.shape(obj_points)[0]
    rvec2=np.zeros((1,3))
    tvec2=np.zeros((1,3))
    R,_=cv2.Rodrigues(rvec)

    T_esti=SE3.from_matrix(np.vstack([np.hstack([R,tvec.T]),np.array([[0,0,0,1]])]))
    cost=0.
    # print(T_esti.as_matrix())
    for iter in range(iters):
        cost=0.
        J=np.zeros((2,6))
        H=np.zeros((6,6))
        b=np.zeros((6,1))
        for i in range(n):
            vector3d = np.dot(T_esti.as_matrix(),np.append(obj_points[i],1).reshape(4,1))
            x=vector3d[0][0]
            y=vector3d[1][0]
            z=vector3d[2][0]
            p_=np.array([[fx * ( x/z ) + cx],[fy * ( y/z ) + cy]])
            e=np.array([[corners[i][0][0]-p_[0][0]],[corners[i][0][1]-p_[1][0]]])
            cost+=e[0][0]*e[0][0]+e[1][0]*e[1][0]
            J[0,0] = -(fx/z)
            J[0,1] = 0
            J[0,2] = (fx*x/(z*z))
            J[0,3] = (fx*x*y/(z*z))
            J[0,4] = -(fx*x*x/(z*z)+fx)
            J[0,5] = (fx*y/z)
            J[1,0] = 0
            J[1,1] = -(fy/z)
            J[1,2] = (fy*y/(z*z))
            J[1,3] = (fy*y*y/(z*z)+fy)
            J[1,4] = -(fy*x*y/(z*z))
            J[1,5] = -(fy*x/z)
        
            H+=np.dot(J.T,J)
            b-=np.dot(J.T,e)

        if iter==0 and show_loss:
            print(f"init loss:  {cost/4}")
        if cost>500:
            return rvec,tvec
        dx=0.5*np.dot(np.linalg.inv(H),b)
        T_esti=SE3.exp(dx.T[0]).dot(T_esti)
            #if cost/4<1: break
        tmp,_=cv2.Rodrigues(T_esti.as_matrix()[0:3,0:3])
        rvec2=tmp.T
        tvec2=T_esti.as_matrix()[0:3,3:4].T
        if show_loss:print(f"final loss:  {cost/4}")
        
    return rvec2,tvec2
