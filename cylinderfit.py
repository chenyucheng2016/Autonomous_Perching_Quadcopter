#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.neighbors import NearestNeighbors
#from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from vision_proj.msg import Pts
from multiprocessing import Process, Queue
import time
def callback(msg):
    start = time.time()
    data = np.zeros((3,msg.size))
    for k in range(msg.size):
         data[0,k] = msg.X[k]
         data[1,k] = msg.Z[k]
         data[2,k] = -msg.Y[k]
    print "Data recieved. Processing Now..."
    """
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')    
    x1 = np.squeeze(np.asarray(data[0]))
    y1 = np.squeeze(np.asarray(data[1]))
    z1 = np.squeeze(np.asarray(data[2]))
    ax.scatter(x1,y1,z1)
    plt.show()
    """
    normals = pts2norm(data.T)
    normV,vn1,vn2 = plane_ransac(normals,250,0.01,0.05)
    mat = np.matrix([vn1,vn2,normV])
    pcl = proj2plane(data,mat)
    qs = []
    tasks = []
    nproc = 5
    for i in range(nproc):
       q = Queue()
       p = Process(target = circle_ransac,args=(pcl,20,0.01,0.05,q))
       qs.append(q)
       tasks.append(p)
    for task in tasks:
       task.start()
    for task in tasks:
       task.join()
    radius = -1
    size = -1
    ctr = [0,0,0]
    for q in qs:
       val = q.get()
       if val[1] > size:
          radius = val[0]
          ctr = val[2]
          size = val[1]
           
    end = time.time()
    print 'radius ',radius
    print 'center ',ctr
    print 'orientation ',normV 
    print 'distance ', msg.dist
    print 'Time elapsed ', end - start
    print ''
#The function uses RANSAC 
def circle_ransac(pts,iterNum,thd,thr,q):
    ptNum = pts.shape[1]
    thInlr = round(thr*ptNum)
    optsz = -1
    radius = -1
    ctr = [0,0,0]
    inlier = np.array([])
    for i in range(iterNum):
        dist = np.zeros((1,ptNum))
        perm = np.random.permutation(ptNum)
        sampleIdx = perm[0:3]
        ptSample = pts[:,sampleIdx]
        p1 = np.squeeze(np.asarray(ptSample[:,0]))
        p2 = np.squeeze(np.asarray(ptSample[:,1]))
        p3 = np.squeeze(np.asarray(ptSample[:,2]))
        if iscollinear(p1,p2,p3):
           print 'collinear'
           continue
        center,r,v1n,v2n = circlefit3d(p1,p2,p3)
        un = np.cross(v1n,v2n)
        un = un/np.sqrt(np.dot(un,un))
        
        for k in range(ptNum):
            vector = np.squeeze(np.asarray(pts[:,k])) 
            proj = np.cross(un,(vector-center.T))
            dist[:,k] = np.sqrt(np.dot(proj,proj))
        inlridx = np.nonzero(abs(dist-r)<thd)
        inlrsz = np.size(inlridx)
        if inlrsz < thInlr:
           continue
        if (inlrsz > optsz):
           optsz = inlrsz
           radius = r
           ctr = center
           inlier = pts[:,inlridx]
    q.put([radius,optsz,ctr])

def proj2plane(pts,mat):
    sol = inv(mat)*pts
    sol[2,:] = 0
    return mat*sol
def circlefit3d(p1,p2,p3):
    v1 = p2 - p1
    v2 = p3 - p1
    l1 = np.sqrt(np.dot(v1,v1))
    l2 = np.sqrt(np.dot(v2,v2))
    v1n = v1/l1
    v2n = v2/l2
    nv = np.cross(v1n,v2n)
    v2nb = np.cross(v1n,nv)
    v2n = v2nb/np.sqrt(np.dot(v2nb,v2nb))
    cor1 = np.dot(v2,v1n)
    cor2 = np.dot(v2,v2n)
    scale1 = 0.5*l1
    scale2 = 0.5*(cor2 - (l1-cor1)*cor1/cor2)
    center = p1+scale1*v1n+scale2*v2n
    rv = p1-center
    r = np.sqrt(np.dot(rv,rv))
    return (center,r,v1n,v2n) 
def iscollinear(p1,p2,p3):
    v1 = p2-p1
    uv = v1/np.sqrt(np.dot(v1,v1))
    t = (p3[0]-p1[0])/uv[0]
    pt = p1+t*uv
    erv = p3-pt
    err = np.sqrt(np.dot(erv,erv))
    if err < 0.00001:
       return True
    else:
       return False
def pts2norm(pts):
    nbrs = NearestNeighbors(n_neighbors = 100, algorithm = 'ball_tree').fit(pts)
    dist,indices = nbrs.kneighbors(pts)
    m = pts.shape[0]
    n = np.zeros((3,m))
    for i in range(m):
        kn = pts[indices[i,0:100],:]
        #print kn
        kn_col = np.sum(kn,axis=0)/100.0
        #print kn_col
        P1 = (kn-np.tile(kn_col,(100,1))).T
        P2 = (kn-np.tile(kn_col,(100,1)))
        P = np.dot(P1,P2)
        #print P
        D,V = la.eig(P)
        ind = np.argmin(D)
        n[:,i] = V[:,ind]
    return n
def plane_ransac(pts,iterNum,thd,thr):
    ptNum = pts.shape[1]
    thInlr = round(thr*ptNum)
    inlrsz = np.zeros((1,iterNum))
    uns = np.zeros((3,iterNum))
    v1s = np.zeros((3,iterNum))
    v3s = np.zeros((3,iterNum))
    for i in range(iterNum):
        perm = np.random.permutation(ptNum)
        sampleIdx = perm[0:2]
        ptSample = pts[:,sampleIdx]
        p1 = ptSample[:,0]
        p2 = ptSample[:,1]
        p3 = -p1
        if iscollinear(p1,p2,p3):
           print 'collinear'
           continue
        un,v1,v3 = fitplane(p1,p2,p3)
        arrange = np.tile(p1,(ptNum,1)).T
        dist = np.dot(un,(pts-arrange))
        inlieridx = np.nonzero(abs(dist)<thd)
        if np.size(inlieridx) < thInlr:
           continue
        else:
           inlrsz[:,i] = np.size(inlieridx)
           uns[:,i] = un
           v1s[:,i] = v1
           v3s[:,i] = v3
    candid = np.argmax(inlrsz)
    normV = uns[:,candid]
    vn1 = v1s[:,candid]
    vn3 = v3s[:,candid]
    return (normV,vn1,vn3)
def fitplane(p1,p2,p3):
    v1 = p2-p1
    v1 = v1/np.sqrt(np.dot(v1,v1))
    v2 = p3-p1
    v2 = v2/np.sqrt(np.dot(v2,v2))
    v3 = np.cross(v1,v2)
    v3 = v3/np.sqrt(np.dot(v3,v3))
    return (v3,v1,v2)

def main():
   rospy.init_node('CylinderFitter',anonymous = True)
   rospy.Subscriber('chatter',Pts,callback)
   rospy.spin()


if __name__ == "__main__":
   main()









