# Python Modules:
import numpy as np
from numpy import matlib
from scipy import sparse
from scipy.sparse import csc_matrix # Column sparse
from scipy.sparse import lil_matrix # Row-based list of lists sparse matrix 
from tqdm import tqdm # progress bar for loops
from scipy.linalg import convolution_matrix
from scipy.interpolate import interp1d
from scipy import signal
import gc

###############################################################################
def createForwMat(Ns,Nt,dx,nx,dsa,arco,ls,nls,DIS,MDIS,vs,to,tf,AS,SF): # 
    """Creating Forward Model-based Matrix
    """    
    normA = False; thresh = False; rsnoise = True; tdo = True
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIS,MDIS,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def createForwMatdotdet(Ns,Nt,dx,nx,dsa,arco,vs,to,tf): # 
    """Creating Forward Model-based Matrix for point sensors
    """    
    ls = 1e-3; nls = 1; DIS = False; MDIS = 1; AS = True; SF = False
    normA = True; thresh = False; rsnoise = True; tdo = True
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIS,MDIS,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def createForwMatdotdetLBW(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,DIR): # 
    """Creating Forward Model-based Matrix for point sensors with limited bandwidth
    """    
    ls = 1e-3; nls = 1; AS = True; SF = False
    normA = False; thresh = False; rsnoise = True; tdo = True
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    # Detector impulse response (limited bandwidth)
    if DIR:
        MDIR = CreateFilterMatrix(Nt,to,tf)
        MDIR = MDIR.astype(np.float32)
    else:
        MDIR = 1
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIR,MDIR,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p): #    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

###############################################################################
def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int64") // array_shape[1])
    cols = (ind.astype("int64") % array_shape[1])
    return (rows, cols)

###############################################################################
def createrectgrid(nx,ny,dx,dy,N,zo):
    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    y, x = ind2sub([nx, ny], np.linspace(0, N - 1,N)) # nornalized coordinate pixel position [x]=(N,); [y]=(N,) 
    rj = np.array([(x - originX) * dx, (y - originY) * dy, zo*np.ones((len(x)))]) # pixel position [rj]=(3,N))
    rj = rj.astype(np.float32)
    return rj

###############################################################################
def SensorMaskCartCircleArc(circle_radius, circle_arc, num_sensor_points):
    """
    Matrix with the Ns locations (num_sensor_points) of the sensors arranged 
    on a circunference arc (circle_arc) with a radius circle_radius
    """
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:(len(th) - 1)]  # Angles
    posSens = np.array([circle_radius * np.cos(th), circle_radius * np.sin(th), np.zeros((len(th)))])  # position of the center of the sensors
    posSens = posSens.astype(np.float32)
    return posSens # (3,Ns)

###############################################################################
def CreateFilterMatrix(Nt,to,tf):
    
    t = np.linspace(to, tf, Nt) # (Nt,)
    
    # FILTER DESIGN
    Fs = 1/(t[1]-t[0]) # Sampling frequency of the original time grid
    filOrd = 4 # filter order
    
    # Low-pass Filter 
    #flc = 20e6 # cutoff frequency [Hz]
    #bl, al = signal.butter(filOrd, flc, 'low', fs = Fs)
    
    # Band-pass Filter design BW% = 1.7 and we are assuming a cutoff freq of 0.1 MHz
    fbc1 = 1.5e6#1.25e6 # cutoff frequency [Hz]
    fbc2 = 3.1e6 # cutoff frequency [Hz]
    bb, ab = signal.butter(filOrd, (fbc1, fbc2), 'bandpass', fs = Fs)
    
    # FILTER IMPULSE RESPONSE
    po=np.zeros(Nt,)
    po[int(Nt/2)]=1
    #impL = signal.filtfilt(bl,al,po)
    impP = signal.filtfilt(bb,ab,po)
    
    # GENERATE FILTER MATRIX
    #FL=convolution_matrix(impL,Nt,'same')
    FP = convolution_matrix(impP,Nt,'same')
    FP = FP.astype(np.float32)
    return FP#,FL

###############################################################################
def build_matrix(nx,dx,Ns,posSens,ls,nls,DIR,MDIR,angsens,SF,vs,tt,normA,thresh,rsnoise,tdo,tlp=0):
    """
    Model-based Matrix by Spatial Impulse Response Approach -> A: (Ns*Nt,N)
    if tdo == False
        VP = A@P0 # where VP: velocity potencial (Ns*Nt,); P0: initial pressure (N,)
    else:
        P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    Ns: number of detectors
    posSens: position of the center of the detectors (3,Ns) [m]
    ls: size of the integrating detector [m], length for linear shape and diameter for disc shape
    nls: number of elements of the divided sensor (discretization)
    DIR: if true, measured detector impulse response is used. 
    MDIR: impulse response matrix (Nt, Nt)
    angsens: if True, the surface elements are sensitive to the angle of the incoming wavefront
    SH: if True, the detector shape (disc) is taking into account.
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    normA: normalize A? True or False
    thresh: threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    rsnoise: reduce shot noise and add laser pulse duration effect? True or False
    tdo: apply time derivative operator? True or False
    tlp: laser pulse duration [s], by default is set to zero
    
    References:
        [1] G. Paltauf, et al., "Modeling PA imaging with scanning focused detector using
        Monte Carlo simulation of energy deposition", J. Bio. Opt. 23, p. 121607 (2018).
        [2] G. Paltauf, et al., "Iterative reconstruction algorithm for OA imaging",
        J. Acoust. Soc. Am. 112, p. 1536 (2002).
    """    
    # Important constants
    Betta = 207e-6  # Thermal expansion coefficient for water at 20 °C [1/K].
    Calp = 4184     # Specific heat capacity at constant pressure for water at 20 °C [J/K Kg].
    rho = 1000      # Density for water or soft tissue [kg/m^3].
    #h0 = 1e4        # h0=mua*FLa; Energy absorbed in the sample per unit volumen (typical value) [J/m^3].
    #p0=100;         # Initial pressure (typical value) [Pa].
    #phi0 = 5e-9     # phi0=-Dt*p0/rho; Initial velocity potential assuming p0=100Pa and Dt=50ns [m^2/s].

    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    dy = dx  # pixel size in the y direction
    dz = dx  # pixel size in the z direction TODO: 3-D images
    DVol = dx * dy * dz  # element volume    
    rj = createrectgrid(nx,ny,dx,dy,N,0) # Coordinates of the pixels [3,N]
    
    # SENSOR DISCRETIZATION (for integrating line detectors in the z direction)
    posSensLin = posSens  # posSensLin[surface elements, xyz, Ns] position of the surface elements (treated as point detectors) of the divided "i" sensor
    if nls > 2:
        posz = np.linspace(-ls / 2, ls / 2, nls) # position of the surface elements (treated as point detectors) of the divided sensor
        posz = np.reshape(posz, (nls, 1))  
        posSensLin = np.reshape(posSens, (1, 3, Ns))
        aux = posSensLin
        for k in range(1, len(posz)):
            posSensLin = np.vstack((posSensLin, aux))
        posSensLin[0:, 2, 0:] = np.matlib.repmat(posz, 1, Ns)
    else:
        posz = 0
        posz = np.array([posz])
        posSensLin = np.reshape(posSens, (1, 3, Ns))
        posSensLin[0:, 2, 0:] = np.matlib.repmat(posz, 1, Ns)

    # TIME GRID
    dt = tt[1] - tt[0] # time step at which velocity potencials are sampled
    to = int(tt[0] / dt)
    tf = int(tt[len(tt) - 1] / dt) + 1
    sampleTimes = np.arange(to, tf, dtype=int)
    Nt = len(sampleTimes)
    
    # SPATIAL IMPULSE RESPONSE (SIR) MATRIX
    print('Creating SIR Matrix...'); # describes the spreading of a delta pulse over the sensor surface
    #Gs = np.zeros((Ns*Nt, N),dtype='float32') # [1/m]
    #Gs = csc_matrix(Gs,dtype='float32')
    Gs = lil_matrix((Ns*Nt, N),dtype='float32')
    currentSens = 0  # Current sensor
    currentTime = 0  # Current time
    for i1 in tqdm(range(Gs.shape[0])):  # Python processing by rows is faster and more efficient
        acum = np.zeros((1, N),dtype='float32') # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        for kk in range(0, nls):  # For each surface element of the divided sensor
            # Calculate the distance between DVol and posSensLin(:,:,i) 
            aux = np.reshape(posSensLin[kk, 0:, currentSens], (3, 1)) @ np.ones((1, N)) # [3,N]
            aux2 = rj - aux # [3,N]
            R = np.sqrt(aux2[0, 0:] ** 2 + aux2[1, 0:] ** 2 + aux2[2, 0:] ** 2) # [N,]
            # Weight factor: shape factor and angle sensitivity 
            wf = np.ones((1,N),dtype='float32') # [1,N]
            if angsens: # angle sensitivity
                nS = -1*posSens[:,currentSens] # [3,]
                mnS = np.sqrt(nS[0] ** 2 + nS[1] ** 2 + nS[2] ** 2) # norm l2
                nS = nS/mnS # detector surface normal
                nS = np.reshape(nS,(3,1)) @ np.ones((1, N),dtype='float32') # [3,N] 
                wf = np.abs(np.sum(nS*aux2,axis=0)/R) # cos(tita) = (nS dot (rd-rj))/(|nS|*|rd-rj|) [N,]
                wf = np.reshape(wf, (1, N)) # [1,N]
            if SF: # shape factor
                # Disc shape:
                dsf = 1
                wf = wf * (1 + dsf*np.abs(6.28*aux[2,0])/ls) # far from the center, means a larger perimeter (more surface elements for a certain z coordinate)
                #wf = wf * (1 + np.abs(6.28*aux[2,0]/2)/dzs/10)
            R = np.reshape(R, (1, N)) # [1,N]
            # All non-zero elements in a row A belong to DVol whose centers lie within a spherical shell of radius vs*tk and width vs*dt around posSens:
            # delta = 1   si    |t_k - R/vs| < dt/2
            #         0         en otro caso
            delta = (np.abs(sampleTimes[currentTime] * dt - R / vs) <= dt / 2) 
            delta = delta * 1  # Bool to int
            acum = acum + wf * delta / R  # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        Gs[i1, 0:] = acum
        currentTime = currentTime + 1  
        if np.mod(i1+1, Nt) == 0: # Calculate for other sensor
            currentSens = currentSens + 1
            currentTime = 0
    
    # LASER PULSE DURATION EFFECT AND SHOT NOISE REDUCTION
    if rsnoise: # Reduce shot noise induced by the arrival of wave from individual volume elements
        print('Reducing shot noise effect...');
        tprop = (dx+dy+dz)/(3*vs) # Average sound propagation time through a volume element
    
        if tprop<tlp:
            tprop = tlp # if tp > tprop, accurate modeling of Gs requires the use of a broader Gaussian function convolution
        
        Ti = np.arange(-np.ceil(Nt/2),np.ceil(Nt/2),dtype=int)*dt
        Gi = 2/(tprop*np.sqrt(np.pi))*np.exp(-1*((2*Ti/tprop)**2))
        Gi = convolution_matrix(Gi,Nt,'same')
        Gi = sparse.kron(csc_matrix(np.eye(Ns,dtype='float32')),csc_matrix(Gi,dtype='float32'))
        Gs = Gi@Gs  
        del Ti,Gi
        gc.collect()
    
    # SYSTEM MATRIZ
    if tdo:
        print('Creating PA Matrix...'); # describes the specific temporal signal from a PA point source
        print('Applying Time Derivative Operator...');
        #Tm=toeplitz(np.arange(0,Nt),np.arange(0,-Nt,-1));
        Tm=np.arange(-np.ceil(Nt/2),np.ceil(Nt/2),dtype=int)
        Tm2=(np.abs(Tm)<=(dx/(vs*dt)))*1; 
        Tm2=Tm2*Tm
        Tm2=Tm2*(-vs*dt/(2*dx));
        Gpa=convolution_matrix(Tm2,Nt,'same') # GPa is adimensional
        Gpa=sparse.kron(csc_matrix(np.eye(Ns,dtype='float32')),csc_matrix(Gpa,dtype='float32'))
        del Tm, Tm2
        gc.collect()
        A = Gpa@Gs
        A = (Betta/(4*np.pi*vs**2)*DVol/dt**2)*A;  # A is adimensional
        del Gpa, Gs
        gc.collect()
    else:
        A = Gs
        A = (-Betta / (4 * np.pi * rho * Calp) * DVol / dt) * A  # [m^5/(J*s)]    
        del Gs
        gc.collect()
    if DIR: # Detector Impulse Response
        print('Applying detector impulse response...');
        A = sparse.kron(np.eye(Ns,dtype='float32'),MDIR)@csc_matrix(A,dtype='float32')
    if normA: # System Matrix Normalization
        print('Normalization...')
        A = A / A.max() 
    
    if thresh>0: # Threshold the matrix to remove small entries and make it more sparse
        print('Removing small entries...')
        A = lil_matrix(A)
        A[abs(A)<10**(-thresh)] = 0
        A = lil_matrix(A)
    
    return A

###############################################################################
def DAS(nx,dx,dsa,posSens,vs,t,p):
    """
    Traditional Reconstruction Method "Delay and Sum" for 2-D OAT
    The output P0 is the initial pressure [P0] = (N,) where N is the total pixels
    of the image region.
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    p: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    
    References:
        [1] X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming 
        Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019).
    """  
    
    # GET Ns, Nt and N
    Ns=p.shape[0]
    Nt=p.shape[1]
    N = nx**2; # Total pixels 
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  
    dy = dx  # pixel size in the y direction 

    originX = np.ceil(nx / 2) # Set image region origin in the x direction
    originY = np.ceil(ny / 2) # Set image region origin in the y direction
    y, x = ind2sub([nx, ny], np.linspace(0, N - 1,N)) # nornalized coordinate pixel position [x]=(N,); [y]=(N,) 
    rj=np.array([(x-originX)*dx,(y-originY)*dy]) # pixel position [rj]=(2,N)
    rj=np.transpose(rj) # (N,2)
    Rj=np.reshape(rj,(1,N*2)) # (1,2*N)
    Rj=np.repeat(Rj,Ns,axis=0) # (Ns,2*N)
    Rj=np.reshape(Rj,(Ns*N*2,1)) # (2*N*Ns,1)
        
    # DETECTOR POSITIONS
    rs=posSens[0:2,:] # (2,Ns)
    rs=np.transpose(rs) # (Ns,2)
    Rs=np.repeat(rs,N,axis=0) # (N*Ns,2)
    Rs=np.reshape(Rs,(Ns*N*2,1)) # (2*N*Ns,1)
 
    # TIME GRID
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # Get norm 2 by row
    Tau=np.reshape(Tau,(Ns,N))
    
    # OBTAIN 2-D IMAGE
    P0=np.zeros((N,))
    t = np.reshape(t,(1,Nt))
    #for i in tqdm(range(0,Ns)):
    for i in range(0,Ns):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        P0=P0+aux
    
    #P0=np.reshape(P0,(nx,nx)) 
    return P0 
