# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def f(r,t):
    xJ = r[0]
    yJ = r[1]
    zJ = r[2]
    vxJ = r[3]
    vyJ = r[4]
    vzJ = r[5]
    xA = r[6]
    yA = r[7]
    zA = r[8]
    vxA = r[9]
    vyA = r[10]
    vzA = r[11]
    xS = r[12]
    yS = r[13]
    zS = r[14]
    vxS = r[15]
    vyS = r[16]
    vzS = r[17]
    
    axJ = -(xJ-xS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)
    ayJ = -(yJ-yS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)
    azJ = -(zJ-zS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)

    axA = -(xA-xS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3.0/2) - q *(xA-xJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3.0/2)
    ayA = -(yA-yS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3.0/2) - q *(yA-yJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3.0/2)
    azA = -(zA-zS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3.0/2) - q *(zA-zJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3.0/2)

    axS = q*(xJ-xS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)
    ayS = q*(yJ-yS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)
    azS = q*(zJ-zS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3.0/2)
    
    return np.array([vxJ,vyJ,vzJ,axJ,ayJ,azJ,vxA,vyA,vzA,axA,ayA,azA,vxS,vyS,vzS,axS,ayS,azS],float)
    
  
def AdaptiveRK4(start, end, totalaccuracy, r):
    
    def timestep(t0,r0,h,step_count):
        if t0 + 2*h > end:
            h = (end-t0)/2
        
        k1 = h*f(r0,t0)
        k2 = h*f(r0+0.5*k1,t0+0.5*h)
        k3 = h*f(r0+0.5*k2,t0+0.5*h)
        k4 = h*f(r0+k3,t0+h)
        r1 = r0 + (k1+2*k2+2*k3+k4)/6
        t1 = t0 + h
        
        k1 = h*f(r1,t1)
        k2 = h*f(r1+0.5*k1,t1+0.5*h)
        k3 = h*f(r1+0.5*k2,t1+0.5*h)
        k4 = h*f(r1+k3,t1+h)
        r1 = r1 + (k1+2*k2+2*k3+k4)/6
       
        k1 = 2*h*f(r0,t0)
        k2 = 2*h*f(r0+0.5*k1,t0+0.5*h*2)
        k3 = 2*h*f(r0+0.5*k2,t0+0.5*h*2)
        k4 = 2*h*f(r0+k3,t0+h*2)
        r2 = r0 + (k1+2*k2+2*k3+k4)/6
        
        error2J = ((r1[0]-r2[0])/30)**2 + ((r1[1]-r2[1])/30)**2 + ((r1[2]-r2[2])/30)**2
        error2A = ((r1[6]-r2[6])/30)**2 + ((r1[7]-r2[7])/30)**2 + ((r1[8]-r2[8])/30)**2
        error2S = ((r1[12]-r2[12])/30)**2 + ((r1[13]-r2[13])/30)**2 + ((r1[14]-r2[14])/30)**2
        error = np.sqrt(error2J+error2A+error2S)
        rho = h * precision / error
        
        if rho >= 1:
            r0 = r1 + (r1-r2)/15
            t0 = t0 + 2*h
            step_count = step_count+1
            tpoints.append(t0)
            xJpoints.append(r0[0])
            yJpoints.append(r0[1])
            zJpoints.append(r0[2])
            xApoints.append(r0[6])
            yApoints.append(r0[7])
            zApoints.append(r0[8])
            xSpoints.append(r0[12])
            ySpoints.append(r0[13])
            zSpoints.append(r0[14])
            
            JA = np.sqrt((r0[0]-r0[6])**2+(r0[1]-r0[7])**2)
            SA = np.sqrt((r0[12]-r0[6])**2+(r0[13]-r0[7])**2)
            SJ = np.sqrt((r0[0]-r0[12])**2+(r0[1]-r0[13])**2)
            angle = np.arccos((SJ**2+SA**2-JA**2)/(2*SA*SJ))
            anglepoints.append(angle)
            radiuspoints.append(SA-SJ)
      
#            tanJ = r0[1]/r0[0]
#            tanA = r0[7]/r0[6]
#            tanJA = (tanJ-tanA)/(1.+tanJ*tanA)
#            anglepoints.append(np.arctan(tanJA)-np.pi/3)
#            radiusJA = np.sqrt(r0[6]**2+r0[7]**2)-np.sqrt(r0[0]**2+r0[1]**2)
#            Epoints.append(0.5*(r0[3]**2+r0[4]**2+r0[5]**2)-1.0/np.sqrt(r0[0]**2+r0[1]**2+r0[2]**2))
#            EApoints.append(0.5*(r0[9]**2+r0[10]**2+r0[11]**2)-1.0/np.sqrt(r0[6]**2+r0[7]**2+r0[8]**2)-1.0/np.sqrt((r0[6]-r0[0])**2+(r0[7]-r0[1])**2+(r0[8]-r0[2])**2)*GMJ/GM)

            if rho**(1/4)<2:
                h = h * rho**(1/4)
            else:
                h = h * 2
        else:
            h = h * rho**(1/4)
        return t0, r0, h, step_count
        
    
    precision = totalaccuracy/(end-start)
    t0 = start
    r0 = r.copy()
    h = float(end-start)/ 10000
    step_count = 0   
    
    tpoints = []
    xJpoints = []
    yJpoints = []
    zJpoints = []
    xApoints = []
    yApoints = []
    zApoints = []
    xSpoints = []
    ySpoints = []
    zSpoints = []
    anglepoints = []
    radiuspoints = []

    tpoints.append(t0)
    xJpoints.append(r0[0])
    yJpoints.append(r0[1])
    zJpoints.append(r0[2])
    xApoints.append(r0[6])
    yApoints.append(r0[7])
    zApoints.append(r0[8])
    xSpoints.append(r0[12])
    ySpoints.append(r0[13])
    zSpoints.append(r0[14])
    
    JA = np.sqrt((r0[0]-r0[6])**2+(r0[1]-r0[7])**2)
    SA = np.sqrt((r0[12]-r0[6])**2+(r0[13]-r0[7])**2)
    SJ = np.sqrt((r0[0]-r0[12])**2+(r0[1]-r0[13])**2)
    angle = np.arccos((SJ**2+SA**2-JA**2)/(2*SA*SJ))-np.pi/3
    anglepoints.append(angle)
    radiuspoints.append(SA-SJ)
    
    while t0 < end:
        t0, r0, h, step_count = timestep(t0 ,r0, h, step_count)
    return np.array(tpoints), np.array(xJpoints), np.array(yJpoints), np.array(zJpoints), \
    np.array(xApoints), np.array(yApoints), np.array(zApoints), \
    np.array(xSpoints), np.array(ySpoints), np.array(zSpoints), \
    np.array(anglepoints), np.array(radiuspoints), step_count
    
        
# Astronomical Data
    
GMS = 1.327124400189*10**20
GMJ = 1.266865349*10**17
q = GMJ/GMS
a = 778.299*10**9 # semi-major axis as unit of length
e = 0.048498  # eccentricity

time_unit_in_second = np.sqrt(a**3/GMS)
Julian_year_in_second = 365.25 * 86400
time_unit_in_years = time_unit_in_second / Julian_year_in_second
period_in_years = 2*np.pi*time_unit_in_years*np.sqrt(GMS/(GMS+GMJ))
t_final = 100.0 / time_unit_in_years



cosalpha = np.sqrt(1+q)/2-q/np.sqrt(1+q)
sinalpha = np.sqrt(1 - cosalpha**2)
c = (1+e)/np.sqrt(1+q)
vc = np.sqrt((1-e)/(1+e)/(1+q))

r_ini = np.array([1./(1.+q)*(1.+e), 0., 0.,  0., 1./(1.+q)*np.sqrt((1-e)/(1+e)), 0.,
                  c*cosalpha, -c*sinalpha, 0., vc*sinalpha, vc*cosalpha, 0.,
                -q/(1.+q)*(1.+e), 0., 0., 0., -q/(1.+q)*np.sqrt((1-e)/(1+e)), 0.])


# calculate exact solution
# need to reformulate
'''
mean_anomaly = t_final - np.pi
eccentric_anomaly = 0.0   # initial guess of eccentric anomaly at final time
delta = 1.0
accuracy = 1e-12
while abs(delta)>accuracy:
    delta = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / (1-e*np.cos(eccentric_anomaly))
    eccentric_anomaly -= delta
x_final_exact = -1 * np.cos(eccentric_anomaly) + e
y_final_exact = -1 * np.sin(eccentric_anomaly) * np.sqrt(1-e**2)
'''

# calculation

tpointsARK4, xJpointsARK4, yJpointsARK4, zJpointsARK4, \
xApointsARK4, yApointsARK4, zApointsARK4, \
xSpointsARK4, ySpointsARK4, zSpointsARK4, \
anglepointsARK4, radiuspointsARK4, step_count = AdaptiveRK4(0, t_final, 10**(-6), r_ini)


# need to reformulate
#error_final = np.sqrt((x_final_exact - xpointsARK4[-1])**2 + (y_final_exact - ypointsARK4[-1])**2 )


# plot of orbits
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(xJpointsARK4, yJpointsARK4, label='Jupiter')
ax1.plot(xApointsARK4, yApointsARK4, label='Asteroid')
ax1.plot(xSpointsARK4, ySpointsARK4, label='Sun')
ax1.set_xlabel(r'$x/a$')
ax1.set_ylabel(r'$y/a$')
plt.title('orbits, Adaptive RK4, Step Count='+str(step_count))
plt.legend(loc=0)
plt.show()

# plot of trajectory of asteroid relative to Lagrangian point L5
fig2 = plt.figure()
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(anglepointsARK4, radiuspointsARK4, label='Step Count='+str(step_count))
ax1.set_xlabel(r'$angle/radian$')
ax1.set_ylabel(r'$radius/a$')
plt.title('trajectory of Trojan asteroid relative to Lagrangian points L5')
plt.legend(loc=0)
plt.show()


'''
# energy plot
fig3 = plt.figure()
ax1 = fig3.add_subplot(1,1,1)
ax1.plot(tpointsARK4, EpointsARK4, label='Energy of Jupiter')
ax1.plot(tpointsARK4, EApointsARK4, label='Energy of Asteroid')
ax1.set_xlabel(r'$t/ \sqrt{\frac{a^3}{GM}}$')
ax1.set_ylabel(r'$E/ \frac{GMm}{a}$')
ax1.set_ylim(-1, 0)
plt.title('plot of energy against time')
plt.legend(loc=0)
plt.show()
'''