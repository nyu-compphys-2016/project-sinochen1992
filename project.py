# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def f(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    vx = r[3]
    vy = r[4]
    vz = r[5]    
    ax = -x/(x**2+y**2+z**2)**(3.0/2)
    ay = -y/(x**2+y**2+z**2)**(3.0/2)
    az = -z/(x**2+y**2+z**2)**(3.0/2)
    return np.array([vx,vy,vz,ax,ay,az],float)
    
  

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
        
        errorx = (r1[0]-r2[0])/30
        errory = (r1[1]-r2[1])/30
        errorz = (r1[2]-r2[2])/30
        error = np.sqrt(errorx**2+errory**2+errorz**2)
        rho = h * precision / error
        
        if rho >= 1:
            r0 = r1 + (r1-r2)/15
            t0 = t0 + 2*h
            step_count = step_count+1
            tpoints.append(t0)
            xpoints.append(r0[0])
            ypoints.append(r0[1])
            zpoints.append(r0[2])
            Epoints.append(0.5*(r0[3]**2+r0[4]**2+r0[5]**2)-1.0/np.sqrt(r0[0]**2+r0[1]**2+r0[2]**2))
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
    h = float(end-start)/ 1000
    step_count = 0   
    
    tpoints = []
    xpoints = []
    ypoints = []
    zpoints = []
    Epoints = []
    tpoints.append(t0)
    xpoints.append(r0[0])
    ypoints.append(r0[1])
    zpoints.append(r0[2])
    Epoints.append(0.5*(r0[3]**2+r0[4]**2+r0[5]**2)-1.0/np.sqrt(r0[0]**2+r0[1]**2+r0[2]**2))
    
    while t0 < end:
        t0, r0, h, step_count = timestep(t0 ,r0, h, step_count)
    return np.array(tpoints), np.array(xpoints), np.array(ypoints), np.array(zpoints),np.array(Epoints), step_count
    
        
# Mars
    
GM = 1.327124400189*10**20
a = 227.9392*10**9 # semi-major axis as unit of length
e = 0.0934  # eccentricity
time_unit_in_second = np.sqrt(a**3/GM)
Julian_year_in_second = 365.25 * 86400
time_unit_in_years = time_unit_in_second / Julian_year_in_second
period_in_years = 2*np.pi*time_unit_in_years
t_final = 5.0 / time_unit_in_years
r_ini = np.array([1.0+e,0.0,0.0,0.0,np.sqrt((1-e)/(1+e)),0.0])


# calculate exact solution

mean_anomaly = t_final - np.pi
eccentric_anomaly = 0.0   # initial guess of eccentric anomaly at final time
delta = 1.0
accuracy = 1e-12
while abs(delta)>accuracy:
    delta = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / (1-e*np.cos(eccentric_anomaly))
    eccentric_anomaly -= delta
x_final_exact = -1 * np.cos(eccentric_anomaly) + e
y_final_exact = -1 * np.sin(eccentric_anomaly) * np.sqrt(1-e**2)



tpointsARK4, xpointsARK4, ypointsARK4, zpointsARK4,EpointsARK4, step_count= AdaptiveRK4(0, t_final, 10**(-8), r_ini)
error_final = np.sqrt((x_final_exact - xpointsARK4[-1])**2 + (y_final_exact - ypointsARK4[-1])**2 )


# plot of orbits
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(xpointsARK4, ypointsARK4, label='Step Count='+str(step_count))
ax1.set_xlabel(r'$x/a$')
ax1.set_ylabel(r'$y/a$')
plt.title('orbit of Mars using Adaptive RK4')
plt.legend(loc=0)
plt.show()

# energy plot
fig2 = plt.figure()
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(tpointsARK4, EpointsARK4, label='Energy')
ax1.set_xlabel(r'$t/ \sqrt{\frac{a^3}{GM}}$')
ax1.set_ylabel(r'$E/ \frac{GMm}{a}$')
ax1.set_ylim(-1, 0)
plt.title('plot of energy against time')
plt.legend(loc=0)
plt.show()