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
    
    axJ = -(xJ-xS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)
    ayJ = -(yJ-yS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)
    azJ = -(zJ-zS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)

    axA = -(xA-xS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3./2) - q *(xA-xJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3./2)
    ayA = -(yA-yS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3./2) - q *(yA-yJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3./2)
    azA = -(zA-zS)/((xA-xS)**2+(yA-yS)**2+(zA-zS)**2)**(3./2) - q *(zA-zJ)/((xA-xJ)**2+(yA-yJ)**2+(zA-zJ)**2)**(3./2)

    axS = q*(xJ-xS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)
    ayS = q*(yJ-yS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)
    azS = q*(zJ-zS)/((xJ-xS)**2+(yJ-yS)**2+(zJ-zS)**2)**(3./2)
    
    return np.array([vxJ,vyJ,vzJ,axJ,ayJ,azJ,vxA,vyA,vzA,axA,ayA,azA,vxS,vyS,vzS,axS,ayS,azS],float)

  
def AdaptiveRK4(start, end, totalaccuracy, r):
    
    def timestep(t0,r0,h,step_count):
        if t0 + 2*h > end:
            h = (end-t0)/2.
        
        k1 = h*f(r0,t0)
        k2 = h*f(r0+k1/2., t0+h/2.)
        k3 = h*f(r0+k2/2., t0+h/2.)
        k4 = h*f(r0+k3,t0+h)
        r1 = r0 + (k1+ 2*k2+ 2*k3+ k4)/6.
        t1 = t0 + h
        
        k1 = h*f(r1,t1)
        k2 = h*f(r1+k1/2., t1+h/2.)
        k3 = h*f(r1+k2/2., t1+h/2.)
        k4 = h*f(r1+k3, t1+h)
        r1 = r1 + (k1+ 2*k2+ 2*k3+ k4)/6.
       
        k1 = 2*h*f(r0,t0)
        k2 = 2*h*f(r0+k1/2.,t0+h)
        k3 = 2*h*f(r0+k2/2.,t0+h)
        k4 = 2*h*f(r0+k3,t0+h*2)
        r2 = r0 + (k1+ 2*k2+ 2*k3+ k4)/6.
        
        error2J = ((r1[0]-r2[0])/30)**2 + ((r1[1]-r2[1])/30)**2 + ((r1[2]-r2[2])/30)**2
        error2A = ((r1[6]-r2[6])/30)**2 + ((r1[7]-r2[7])/30)**2 + ((r1[8]-r2[8])/30)**2
        error2S = ((r1[12]-r2[12])/30)**2 + ((r1[13]-r2[13])/30)**2 + ((r1[14]-r2[14])/30)**2
        error = np.sqrt(error2J+error2A+error2S)
        rho = h * precision / error
        
        if rho > 1.:
            r0 = r1 + (r1-r2)/15
            t0 = t0 + 2*h
            step_count = step_count+1
            tpoints.append(t0)
            xJpoints.append(r0[0])
            yJpoints.append(r0[1])
            zJpoints.append(r0[2])
            vxJpoints.append(r0[3])
            vyJpoints.append(r0[4])
            vzJpoints.append(r0[5])
            xApoints.append(r0[6])
            yApoints.append(r0[7])
            zApoints.append(r0[8])
            xSpoints.append(r0[12])
            ySpoints.append(r0[13])
            zSpoints.append(r0[14])
            vxSpoints.append(r0[15])
            vySpoints.append(r0[16])
            vzSpoints.append(r0[17])            
            
            if rho**(1./4)<2.:
                h = h * rho**(1./4)
            else:
                h = h * 2.
        else:
            h = h * rho**(1./4)
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
    vxJpoints = []
    vyJpoints = []
    vzJpoints = []    
    xApoints = []
    yApoints = []
    zApoints = []
    xSpoints = []
    ySpoints = []
    zSpoints = []
    vxSpoints = []
    vySpoints = []
    vzSpoints = []
    
    tpoints.append(t0)
    xJpoints.append(r0[0])
    yJpoints.append(r0[1])
    zJpoints.append(r0[2])
    vxJpoints.append(r0[3])
    vyJpoints.append(r0[4])
    vzJpoints.append(r0[5])
    xApoints.append(r0[6])
    yApoints.append(r0[7])
    zApoints.append(r0[8])
    xSpoints.append(r0[12])
    ySpoints.append(r0[13])
    zSpoints.append(r0[14])
    vxSpoints.append(r0[15])
    vySpoints.append(r0[16])
    vzSpoints.append(r0[17])
            
    while t0 < end:
        t0, r0, h, step_count = timestep(t0 ,r0, h, step_count)
        
    return np.array(tpoints), \
    np.array(xJpoints), np.array(yJpoints), np.array(zJpoints), \
    np.array(vxJpoints), np.array(vyJpoints), np.array(vzJpoints), \
    np.array(xApoints), np.array(yApoints), np.array(zApoints), \
    np.array(xSpoints), np.array(ySpoints), np.array(zSpoints), \
    np.array(vxSpoints), np.array(ySpoints), np.array(vzSpoints), \
    step_count
    
        
# Astronomical Data
    
GMS = 1.327124400189*10**20
GMJ = 1.266865349*10**17
q = GMJ/GMS
e = 0.048498  # eccentricity
period = 2. *np.pi / np.sqrt(1+q)


# initial conditions

c = (1+e)*np.sqrt(1-q/(1.+q)+(q/(1+q))**2) # distance from Sun to Lagrangian point
vc = np.sqrt(1-q/(1+q)+(q/(1+q))**2) * np.sqrt((1-e)/(1+e)*(1+q))  # velocity of asteroid at Lagrangian point given perfect conditions
perturbation = 0.98
vcp = vc*perturbation  #perturbation to vc 
cosalpha = -(q/(1+q)-1./2)/np.sqrt(1-q/(1+q)+(q/(1+q))**2) # alpha is angle of Jupiter-Barycenter-Asteroid
sinalpha = np.sqrt(1 - cosalpha**2)

r_ini = np.array([0., 1/(1+q)*(1+e), 0.,  -1/(1+q)*np.sqrt((1-e)/(1+e)*(1+q)), 0., 0.,
                  c*sinalpha, c*cosalpha, 0,  -1*vcp*cosalpha, vcp*sinalpha, 0.,
                 0., -q/(1+q)*(1+e), 0.,   q/(1+q)*np.sqrt((1-e)/(1+e)*(1+q)), 0., 0.])



# calculation

t_final = 100 * period
totalaccuracy = 1E-8

tpointsARK4, \
xJpointsARK4, yJpointsARK4, zJpointsARK4, \
vxJpointsARK4, vyJpointsARK4, vzJpointsARK4, \
xApointsARK4, yApointsARK4, zApointsARK4, \
xSpointsARK4, ySpointsARK4, zSpointsARK4, \
vxSpointsARK4, vySpointsARK4, vzSpointsARK4, \
step_count = AdaptiveRK4(0, t_final, totalaccuracy, r_ini)


# more analysis

JS = np.sqrt((xJpointsARK4-xSpointsARK4)**2+(yJpointsARK4-ySpointsARK4)**2)
AS = np.sqrt((xApointsARK4-xSpointsARK4)**2+(yApointsARK4-ySpointsARK4)**2)
JA = np.sqrt((xJpointsARK4-xApointsARK4)**2+(yJpointsARK4-yApointsARK4)**2)
angleJSA = np.arctan2(yJpointsARK4-ySpointsARK4, xJpointsARK4-xSpointsARK4) - np.arctan2(yApointsARK4-ySpointsARK4, xApointsARK4-xSpointsARK4)
for i in range(angleJSA.shape[0]):
    if angleJSA[i]<0:
        angleJSA[i]=angleJSA[i]+2*np.pi
# angleJSA = np.arctan((tanJS-tanAS)/(1+tanJS*tanAS))   give angle wrong range!
# angleJSA = np.arccos((JS**2+AS**2-JA**2)/(2*AS*JS))   give angle wrong range!

energy = 0.5 * (vxJpointsARK4**2+vyJpointsARK4**2+vzJpointsARK4**2) \
+ 0.5 * (vxSpointsARK4**2+vySpointsARK4**2+vzSpointsARK4**2)/q \
- 1./JS


# calculate exact solution and check

mean_anomaly = t_final*np.sqrt(1.+q) - np.pi
eccentric_anomaly = 1.0   # initial guess of eccentric anomaly at final time
delta = 1.0 # initialize delta
accuracytarget = 1E-15
while abs(delta)>accuracytarget:
    delta = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / (1-e*np.cos(eccentric_anomaly))
    eccentric_anomaly -= delta
JS_exact = 1-e*np.cos(eccentric_anomaly)
JS_error = abs(JS_exact - JS[-1])

energy_error = abs(energy[-1]-energy[0])


print('step count=', step_count)
print('error in final JS distance=', JS_error)
print('error in final energy=', energy_error)

if JS_error <= totalaccuracy:
    print('Success!')
else:
    print('Fail!', JS_error)
    
print('calculation complete, now plotting...')


# plot and animation of orbits in stationary frame
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(xSpointsARK4, ySpointsARK4, label='Sun')
ax1.plot(xJpointsARK4, yJpointsARK4, label='Jupiter')
ax1.plot(xApointsARK4, yApointsARK4, label='Asteroid')
ax1.set_xlabel(r'$x/a$',fontsize=16)
ax1.set_ylabel(r'$y/a$',fontsize=16)
plt.axes().set_aspect('equal')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.title('orbits in stationary frame \n perturbation ratio = {0:.3g} \n total time = {1:.3g} Jupiter years'.format(perturbation, t_final/period),fontsize=16)
plt.legend(loc=0)
plt.tight_layout()
plt.show()


'''
totalnumberofframe = 600
for i in range(totalnumberofframe+1):
    t = t_final * i/totalnumberofframe
    j = np.argmin(np.fabs(tpointsARK4-t))
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,1,1)
    ax1.plot(xSpointsARK4[j],ySpointsARK4[j], 'o',label='Sun')
    ax1.plot(xJpointsARK4[j],yJpointsARK4[j], 'o',label='Jupiter')
    ax1.plot(xApointsARK4[j],yApointsARK4[j], 'o',label='Asteroid')
    ax1.set_xlabel(r'$x/a$',fontsize=16)
    ax1.set_ylabel(r'$y/a$',fontsize=16)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.axes().set_aspect('equal')
    plt.title('orbits in stationary frame \n perturbation ratio = {0:.3g} \n time = {1:.3g} Jupiter years'.format(perturbation, t/period), fontsize=16)
    plt.legend(loc=1)
    plt.tight_layout()    
    plt.savefig('plotst_{0:03d}.png'.format(i))
    plt.close(fig2)
'''

# plot and animation of orbits in frame of constant rotation speed

fig3 = plt.figure()
ax1 = fig3.add_subplot(1,1,1)
ax1.plot(np.cos(tpointsARK4*np.sqrt(1+q))* xSpointsARK4 + np.sin(tpointsARK4*np.sqrt(1+q))*ySpointsARK4, -1*np.sin(tpointsARK4*np.sqrt(1+q))* xSpointsARK4 + np.cos(tpointsARK4*np.sqrt(1+q))*ySpointsARK4, label='Sun')
ax1.plot(np.cos(tpointsARK4*np.sqrt(1+q))* xJpointsARK4 + np.sin(tpointsARK4*np.sqrt(1+q))*yJpointsARK4, -1*np.sin(tpointsARK4*np.sqrt(1+q))* xJpointsARK4 + np.cos(tpointsARK4*np.sqrt(1+q))*yJpointsARK4, label='Jupiter')
ax1.plot(np.cos(tpointsARK4*np.sqrt(1+q))* xApointsARK4 + np.sin(tpointsARK4*np.sqrt(1+q))*yApointsARK4, -1*np.sin(tpointsARK4*np.sqrt(1+q))* xApointsARK4 + np.cos(tpointsARK4*np.sqrt(1+q))*yApointsARK4, label='Asteroid')
ax1.set_xlabel(r'$x/a$',fontsize=16)
ax1.set_ylabel(r'$y/a$',fontsize=16)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.axes().set_aspect('equal')
plt.title('orbits in frame of constant rotation \n perturbation ratio = {0:.3g} \n total time = {1:.3g} Jupiter years'.format(perturbation, t_final/period),fontsize=16)
plt.legend(loc=0)    
plt.tight_layout()
plt.show()


'''
totalnumberofframe = 300
for i in range(totalnumberofframe+1):
    t = t_final * i/totalnumberofframe
    j = np.argmin(np.fabs(tpointsARK4-t))
    fig4 = plt.figure()
    ax1 = fig4.add_subplot(1,1,1)
    ax1.plot(np.cos(tpointsARK4[j]*np.sqrt(1+q))* xSpointsARK4[j] + np.sin(tpointsARK4[j]*np.sqrt(1+q))*ySpointsARK4[j], -1*np.sin(tpointsARK4[j]*np.sqrt(1+q))* xSpointsARK4[j] + np.cos(tpointsARK4[j]*np.sqrt(1+q))*ySpointsARK4[j], 'o',label='Sun')
    ax1.plot(np.cos(tpointsARK4[j]*np.sqrt(1+q))* xJpointsARK4[j] + np.sin(tpointsARK4[j]*np.sqrt(1+q))*yJpointsARK4[j], -1*np.sin(tpointsARK4[j]*np.sqrt(1+q))* xJpointsARK4[j] + np.cos(tpointsARK4[j]*np.sqrt(1+q))*yJpointsARK4[j], 'o',label='Jupiter')
    ax1.plot(np.cos(tpointsARK4[j]*np.sqrt(1+q))* xApointsARK4[j] + np.sin(tpointsARK4[j]*np.sqrt(1+q))*yApointsARK4[j], -1*np.sin(tpointsARK4[j]*np.sqrt(1+q))* xApointsARK4[j] + np.cos(tpointsARK4[j]*np.sqrt(1+q))*yApointsARK4[j], 'o',label='Asteroid')
    ax1.set_xlabel(r'$x/a$',fontsize=16)
    ax1.set_ylabel(r'$y/a$',fontsize=16)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.axes().set_aspect('equal')
    plt.title('orbits in frame of constant rotation \n perturbation ratio = {0:.3g} \n time = {1:.3g} Jupiter years'.format(perturbation, t/period), fontsize=16)
    plt.legend(loc=0)    
    plt.tight_layout()
    plt.savefig('plotcr_{0:03d}.png'.format(i))
    plt.close(fig4)
'''
 
  
# plot and animation of orbits in a special frame of relative positions
  
fig5 = plt.figure()
ax1 = fig5.add_subplot(1,1,1)
ax1.plot(0,0, '.', label='Sun')
ax1.plot(np.zeros(JS.shape),JS, label='Jupiter')
ax1.plot(AS*np.sin(angleJSA), AS*np.cos(angleJSA), label='Asteroid')
ax1.set_xlabel(r'$x/a$',fontsize=16)
ax1.set_ylabel(r'$y/a$',fontsize=16)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.axes().set_aspect('equal')
plt.title('orbits in special frame \n perturbation ratio = {0:.3g} \n total time = {1:.3g} Jupiter years'.format(perturbation, t_final/period),fontsize=16)
plt.legend(loc=0)    
plt.tight_layout()
plt.show()

'''
totalnumberofframe = 300
for i in range(totalnumberofframe+1):
    t = t_final * i/totalnumberofframe
    j = np.argmin(np.fabs(tpointsARK4-t))
    fig6 = plt.figure()
    ax1 = fig6.add_subplot(1,1,1)
    ax1.plot(0,0, 'o',label='Sun')
    ax1.plot(0,JS[j], 'o',label='Jupiter')
    ax1.plot(AS[j]*np.sin(angleJSA[j]), AS[j]*np.cos(angleJSA[j]), 'o',label='Asteroid')
    ax1.set_xlabel(r'$x/a$',fontsize=16)
    ax1.set_ylabel(r'$y/a$',fontsize=16)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.axes().set_aspect('equal')
    plt.title('orbits in special frame \n perturbation ratio = {0:.3g} \n time = {1:.3g} Jupiter years'.format(perturbation, t/period), fontsize=16)
    plt.legend(loc=0)    
    plt.tight_layout()
    plt.savefig('plotsp_{0:03d}.png'.format(i))
    plt.close(fig6)
'''

 
 
# plot of trajectory of asteroid relative to Lagrangian point L5
fig7 = plt.figure()
ax1 = fig7.add_subplot(1,1,1)
ax1.plot(angleJSA, AS-JS, label='Step Count='+str(step_count))
ax1.set_xticks([0., np.pi/3., 2./3*np.pi, np.pi, 4./3*np.pi, 5./3*np.pi, 2*np.pi])
ax1.set_xticklabels(["$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"])
ax1.set_xlabel(r'$\angle JSA$ /radian',fontsize=16)
ax1.set_ylabel(r'$(|\overrightarrow{AS}|-|\overrightarrow{JS}|)/a$',fontsize=16)
plt.ylim(-0.15, 0.15)
plt.title('trajectory of asteroid \n perturbation ratio = {0:.3g} \n total time = {1:.3g} Jupiter years'.format(perturbation, t_final/period),fontsize=16)
plt.legend(loc=0)
plt.tight_layout()
plt.show()


fig8 = plt.figure()
ax1 = fig8.add_subplot(1,1,1)
ax1.plot(angleJSA, AS-JS, label='Step Count='+str(step_count))
ax1.set_xlabel(r'$\angle JSA$ /radian',fontsize=16)
ax1.set_ylabel(r'$(|\overrightarrow{AS}|-|\overrightarrow{JS}|)/a$',fontsize=16)
plt.title('trajectory of asteroid \n perturbation ratio = {0:.3g} \n total time = {1:.3g} Jupiter years'.format(perturbation, t_final/period),fontsize=16)
plt.legend(loc=0)
plt.tight_layout()
plt.show()




'''
# convergence plot

t_final = 100 * period
step_count_list = []
totalaccuracy_list = []
JS_error_list = []
energy_error_list = []

for i in np.arange(4,9):
    totalaccuracy = 10**(-i)
    
    tpointsARK4, \
    xJpointsARK4, yJpointsARK4, zJpointsARK4, \
    vxJpointsARK4, vyJpointsARK4, vzJpointsARK4, \
    xApointsARK4, yApointsARK4, zApointsARK4, \
    xSpointsARK4, ySpointsARK4, zSpointsARK4, \
    vxSpointsARK4, vySpointsARK4, vzSpointsARK4, \
    step_count = AdaptiveRK4(0, t_final, totalaccuracy, r_ini)


    JS = np.sqrt((xJpointsARK4-xSpointsARK4)**2+(yJpointsARK4-ySpointsARK4)**2)

    mean_anomaly = t_final*np.sqrt(1.+q) - np.pi
    eccentric_anomaly = 1.0   # initial guess of eccentric anomaly at final time
    delta = 1.0 # initialize delta
    accuracytarget = 1E-15
    while abs(delta)>accuracytarget:
        delta = (eccentric_anomaly - e * np.sin(eccentric_anomaly) - mean_anomaly) / (1-e*np.cos(eccentric_anomaly))
        eccentric_anomaly -= delta
    JS_exact = 1-e*np.cos(eccentric_anomaly)
    JS_error = abs(JS_exact - JS[-1])


    energy = 0.5 * (vxJpointsARK4**2+vyJpointsARK4**2+vzJpointsARK4**2) \
    + 0.5 * (vxSpointsARK4**2+vySpointsARK4**2+vzSpointsARK4**2)/q \
    - 1./JS
    
    energy_error = abs(energy[-1]-energy[0])

    step_count_list.append(step_count)
    totalaccuracy_list.append(totalaccuracy)
    JS_error_list.append(JS_error)
    energy_error_list.append(energy_error)


m_ta, b_ta = np.polyfit(np.log(step_count_list), np.log(totalaccuracy_list), 1)
m_JS, b_JS = np.polyfit(np.log(step_count_list), np.log(JS_error_list), 1)
m_energy, b_energy = np.polyfit(np.log(step_count_list), np.log(energy_error_list), 1) 


fig9 = plt.figure()
ax1 = fig9.add_subplot(1,1,1)
ax1.plot(step_count_list, totalaccuracy_list, label='target accuracy')
ax1.plot(step_count_list, np.exp(b_ta)* step_count_list **(m_ta), label=r'fit line, slope={0:.3g}'.format(m_ta))
ax1.plot(step_count_list, JS_error_list, label=r'error in $|\overrightarrow{JS}|$')
ax1.plot(step_count_list, np.exp(b_JS)* step_count_list **(m_JS), label=r'fit line, slope={0:.3g}'.format(m_JS))
ax1.plot(step_count_list, energy_error_list, label=r'error in total energy')
ax1.plot(step_count_list, np.exp(b_energy)* step_count_list **(m_energy), label=r'fit line, slope={0:.3g}'.format(m_energy))
ax1.set_xlabel(r'step count',fontsize=16)
ax1.set_ylabel(r'$L_1$ error',fontsize=16) 
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.title('convergence plot \n total time = {0:.3g} Jupiter years'.format(t_final/period),fontsize=16)
plt.legend(loc=0)
plt.tight_layout()
plt.show()
'''