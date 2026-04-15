import numpy as np

# ----------------------
# 常數
pi = 3.14159
G  = 6.674e-8       # cm^3 g^-1 s^-2
Ms = 1.989e33       # g
au = 1.496e13       # cm
yr = 3.156e7        # s

# ----------------------
# 參數
M1 = 1.5*Ms
M2 = 1.0*Ms
a  = 6.0*au / (1.0 + M2/M1) 
e  = 0.5
b  = a * np.sqrt(1-e**2)

n  = np.sqrt(G*(M1+M2)/(a*(1+M2/M1))**3)
P  = 2*pi/n
print("P =", P/yr, "yr")

dt = 0.01*yr
xt1 = -(a - a*e)
yt1 = 0.0

tau = P/2.0

ji = 0
je = int(93*yr/dt)
print("ji, je =", ji, je)

# 開檔
f_orbit  = open("orbit-M24.dat", "w")
f_orbit1 = open("orbit1-M24.dat", "w")
f_sep    = open("separation-M24.dat", "w")
f_vrel    = open("vrel-M24.dat", "w")


t0 = 0.0

for j in range(ji, je+1):
    t = j*dt
    M = n*(t - tau)
    Ei = M
    # Solve Kepler's equation iteratively
    for i in range(10):
        Ei = Ei - (Ei - e*np.sin(Ei) - M)/(1 - e*np.cos(Ei))

    r    = a*(1 - e*np.cos(Ei))
    cosf = (np.cos(Ei) - e)/(1 - e*np.cos(Ei))
    sinf = np.sqrt((1+e)/(1-e))*np.tan(Ei/2)*(1+cosf)

    x = -r*cosf
    y = -r*sinf

    vx = (x - xt1)/dt
    vy = (y - yt1)/dt
    v  = np.sqrt(vx**2 + vy**2)
    if t == ji*dt:
        v = 0.0

    # orbit.dat
    f_orbit.write("{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}\n".format(
        t/yr, x/au, y/au, r/au, v/1e5, abs(x/au)*(1+M2/M1), r*(1+M2/M1)/au
    ))

    # orbit1.dat
    f_orbit1.write("{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}\n".format(
        t/yr, (-x*M2/M1)/au, (-y*M2/M1)/au, (r*M2/M1)/au, (v*M2/M1)/1e5
    ))

    # separation.dat
    f_sep.write("{:10.3f} {:10.6f}\n".format(t/yr, np.sqrt((x*x+y*y)*((M1+M2)/M1)**2)/au))

    # vrel.dat
    f_vrel.write("{:10.3f} {:10.6f}\n".format(t/yr, v*((M1+M2)/M1)/1e5))

    # 更新上一步位置
    xt1 = x
    yt1 = y

    if x < 0.0 and y < 0.0:
        t0 = t

f_orbit.close()
f_orbit1.close()
f_sep.close()
f_vrel.close()

print("t0 =", (t0+dt)/yr)

