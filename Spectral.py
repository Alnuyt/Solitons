"""
Simulation de la Propagation des Solitons et Analyse des Erreurs Numériques

Auteur : Alexandre Nuyt et Axel Guerlus

Ce script simule la propagation de solitons à l'aide de la méthode de Fourier pour la résolution numérique des équations aux dérivées partielles.

### Fonctionnalités principales :
- Définition des paramètres physiques du système (constantes, longueur d'onde, pas de temps, etc.).
- Implémentation de la transformation de Fourier pour résoudre numériquement l'évolution des solitons.
- Comparaison entre solutions analytiques et numériques.
- Visualisation des résultats sous forme de graphiques et d'animations.
- Évaluation de l'erreur entre solutions analytiques et numériques pour différentes valeurs de dt.

### Applications :
Ce programme permet d'étudier la stabilité des schémas numériques appliqués à la propagation de solitons, un phénomène crucial en physique non linéaire et en optique.

"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#définition paramètres
c_1 = 0.75
a_1 = 0.33
c_2 = 0.4
a_2 = 0.65
L = 50
t_max = 50
dt = 0.0004
N_x = 256
N_t = int(t_max/dt)
x = np.linspace(0, L, N_x)
t = np.linspace(0, t_max, N_t)
k = np.linspace(-(N_x/2), (N_x/2)-1, N_x)

#matrice dont une colonne represente la "solution" analytique à un pas de temps
sol = np.zeros((N_x, N_t))

#matrice dont une colonne represente la solution numérique à un pas de temps
u_n = np.zeros((N_x, N_t))

#u_n_0 est la condition initiale
u_n_0 = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 *L)))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L)))**(-2)
u_k = np.zeros((N_x, N_t), dtype=complex)
g_n = np.zeros((N_x, N_t))
g_k = np.zeros((N_x, N_t),dtype=complex)
u_n[:,0] = u_n_0

for i in range(N_t-1):
    sol[:,i] = (c_1/2) * (np.cosh((np.sqrt(c_1)/ 2) * (x - a_1 * L - c_1 * t[i])))**(-2) + (c_2/ 2) * (np.cosh((np.sqrt(c_2)/ 2) * (x - a_2 * L - c_2 * t[i])))**(-2)
    u_k[:,i] = (np.fft.fftshift(np.fft.fft(u_n[:,i])))
    g_k[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt) * u_k[:,i]
    g_n[:,i] = (np.fft.ifft(np.fft.ifftshift(g_k[:,i])))
    g_n_squared = g_n[:,i]**2
    dg2dx = np.fft.ifft(np.fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (np.fft.fftshift(np.fft.fft(g_n_squared))))))
    u_n[:,i+1] = np.real(g_n[:,i]-3 * dg2dx *dt)

#figure 1 de l'énoncé
c = np.linspace(np.min(u_n), np.max(u_n), 101)
xx,tt = np.meshgrid(x, t)
cs = plt.contourf(xx, tt, u_n.T, c, figsize = (10, 10))
plt.axis('scaled')
plt.colorbar()
plt.show()

#graphe (animation) hauteurs pour 2 solitons (numerique+analytique)
fig, ax = plt.subplots()
line1, = ax.plot(x, u_n[:,0])
line2, = ax.plot(x, sol[:,0])
   
def init():
        line1.set_data(x, u_n[:,0])
        line2.set_data(x, sol[:,0])
        return line1,line2
    
def animate(frame):
        y_1 = u_n[:,frame]
        y_2 = sol[:,frame]
        line1.set_data((x, y_1))
        line2.set_data((x,y_2))
        return line1, line2

anim = FuncAnimation(fig, animate, frames = np.arange(0, N_t, 280), init_func = init, interval = 0.3)
plt.axis([0, 50, 0, 0.6])
plt.xticks(np.arange(0, 50, 5))
plt.yticks((np.arange(0, 0.6, 0.06)))
plt.title("Comparaison evolution hauteur vagues", fontsize = 25)
plt.xlabel("Distance (m)", fontsize = 20)
plt.ylabel("Hauteur (m)", fontsize = 20)
plt.legend()
plt.legend(["numerique", "analytique"], bbox_to_anchor = (1, 1), ncol = 1, fontsize = 18, markerscale = 10)
ax.grid(True, 'both','both')

plt.show()

#matrice des ecarts entre numerique et analytique
delta_1 = sol - u_n
# erreur globale entre les 2 méthodes
error_1 = np.zeros((1, N_t))
for i in range(N_t - 1):
    error_1[0, i] = np.linalg.norm(delta_1[:,i])

plt.plot(t, error_1[0, :])
plt.xlim(0,15)
plt.ylim(0,0.015)
plt.title("Erreur (analytique/numérique)", fontsize = 25)
plt.xlabel("Temps (s)", fontsize = 20)
plt.ylabel("Erreur (m)", fontsize = 20)

plt.show()

#comparaison hauteur pour 1 soliton (numerique et analytique), pour differents dt
c = 0.75
a = 0.33
L = 50
t_max = 50
dt_2 = 0.0004
N_x = 256
N_t_2 = int(t_max/dt_2)
x = np.linspace(0, L, N_x)
t_2 = np.linspace(0, t_max, N_t_2)
k = np.linspace(-(N_x/2), (N_x/2)-1, N_x)
C_I = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L)))**(-2) 
sol_1 = np.zeros((N_x, N_t_2))
sol_2 = np.zeros((N_x, N_t_2))
u_k = np.zeros((N_x, N_t_2), dtype=complex)
g_n = np.zeros((N_x, N_t_2))
g_k = np.zeros((N_x, N_t_2),dtype=complex)
sol_1[:,0] = C_I
for i in range(N_t-1):
    sol_2[:,i] = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L - c * t_2[i])))**(-2) 
    u_k[:,i] = (np.fft.fftshift(np.fft.fft(sol_1[:,i])))
    g_k[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt_2) * u_k[:,i]
    g_n[:,i] = (np.fft.ifft(np.fft.ifftshift(g_k[:,i])))
    g_n_squared = g_n[:,i]**2
    dg2dx = np.fft.ifft(np.fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (np.fft.fftshift(np.fft.fft(g_n_squared))))))
    sol_1[:,i+1] = np.real(g_n[:,i]-3 * dg2dx *dt_2)


c = 0.75
a = 0.33
L = 50
t_max = 50
dt_3 = 0.0002
N_x = 256
N_t_3 = int(t_max/dt_3)
x = np.linspace(0, L, N_x)
t_3 = np.linspace(0, t_max, N_t_3)
k = np.linspace(-(N_x/2), (N_x/2)-1, N_x)
C_I = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L)))**(-2) 
sol_3 = np.zeros((N_x, N_t_3))
sol_4 = np.zeros((N_x, N_t_3))
u_k = np.zeros((N_x, N_t_3), dtype=complex)
g_n = np.zeros((N_x, N_t_3))
g_k = np.zeros((N_x, N_t_3),dtype=complex)
sol_3[:,0] = C_I
for i in range(N_t_3-1):
    sol_4[:,i] = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L - c * t_3[i])))**(-2) 
    u_k[:,i] = (np.fft.fftshift(np.fft.fft(sol_3[:,i])))
    g_k[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt_3) * u_k[:,i]
    g_n[:,i] = (np.fft.ifft(np.fft.ifftshift(g_k[:,i])))
    g_n_squared = g_n[:,i]**2
    dg2dx = np.fft.ifft(np.fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (np.fft.fftshift(np.fft.fft(g_n_squared))))))
    sol_3[:,i+1] = np.real(g_n[:,i]-3 * dg2dx *dt_3)

c = 0.75
a = 0.33
L = 50
t_max = 50
dt_4 = 0.0008
N_x = 256
N_t_4 = int(t_max/dt_4)
x = np.linspace(0, L, N_x)
t_4 = np.linspace(0, t_max, N_t_4)
k = np.linspace(-(N_x/2), (N_x/2)-1, N_x)
C_I = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L)))**(-2) 
sol_5 = np.zeros((N_x, N_t_4))
sol_6 = np.zeros((N_x, N_t_4))
u_k = np.zeros((N_x, N_t_4), dtype=complex)
g_n = np.zeros((N_x, N_t_4))
g_k = np.zeros((N_x, N_t_4),dtype=complex)
sol_5[:,0] = C_I
for i in range(N_t_4-1):
    sol_6[:,i] = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L - c * t_4[i])))**(-2) 
    u_k[:,i] = (np.fft.fftshift(np.fft.fft(sol_5[:,i])))
    g_k[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt_4) * u_k[:,i]
    g_n[:,i] = (np.fft.ifft(np.fft.ifftshift(g_k[:,i])))
    g_n_squared = g_n[:,i]**2
    dg2dx = np.fft.ifft(np.fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (np.fft.fftshift(np.fft.fft(g_n_squared))))))
    sol_5[:,i+1] = np.real(g_n[:,i]-3 * dg2dx *dt_4)    

c = 0.75
a = 0.33
L = 50
t_max = 50
dt_5 = 0.002
N_x = 256
N_t_5 = int(t_max/dt_5)
x = np.linspace(0, L, N_x)
t_5 = np.linspace(0, t_max, N_t_5)
k = np.linspace(-(N_x/2), (N_x/2)-1, N_x)
C_I = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L)))**(-2) 
sol_7 = np.zeros((N_x, N_t_5))
sol_8 = np.zeros((N_x, N_t_5))
u_k = np.zeros((N_x, N_t_5), dtype=complex)
g_n = np.zeros((N_x, N_t_5))
g_k = np.zeros((N_x, N_t_5),dtype=complex)
sol_7[:,0] = C_I
for i in range(N_t_5-1):
    sol_8[:,i] = (c/2) * (np.cosh((np.sqrt(c)/ 2) * (x - a * L - c * t_5[i])))**(-2) 
    u_k[:,i] = (np.fft.fftshift(np.fft.fft(sol_7[:,i])))
    g_k[:,i] = np.exp((1j * (((2 * np.pi)/ L) *k)**3)*dt_5) * u_k[:,i]
    g_n[:,i] = (np.fft.ifft(np.fft.ifftshift(g_k[:,i])))
    g_n_squared = g_n[:,i]**2
    dg2dx = np.fft.ifft(np.fft.ifftshift(((1j * ((2 *np.pi)/ L) * k) * (np.fft.fftshift(np.fft.fft(g_n_squared))))))
    sol_7[:,i+1] = np.real(g_n[:,i]-3 * dg2dx *dt_5)

delta_2 = sol_1 - sol_2
delta_3 = sol_3 - sol_4
delta_4 = sol_5 - sol_6
delta_5 = sol_7 - sol_8

error_2 = np.zeros((1, N_t_2))
error_3 = np.zeros((1, N_t_3))
error_4 = np.zeros((1, N_t_4))
error_5 = np.zeros((1, N_t_5))

for i in range(N_t_2 - 1):
    error_2[0, i] = np.linalg.norm(delta_2[:,i])
for i in range(N_t_3 - 1):   
    error_3[0, i] = np.linalg.norm(delta_3[:,i])
for i in range(N_t_4 - 1):    
    error_4[0, i] = np.linalg.norm(delta_4[:,i])
for i in range(N_t_5 - 1):   
    error_5[0, i] = np.linalg.norm(delta_5[:,i])

plt.plot(t_2, error_2[0, :])
plt.plot(t_3, error_3[0, :])
plt.plot(t_4, error_4[0, :])
plt.plot(t_5, error_5[0, :])
plt.xlim(0, 20)
plt.ylim(0, 0.1)
plt.legend(["dt = 0.0004", "dt = 0.0002", "dt = 0.0008", "dt = 0.002"], bbox_to_anchor = (1, 1), ncol = 1, fontsize = 18, markerscale = 10)
plt.title("Erreur analytique/numérique (1 soliton)", fontsize = 25)
plt.xlabel("Temps (s)", fontsize = 20)
plt.ylabel("Erreur (m)", fontsize = 20)

plt.show()

plt.plot(t_2, error_2[0, :])
plt.plot(t_3, error_3[0, :])
plt.plot(t_4, error_4[0, :])
plt.plot(t_5, error_5[0, :])
plt.xlim(40, 50)
plt.ylim()
plt.legend(["dt = 0.0004", "dt = 0.0002", "dt = 0.0008", "dt = 0.002"], bbox_to_anchor = (0.2, 1), ncol = 1, fontsize = 18, markerscale = 10)
plt.title("Erreur analytique/numérique (1 soliton)", fontsize = 25)
plt.xlabel("Temps (s)", fontsize = 20)
plt.ylabel("Erreur (m)", fontsize = 20)

plt.show()
