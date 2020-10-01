from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gen_vec_perlin import VecPerlinNoise

Z_LENGTH = 200j
Z_INT = 200
start = datetime.now().timestamp()
noise_gen = VecPerlinNoise(dimensions=3, period=50)
x, y, z = np.mgrid[0:50:200j, 0:50:200j, 0:20:Z_LENGTH]
# x, y = np.mgrid[0:50:1000j, 0:50:1000j] 2-d
print(datetime.now().timestamp() - start)

start = datetime.now().timestamp()
noise = noise_gen.noise(x, y, z, octaves=4, base_freq=.2, persistence=.5)
print(datetime.now().timestamp() - start)

figure = plt.figure()
im = plt.imshow(noise[..., 0])
def init():
    im.set_data(noise[..., 0])
    return [im]

def animate(i):
    b = i
    if i >= Z_INT:
        b = 2 * Z_INT - 1 - i
    im.set_array(noise[..., b])
    return [im]

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = animation.FuncAnimation(figure, animate, init_func=init, frames=(2*Z_INT), interval=40, blit=True)
anim.save('octave_animation.gif', writer=writer)
plt.show()
