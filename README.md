## Vectorized, General-Dimension Classic Perlin Noise

#### Usage   

This code can be used to create perlin noise for a grid (or set of points).
Instantiate an instance of VecPerlinNoise, with the dimensions argument for the number of dimensions you'd like to create noise in.  
To generate a block of 3d noise, for example:  
```
noise_gen = VecPerlinNoise(dimensions=3)  
x, y, z = np.mgrid[0:20:100j, 0:20:100j, 0:20:100]  
noise = noise_gen.noise(x, y, z, octaves=4, base_freq=.2, persistence=.5)
```  
Note that points should be in float form, spread between 0 and max_index  
See gen_vec_animator.py for more info

I know it is not pythonic to reinvent the wheel, but it's a good way to learn.  Plus I haven't seen a generalized implementation yet.


