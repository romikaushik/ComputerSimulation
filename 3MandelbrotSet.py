import numpy as np 
import matplotlib.pyplot as plt 
 
def mandelbrot(c, max_iter): 
    z = 0  
    for n in range(max_iter): 
        if abs(z) > 2:  
            return n  
        z = z**2 + c  
    return max_iter  

def generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter): 
    real = np.linspace(xmin, xmax, width) 
    imag = np.linspace(ymin, ymax, height) 
    mandelbrot_set = np.empty((height, width)) 
     
    for i, im in enumerate(imag):  
        for j, re in enumerate(real):  
            c = complex(re, im)  
            mandelbrot_set[i, j] = mandelbrot(c, max_iter)  
     
    return mandelbrot_set  

def plot_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax): 
    plt.figure(figsize=(10, 10)) 
    plt.imshow(mandelbrot_set, extent=[xmin, xmax, ymin, ymax],  
               cmap='hot', interpolation='bilinear') 
    plt.colorbar(label='Iteration count')  
    plt.title("Mandelbrot Set")  
    plt.xlabel("Re(C)")  
    plt.ylabel("Im(C)")  
    plt.show()  

xmin, xmax = -2.025, 0.6  
ymin, ymax = -1.125, 1.125  
width, height = 1000, 1000  
max_iter = 255  

mandelbrot_set = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)  
plot_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax)  
