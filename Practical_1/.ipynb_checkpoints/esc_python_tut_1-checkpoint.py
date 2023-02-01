# Title: A refresher on Python
# Author: Frans-Paul Pieterse
# Description: This file is to refresh the students' Python skills and demonstrate
# some of the basic techniques that can be useful in the course. The SDR interfacing
# has be omitted as this document focusses on data manipulation.


# GOOGLE IS YOUR FRIEND !!! 
# If you ever struggle with any function, just Google the library and function name
# i.e. "Numpy How to use arange"
 

# *** *** Starting Off *** ***


# Import the Numpy library
# Numpy contains most of the tools needed for mathematical operations and 
# some tools can be used for signal processing 

import numpy as np # "as np" creates the alias np so that we do not need to type out numpy

# Import the Matplotlib library, but only include Pyplot. This is what we will use to
# plot data. We will use the alias, "plt".

import matplotlib.pyplot as plt


# State the sampling rate to be used. The SDR has a max sampling rate of 2.4 MHz,
# but for this example I am going to use 50 kHz (selected arbitrarily). 
# NB! Always adhere to Nyquist!

fs = 50e3


# *** *** Time Domain Examples with Constellation Diagram *** ***

# Now we are going to create a time array, to create time-dependent functions later
# We will create an array starting at 0 s and ending at 10 ms. Due to our sampling rate
# each sample will be spaced 1/fs apart i.e. 0 1/fs 2/fs 3/fs ... 10e-3.
# To do this we will use the 'arange' function from numpy.

t = np.arange(0, 10e-3, 1/fs)


# Create a cosine with a frequency of 1 kHz. Due to the time array ending at 10 ms, 
# there should be 10 cycles. 
# fc will specify the frequency of the cosine and y_r will be the array containing the
# cosine.  
# This cosine will be our real or in-phase samples - as if we recorded it.

fc = 1000;
y_r = np.cos(2*np.pi*fc*t) # Note that both cos and pi are from Numpy.

# We are going to plot the cosine. As it is a function of time, t will be on the x-axis.
# Note: We are labelling our axes, but we are not adding a title. In a report, each
# figure should have a caption below, and therefore a title is redundant. 
# Note that y_r is a set of real float values.

plt.figure(1) # This just keeps all of the properties of this plot together
plt.plot(t,y_r) # Plot y_r as a function of time.
plt.xlabel("Time (s)") # Always label your axes.
plt.ylabel("Amplitude") # We did not specify a unit.
plt.grid() # Use a grid
plt.xlim(0,0.01) # This makes sure that the plot visually starts exactly on the left vertical
# line and ends at the right vertical line - It looks better!
plt.show() # It is good to add this line.

# Create a sine function and do the plotting again. Although we can use "np.sin()" let's use "np.cos()" to
# demonstrate an added phase. The frequency will still be 1000 kHz and have the same
# start and stop time as the cosine above. This sine will our quadrature samples -for demonstration.

phi = np.pi/2 # 90 degrees phase 
y_i = np.cos(2*np.pi*fc*t-phi)

# All of the plot properties are chosen as above.
# Note that y_i is also a set of real float values.
plt.figure(2) 
plt.plot(t,y_i) 
plt.xlabel("Time (s)") 
plt.ylabel("Amplitude") 
plt.grid()
plt.xlim(0,0.01) 
plt.show()

# Let us plot the cosine and sine together for demontration. If you ever do this, remember
# to add a legend.
plt.figure(3)
plt.plot(t,y_r,t,y_i) # Note that I use the form plot(x1,y1,x2,y2)
plt.xlabel("Time (s)") 
plt.ylabel("Amplitude") 
plt.grid()
plt.xlim(0,0.01) 
plt.show()

# Let us create a complex array by using our synthetic in-phase and quadrature arrays.
# Note: Remember that cos(x) + j sin(x) = e^(jx) Euler's Identity. This is also called
# a complex sinusoid.

y = y_r + 1j*y_i # Yes, it is that easy.



# And now, let us plot a constellation diagram of our complex signal. 
# We can always plot it as plt.plot(y_r,y_i), but let us imagine that we do not have
# y_r and y_i - just the complex parts.
# We will plot the imaginary (quadrature) part of the complex signal as a function of
# the real part (in-phase)
# Note: Because we have a complex sinusoid with a constant frequency, we also 
# have a constant rate of change in phase. The magnitude of our complex sinusoid is 1,
# so we expect a circle (samples at all phase values) with a magnitude of 1.

plt.figure(4)
plt.plot(np.real(y), np.imag(y)) 
plt.xlabel("In-Phase") 
plt.ylabel("Quadrature") 
plt.grid()
plt.show()

# Just to demonstrate the magnitude, let's plot the magnitude of the complex
# sinusoid

y_mag = np.abs(y) # The magnitude is the absolute value

plt.figure(5) 
plt.plot(t,y_mag) 
plt.xlabel("Time (s)") 
plt.ylabel("Magnitude") 
plt.grid()
plt.xlim(0,0.01)
plt.ylim(0,1.5) # We can also adjsut the y-limit (Solely asthetic purposes!) 
plt.show()


# *** *** Frequency Domain Examples *** ***
# Read up on sampling! Watch any lectures from Prof. du Plessis about real sampling
# and complex sampling (I and Q)


# Let us take the FFT of the cosine y_r and plot it as demonstration.
# First we are going to select an FFT size and create an array for the frequency values.
N_fft = 500; # You can just use the size of the input array too

f_1 = np.arange(0, fs, fs/N_fft) # An array from 0 Hz to fs Hz. It has N_fft values,
# so the interval should be the span divided by the size (fs/N_fft).

# Now the FFT. It will have a real input and complex output.
Y_r = np.fft.fft(y_r, N_fft)/N_fft # This array is complex, so has a magnitude and phase.
# It is also good to normalize the FFT by dividing the output by the size



# Let us plot the FFT magnitude
Y_r_mag = 10*np.log10(np.abs(Y_r))  # The magnitude of the complex FFT output. We always use 
# the logarithmic magnitude for frequency plots as it conveys more information!

# Note that there is a component at 1 kHz, that the frequency spans from 0 to fs,
# and that the spectrum is mirrored around fs/2 (25 kHz)

plt.figure(6)
plt.plot(f_1,Y_r_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.xlim(0,fs)
plt.show()

# We can also plot the phase of the FFT.

Y_r_phase = np.angle(Y_r) # "np.angle returns the equivalent of atan(imag/real)

plt.figure(7)
plt.plot(f_1,Y_r_phase)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()
plt.xlim(0,fs)
plt.show()


# If we want to know at which bin (index) the frequency component is on the magnitude plot, we can just omit the
# f_1 in the plot function . Note that the 1 kHz component is at the 6th bin (bin 5 - we index from 0)
plt.figure(8)
plt.plot(Y_r_mag)
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude (dB)")
plt.xlim(0,N_fft)
plt.grid()
plt.show()

# Important note! You can find the corresponding phase by plotting the phase in the same way
# and reading the phase off at bin 5. Try this yourself.


# A (visually) nicer way to plot the magnitude and phase of the FFT, is to plot it from -fs/2 to fs/2

f_2 = np.arange(-fs/2, fs/2, fs/N_fft)

# Due to spectrum repitition during sampling, we know that the part of the signal from fs/2 to fs 
# is the same as that from -fs/2 to 0. We can therefore just shift it. We can use fftshift.

Y_r_shifted = np.fft.fftshift(Y_r)

# We can then get the magnitude and phase of the result. We will just plot the magnitude.

Y_r_shifted_mag = 10*np.log10(np.abs(Y_r_shifted))

plt.figure(9);
plt.plot(f_2,Y_r_shifted_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.xlim(-fs/2,fs/2)
plt.show()



# For the last part, we will look at what happens when we take the FFT of complex data (such as y above).

# The FFT will take a complex input and gives a complex output. The output can then
# also be converted into magnitude and phase. We will again use N_fft as is

Y = np.fft.fft(y,N_fft)/N_fft
Y_mag = 10*np.log10(np.abs(Y))

plt.figure(10)
plt.plot(f_1,Y_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0,fs)
plt.grid()
plt.show()

# Note that this spectrum has no reflections around fs/2 (Magic! Another reason why you should read
# up on complex sampling and follow Prof. du Plessis's lectures thoroughly)
# I and Q sampling produces single-sided spectra, where the entire band can be used. 

# Lastly, let us plot it shifted.

Y_shifted = np.fft.fftshift(Y)
Y_shifted_mag = 10*np.log10(np.abs(Y_shifted))

plt.figure(11)
plt.plot(f_2,Y_shifted_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(-fs/2, fs/2)
plt.grid()
plt.show()

# Last note: Things like loops, conditionals, basic syntax and other functions can
# just be Googled


# *** ***  That's all folks *** ***

# Thank you for watching and have a pleasant and safe journey home.