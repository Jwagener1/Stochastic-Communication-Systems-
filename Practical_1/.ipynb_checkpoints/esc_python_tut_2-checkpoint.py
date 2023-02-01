# Title: Importing Example data into Python
# Author: Frans-Paul Pieterse

import numpy as np
import matplotlib.pyplot as plt

# Create a file handler.
file = open("data.bin", "r")
# Use "fromfile" to read file data into Numpy array. As stated on Clickup, 
# the data consists of 8-bit unsigned integers.
interleaved_data = np.fromfile(file, np.uint8)
# Always close your file.
file.close()

# I used this trick to de-interleave the data. There are many other methods.
# The data is in the form " real imag real imag real imag..." (interleaved)
# We want two separate arrays - one containing the real data and one containing the imag. 
I_data_raw = interleaved_data[0:len(interleaved_data):2] # This keeps every second 
# sample, starting from index 0 (all of the even index values)
Q_data_raw = interleaved_data[1:len(interleaved_data):2] # This keeps every second
# sample, starting from index 1 (all of the odd index values)

# Note: There are other ways of doing the de-interleaving. One other way is by using a loop.

# As stated on Clickup, 127.5 is the zero value. We therefore need to subtract it
# to remove the offset and center the data around zero. We also have to divide both arrays by
# the largest value to normalize the data.

I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5

# Make the data complex.
complex_data = I_samples + 1j*Q_samples

# Plot the in-phase data.
plt.figure(1)
plt.plot(I_samples)
plt.xlabel("Time Bins")
plt.ylabel("Normalized Amplitude")
plt.xlim(0,len(I_samples))
plt.title("In-Phase Data (5 Bursts: OOK, 4-ASK, DBPSK, DQPSK, D8PSK)")
plt.grid()
plt.show()

# Plot the quadrature data.
plt.figure(2)
plt.plot(Q_samples)
plt.xlabel("Time Bins")
plt.ylabel("Normalized Amplitude")
plt.xlim(0,len(Q_samples))
plt.title("Quadrature Data (5 Bursts: OOK, 4-ASK, DBPSK, DQPSK, D8PSK)")
plt.grid()
plt.show()

# Extract the OOK burst from the data - we call this splicing.
OOK_spliced = complex_data[10000:40000] # format the_array_name[start_index:stop_index]

# Plot the magnitude of the spliced OOK burst.
plt.figure(3)
plt.plot(np.abs(OOK_spliced))
plt.xlabel("Time Bins")
plt.ylabel("Normalized Magnitude")
plt.xlim(0,len(OOK_spliced))
plt.title("Magnitude Plot of OOK Burst")
plt.grid()
plt.show()

# Splice further to "zoom in: on the time axis.
OOK_spliced_further = OOK_spliced[4400:5400]

#Plot the magnitude of the zoomed-in data.
plt.figure(4)
plt.plot(np.abs(OOK_spliced_further))
plt.xlabel("Time Bins")
plt.ylabel("Normalized Magnitude")
plt.xlim(0,len(OOK_spliced_further))
plt.title("Zoomed-In Magnitude Plot of OOK Burst")
plt.grid()
plt.show()

# Note for figure 4:
# From index 205 to 455 is the magnitude of the unmodulated carrier.
# Note the pulse shapes. This is what raised cosine pulse shapes look like if you
# plot their magnitude. 





