import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Rectangle
from matplotlib.ticker import EngFormatter
import scipy.signal as sp
from cycler import cycler
from quantiphy import Quantity
formatter0 = EngFormatter(unit='Hz')
rc('font',family='serif')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

file = open("data2.raw", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()

I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]

I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5

complex_data = I_samples + 1j*Q_samples

    
start =  53582
end =  77153

DATA_I = I_samples[start:end]
DATA_Q = Q_samples[start:end]
DATA_mag = np.abs(complex_data)[start:end]
DATA_I = DATA_I / DATA_mag.max()
DATA_Q = DATA_Q / DATA_mag.max()
DATA_signal = DATA_I + 1j*DATA_Q

fs = 2.4E6 #Carrier frequency of SDR
dt = 1/(fs) #Timestep between samples 
freq = np.fft.fftfreq(len(DATA_signal),dt)

bins=np.arange(0,len(DATA_signal),1)
t=bins*dt

DATA_SIGNAL_O = np.fft.fft(DATA_signal)
DATA_MAG_O = 10*np.log10(np.abs(DATA_SIGNAL_O))
Δ_Φ = np.diff(np.unwrap((np.angle(DATA_signal[2:227]))))
Δ_f = np.median(Δ_Φ)  / (np.pi*2*dt)



DATA_signal = DATA_signal*(np.cos(2*np.pi*(-1*Δ_f)*t) + 1j*np.sin(2*np.pi*(-1*Δ_f)*t))
DATA_SIGNAL = np.fft.fft(DATA_signal)
DATA_MAG = 10*np.log10(np.abs(DATA_SIGNAL))

DATA_SIGNAL_ABS = np.fft.fft(np.abs(DATA_signal))
DATA_MAG_ABS = 10*np.log10(np.abs(DATA_SIGNAL_ABS))
DATA_SIGNAL_Φ = np.angle(DATA_SIGNAL_ABS)

#Thus we obtain the frequency of the clock by lDBPSKing for the largest spike above the noise.
Index_max = 1000+(DATA_MAG_ABS[1000:4500].argmax())
f_clk=freq[Index_max]
freq_δ = np.abs(freq[0]-freq[1])/2
#Now need to obtain the phase of the clock.
# I did this by using the index function which searches the array and returns the bin where that value is located
for i, j in enumerate(freq):
    if (f_clk-freq_δ) < j < (f_clk+freq_δ):
            freq_bin=(i)


Φ = DATA_SIGNAL_Φ[freq_bin]
f = freq[freq_bin]
ω = 2 * np.pi * f
NCO = np.cos((ω*t)+Φ) + 1j*np.sin((ω*t)+Φ)
NCO = (NCO+1) * (0.4)

peak_bins = sp.find_peaks(np.real(NCO))
symbol_data = []
for k in peak_bins[0]:
        symbol_data.append(DATA_signal[k])
sym_max = np.abs(symbol_data).max()
norm_data = np.real(symbol_data)/sym_max + 1j*np.imag(symbol_data)/sym_max
norm_data = norm_data * np.e**(-1j*np.angle(norm_data[0]))

sd_after=[]
Φ_error = []
Δ_Φ = []
PLL = []
p0 = 1 + 1j*0
ΔΦ = 0
Φ_e = 0
for i in range (len(norm_data)):
    sd_after.append(norm_data[i])
    if i == 0:
        ΔΦ = np.angle(sd_after[i]*np.conj(p0))
        Φ_e += ΔΦ
    else:
        ΔΦ = np.angle(norm_data[i]*np.conj(norm_data[i-1]))
        if 135 < (ΔΦ*57.2958) < 225:
            Φ_e += (ΔΦ - np.pi)
        elif -225 < (ΔΦ*57.2958) < -135:
            Φ_e += (ΔΦ + np.pi)
        else:
            Φ_e += ΔΦ
    sd_after[i] = sd_after[i] * np.e**(-1j*Φ_e)
    Φ_error.append(Φ_e)
    Δ_Φ.append(ΔΦ) 
    PLL.append(np.e**(-1j*Φ_e))

sample = np.real(sd_after[10:])
data=[]
for i, j in enumerate(sample):
    if  -1 <j < - 0.5:
        data.append('0')
        data.append('0')
    elif -0.5 < j < 0:
        data.append('0')
        data.append('1')
    elif 0 < j < 0.5:
        data.append('1')
        data.append('1')
    elif 0.5 < j < 1:
        data.append('1')
        data.append('0')
data = ''.join(data)
message = ""
for i in  range(0,544,8):
    temp_data = data[i:i + 8]
    decimal_data = int(temp_data, 2) 
    message = message + chr(decimal_data)
if message[0] == '^':
    if message[1] == 'I':
        if message[2] != '\n':
            print(start, end)
            print(message)


