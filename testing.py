import os
from dataset import KAMPdataset
import numpy as np
import matplotlib.pyplot as plt
from xwt import xwt

dataset = KAMPdataset(data_path=os.path.join(os.getcwd(), 'data','test.csv'),
                      window_size=4096,
                      stride=100)
img, signal, state = dataset[1]
print(state)
s1 = signal[:4096].numpy()
s2 = signal[4096:].numpy()
print(len(s1))
print(len(s2))
WXamp, WXspec, WXangle, Wcoh, WXdt, freqs, coi = xwt(s1, s2, 140000/14, 3, 0.005, 20, 1, 25, 4096)

time = [i*0.00001 for i in range(4096)]

## Plotting

# plt.figure(1)

# plt.subplot(2, 2, 1)
# plt.tight_layout()
plt.pcolormesh(time, freqs, WXdt, cmap='jet_r', edgecolors='none')
# plt.clim([-0.02, 0.01])
# plt.colorbar()
# plt.plot(time, 1/coi, 'w--', linewidth=2)
# plt.ylim(freqs[-1], freqs[0])
# plt.title('Smoothed Time difference', fontsize=13)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
plt.show()
# plt.subplot(2, 2, 2)
# plt.tight_layout()
# plt.pcolormesh(time, freqs, Wcoh, cmap='jet', edgecolors='none')
# plt.clim([0.985, 1])
# plt.colorbar()
# plt.plot(time, 1/coi, 'w--', linewidth=2)
# plt.ylim(freqs[-1], freqs[0])
# plt.title('Wavelet Coherence', fontsize=13)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.show

# plt.subplot(2, 2, 3)
# plt.tight_layout()
# plt.pcolormesh(time, freqs, np.log(WXamp), cmap='jet', edgecolors='none')
# plt.clim([-50, 0])
# plt.colorbar()
# plt.plot(time, 1/coi, 'w--', linewidth=2)
# plt.ylim(freqs[-1], freqs[0])
# plt.title('(Logarithmic) Amplitude', fontsize=13)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')

# plt.show()