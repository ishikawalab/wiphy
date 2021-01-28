from tqdm import trange
from numpy import *
from wiphy.util.general import *
set_printoptions(linewidth=infty)


snr_dBs = linspace(-20, 20, 21)
sigmav2s = 1.0 / inv_dB(snr_dBs)
capacity = zeros(len(snr_dBs))

IT = int(1e4)
for _ in trange(IT):
    H = randn_c(4, 4)
    evalues = real(linalg.eig(H.T.conj() @ H)[0])
    for i in range(len(snr_dBs)):
        capacity[i] += sum(log2(1 + evalues / sigmav2s[i]))
capacity /= IT

ret = {"snr_dB": snr_dBs, "capacity": capacity}
print(repr(ret))
