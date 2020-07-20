from numpy import *

from wiphy.code.modulator import *
from wiphy.util.general import *

psksymbols = generatePSKSymbols(4)
print(f"QPSK = {psksymbols}")
print(f"Mean = {mean(psksymbols)}")
print(f"MED = {getMinimumEuclideanDistance(psksymbols.reshape(4, 1, 1))}")

qamsymbols = generateQAMSymbols(16)
print(f"16-QAM = {qamsymbols}")
print(f"Mean = {mean(qamsymbols)}")
print(f"Variance = {mean(square(abs(qamsymbols)))}")
print(f"MED = {getMinimumEuclideanDistance(qamsymbols.reshape(16, 1, 1))}")
