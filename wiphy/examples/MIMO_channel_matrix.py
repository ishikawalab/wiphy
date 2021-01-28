from numpy import *
from wiphy.util.general import *

# 2x2 MIMO
H = randn_c(2, 2)
print("H = %s = U d V" % H)

U, s, VH = linalg.svd(H)
print(f"U = {U}")
print(f"s = {s}")
print(f"VH = {VH}")

# Generate 1000 complex Gaussian variables
print(f"Mean = {mean(randn_c(1000))}")
print(f"Variance = {mean(square(abs(randn_c(1000))))}")
