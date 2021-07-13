import os
from glob import glob

n_test = 16

pattern = '/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/Calu3_MOI0.5_24h_H2/bdv/tomos/*.h5'
files = glob(pattern)
files.sort()

test_folder = './test_input'
os.makedirs(test_folder, exist_ok=True)

for ff in files[:n_test]:
    fname = os.path.split(ff)[1]
    out = os.path.join(test_folder, fname)
    os.symlink(ff, out)
