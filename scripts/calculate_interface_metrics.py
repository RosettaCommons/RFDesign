import sys
import glob
import subprocess
import os
output_dir = sys.argv[1]

split_dirs = []
for dir_name in glob.glob(f'{output_dir}/*/'):
    if os.path.basename(os.path.normpath(dir_name)).isdigit():
        split_dirs.append(dir_name)
        

print("Calculating AF2 interface metrics...")
for split_dir in split_dirs:
    os.chdir(split_dir)
    print(f"Generating silent file from split_dir {split_dir}")
    subprocess.run(['/bin/bash', '-c', 'export PATH=$PATH:/software/silent_tools && silentfrompdbs_patch *pdb > in.silent && sed -i -e 1,2d in.silent'])
    print(f"Running AF2 interface metrics on silent file")
    subprocess.run(['/home/biolib/RFDesign/scripts/af2_interface_metrics.py', '-silent', 'in.silent'])


print("Calculating PyRosetta Interface Metrics...")
for split_dir in split_dirs:
    os.chdir(split_dir)
    subprocess.run(['/home/biolib/RFDesign/scripts/get_interface_metrics.py', '.'])


print("Getting binder rmsd for batches")
batch_dir_names = [os.path.basename(os.path.normpath(dir_name)) for dir_name in split_dirs]
for batch_dir_name in batch_dir_names:
    os.chdir(output_dir)
    print("Getting binder RMSD for batch ", batch_dir_name)
    subprocess.run(['/home/biolib/RFDesign/scripts/get_binder_rmsd.py', batch_dir_name])