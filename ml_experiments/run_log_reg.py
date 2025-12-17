import os
n_experiments = 10

print('############## LogReg Experiment Starting ############')
for i in range(n_experiments):
    print(F'run {i} of {n_experiments}')
    for j in range(30, 150, 10):
        cmd = F"python3 HospitalMortalityClassifier_rfe_LogReg.py -n={j}"
        print(F'running {cmd}...')
        os.system(cmd)
    cmd = "python3 HospitalMortalityClassifier_dcor_LogReg.py"
    print(F'running {cmd}...')
    os.system(cmd)

print('############## LogReg  Experiment Run Completed ############')
