if __name__ == "__main__":
    import os
    for i in range(10):
        print(F'running dcor exp {i} ...')
        os.system(F'python3 hsp_mort_evalml_dcor_scaled_w_isobs.py')
        print(F'running dcor exp with isobs {i} ...')
        os.system(F'python3 hsp_mort_evalml_dcor_scaled.py')

        print(F'running rfe exp {i}....')
        for j in range(60, 200, 10):
            print(F'buiding rfe model with {j}....')
            os.system(F'python3 hsp_mort_evalml_rfe_w_isobs.py -n={j}')
            os.system(F'python3 hsp_mort_evalml_rfe.py -n={j}')
