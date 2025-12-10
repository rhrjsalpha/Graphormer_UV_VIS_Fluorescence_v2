import pandas as pd

test = pd.read_csv("EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv")
training = pd.read_csv("EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv")
test_1 = pd.read_csv("ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv")
trainig_1 = pd.read_csv("ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv")

solv = pd.concat([test["Solvent"], training["Solvent"], test_1["Solvent"], trainig_1["Solvent"]], axis=0)
print(solv)
print(test.columns)
print(solv.unique())

def matcher(qury:str, ref:list):
    for str in ref:
        if qury in str or str in qury:
            return False
        else:
            pass
    return True

print("test matcher",matcher("water", ["water ", "solv"]), )

unique_list = list()
for i in solv.unique():
    if "+" in i:
        splited = i.split("+")
        #print(splited)
        for j in splited:
            j = j.replace(" ", "")
            if not j in unique_list:
                unique_list.append(j)
    else:
        i = i.replace(" ", "")
        print(i)
        if not i in unique_list:
            unique_list.append(i)
        #print(i)

print(unique_list)