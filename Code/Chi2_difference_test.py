# ----------------------------------------B2建筑----------------------------------------------------------
import semopy
import pandas as pd
import numpy as np
desc = semopy.examples.political_democracy.get_model()
print(desc)
# data = df.iloc[100:400]
data = df.iloc[96:384]
print(data.head())
formula = """d_BPI~BPI_l1+Rain+DW+UAV
# d_BPI~BPI_l1+Rain+DW+UAV+LSD
#             DW~DW_l1+UAV+Rain
            # DW~DW_l1+UAV+Rain
            # UAV~UAV_l1+Rain
            # DW~~UAV
            # UAV~~UAV_l1
"""
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')
# opt.optimize(objective='MLW')
# opt = Optimizer(model)
# objective_function_value = opt.optimize()
print(res)
ins = mod.inspect()
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)
b2_index=[float(stats.chi2),stats['chi2 p-value'],stats.GFI,stats['RMSEA']]
b2_chi=[float(stats.chi2),float(stats.DoF)]

p=chi2.sf(np.abs(float(b2_chi[0]-b1_chi[0])),np.abs(float(b2_chi[1]-b1_chi[1])))
#GFI的提升率百分比
b2_ip=(b2_index[2]-b1_index[2])/b1_index[2]
b12_ip=(b1_index[2]-b2_index[2])/b2_index[2]

ins['p-value']=[round (e,3) for e in list(ins['p-value'])]
ins['z-value']=[round (e,3) for e in list(ins['z-value'])]
ins['Std. Err']=[round (e,3) for e in list(ins['Std. Err'])]
ins['Estimate']=[round (e,3) for e in list(ins['Estimate'])]
ins.to_excel(".\\SEM_result\\B2.xlsx")

