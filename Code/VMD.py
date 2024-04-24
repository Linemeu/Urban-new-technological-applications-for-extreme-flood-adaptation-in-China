#计算多元回归模型的MA
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft
wurenji_hour=np.load('wurenji_hour.npy',allow_pickle=True)
gaode_hour=np.load('gaode_hour.npy',allow_pickle=True)
wendang_hour=np.load('wendang_hour.npy',allow_pickle=True)
longxishui_hour=np.load('longxishui_hour.npy',allow_pickle=True)
zhihuishuiwu_hour=np.load('zhihuishuiwu_hour.npy',allow_pickle=True)
TPI=np.load('TPI_hour.npy',allow_pickle=True)
WEPI=np.load('WPI_hour.npy',allow_pickle=True)
CPI=np.load('CPI_hour.npy',allow_pickle=True)
BPI=np.load('BPI_hour.npy',allow_pickle=True)
API=np.load('API_hour.npy',allow_pickle=True)
SPI=np.load('SPI_hour.npy',allow_pickle=True)
CAPI=np.load('CAPI_hour.npy',allow_pickle=True)
rain=np.load('maxrain.npy',allow_pickle=True)

alpha = 9000 # moderate bandwidth constraint
tau = 0.000001  # noise-tolerance (no strict fidelity enforcement)
K = 2  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7


name=['TPI_hour','WEPI_hour','CPI_hour','BPI_hour','API_hour','SPI_hour','CAPI_hour',\
      'wurenji_hour','gaode_hour','wendang_hour','longxishui_hour','zhihuishuiwu_hour','rain']
# plt.subplots(5,3,dpi=320)
for i,f in enumerate([TPI,WEPI,CPI,BPI,API,SPI,CAPI,\
                      wurenji_hour,gaode_hour,wendang_hour,longxishui_hour,zhihuishuiwu_hour,rain]):
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
    plt.subplot(5,3,i+1)
    plt.plot(u.T)
    plt.title(name[i]+' decomposed modes',fontproperties='Times New Roman', size=5, weight='bold')
    # 修改坐标轴字体及大小
    plt.yticks(fontproperties='Times New Roman', size=5, weight='bold')  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=5)
    plt.tight_layout()  # 解决绘图时上下标题重叠现象
    np.save(name[i] + "_vmd.npy", u[0])
    # np.save(name[i]+"_vmd.npy",u[0])
plt.figure()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

#接下来整理变量
#1.感知指数；2.降雨数据 3.新技术指数
TPI=np.load('TPI_hour_vmd.npy',allow_pickle=True)
WPI=np.load('WEPI_hour_vmd.npy',allow_pickle=True)
CAPI=np.load('CAPI_hour_vmd.npy',allow_pickle=True)
BPI=np.load('BPI_hour_vmd.npy',allow_pickle=True)
CPI=np.load('CPI_hour_vmd.npy',allow_pickle=True)


