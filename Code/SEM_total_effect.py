

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

#接下来整理变量
#1.感知指数；2.降雨数据 3.新技术指数
# EPI=np.load('EPI_hour_vmd.npy',allow_pickle=True)
TPI=np.load('TPI_hour_vmd.npy',allow_pickle=True)
WPI=np.load('WEPI_hour_vmd.npy',allow_pickle=True)
SPI=np.load('SPI_hour_vmd.npy',allow_pickle=True)
CAPI=np.load('CAPI_hour_vmd.npy',allow_pickle=True)
BPI=np.load('BPI_hour_vmd.npy',allow_pickle=True)
API=np.load('API_hour_vmd.npy',allow_pickle=True)
CPI=np.load('CPI_hour_vmd.npy',allow_pickle=True)

rain=np.load('maxrain.npy',allow_pickle=True)
# rain=np.load('rain_vmd.npy',allow_pickle=True)
wurenji_hour=np.load('wurenji_hour_vmd.npy',allow_pickle=True)
gaode_hour=np.load('gaode_hour_vmd.npy',allow_pickle=True)
wendang_hour=np.load('wendang_hour_vmd.npy',allow_pickle=True)
longxishui_hour=np.load('longxishui_hour_vmd.npy',allow_pickle=True)
zhihuishuiwu_hour=np.load('zhihuishuiwu_hour_vmd.npy',allow_pickle=True)
#---------------------------------------------------------------



df=pd.DataFrame()


# 1.对EPI进行ADL建模
df=pd.DataFrame()
df['TPI']=TPI
# df['TPI']=np.log(TPI+1)
df['d_TPI']=df['TPI'].diff(periods=1)

df['CPI']=CPI
# df['CPI']=np.log(CPI+1)
df['d_CPI']=df['CPI'].diff(periods=1)

df['BPI']=BPI
# df['BPI']=np.log(BPI+1)
df['d_BPI']=df['BPI'].diff(periods=1)

df['WPI']=WPI
# df['WEPI']=np.log(WEPI+1)
df['d_WPI']=df['WPI'].diff(periods=1)

df['SPI']=SPI
# df['SPI']=np.log(SPI+1)
df['d_SPI']=df['SPI'].diff(periods=1)

df['CAPI']=CAPI
# df['CAPI']=np.log(CAPI+1)
df['d_CAPI']=df['CAPI'].diff(periods=1)

df['API']=API
# df['API']=np.log(API+1)
df['d_API']=df['API'].diff(periods=1)

df['BPI_l1']=df['BPI'].shift(-1)
df['TPI_l1']=df['TPI'].shift(-1)
df['WPI_l1']=df['WPI'].shift(-1)

df['rain']=rain
account_t=6
account_t2=4


#-----------------------无人机----------------------------------------------
df['wurenji']=wurenji_hour
df['wurenji']=np.log(df['wurenji']+1)

wurenji_T=[]
# 航空工业“翼龙”-2H应急救灾型无人机于7月21日14时22分起飞，于18时21分进入米河镇通信中断区
#https://www.thepaper.cn/newsDetail_forward_13756495
#因此7.21号下午18时作为分界点，i>234
#
for i in df.index:
    e=1
    # if i > 240:
    if i >=234:
        e = 1
    else:
        e = 0
    wurenji_T.append(e)
df['wurenji_T']=wurenji_T
df['wurenji']=df['wurenji']*df['wurenji_T']
delay_Ir=[]
for i,e in enumerate(df['wurenji']):
    alpha=0.8
    f=np.array([np.power(alpha,i-j) for j in range(i+1)])
    I=df['wurenji'][0:i+1]
    Ir=sum(list(I*f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_wurenji']=delay_Ir
# df['wurenji']=np.log(df['dalay_wurenji']+1)
df['UAV']=df['dalay_wurenji']

#---------------------文档-------------------------------------------------
df['wendang']=wendang_hour
df['wendang']=np.log(df['wendang']+1)
delay_Ir=[]
for i,e in enumerate(df['wendang']):
    alpha=0.8
    f=np.array([np.power(alpha,i-j) for j in range(i+1)])
    I=df['wendang'][0:i+1]
    Ir=sum(list(I*f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_wendang']=delay_Ir
# df['wendang']=np.log(df['dalay_wendang']+1)
df['LSD']=df['dalay_wendang']

#-----------------高德-----------------------------------------------------
df['gaode']=gaode_hour
df['gaode']=np.log(df['gaode']+1)
delay_Ir=[]
for i,e in enumerate(df['gaode']):
    alpha=0.8
    f=np.array([np.power(alpha,i-j) for j in range(i+1)])
    I=df['gaode'][0:i+1]
    Ir=sum(list(I*f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_gaode']=delay_Ir
df['AMap']=df['dalay_gaode']



#----------------------龙吸水-----------------------------------------
#7月22日，武汉市水务局派出16辆“龙吸水”，86位救援人员支援郑州。据长江日报此前报道，武汉驰援的16台“龙吸水”在郑州的11个下穿通道和隧道、涵洞处抽排水量近10万立方米。
#因此取分界点为7.22号0点
#https://www.thepaper.cn/newsDetail_forward_13711400
longxishui_T=[]
for i in df.index:
    #无人机是7月21日下午14时22分，翼龙-2H应急救灾型无人机从贵州安顺机场起飞，在河南上空执行5-6小时的侦查和中继任务
    e=1
    if i >=288:
        e = 1
    else:
        e = 0
    longxishui_T.append(e)
df['longxishui_T']=longxishui_T
df['longxishui']=longxishui_hour
# df['longxishui']=df['longxishui']*df['longxishui_T']

df['longxishui']=np.log(df['longxishui']+1)
delay_Ir=[]
for i,e in enumerate(df['longxishui']):
    alpha=0.8
    f=np.array([np.power(alpha,i-j) for j in range(i+1)])
    I=df['longxishui'][0:i+1]
    Ir=sum(list(I*f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_longxishui']=delay_Ir
# df['longxishui']=df['dalay_longxishui']
df['DW']=df['dalay_longxishui']

df['wurenji']=(df['wurenji']-df['wurenji'].mean())/df['wurenji'].std()

rain_T=[]
for i in df.index:
    e=1
    rain_T.append(e)


#-------------------rain-------------------------------------------------------------------------------------
df['rain']=rain
# df['rain']=np.log(df['rain']+1)
df['Rain']=df['rain']
df['rain_T']=rain_T
df['rain']=df['rain']*df['rain_T']
df['rain']=(df['rain']-df['rain'].mean())/df['rain'].std()
df['rain_l6']=df['rain'].shift(6)
delay_Ir=[]
for i,e in enumerate(df['rain_l6']):
    alpha=0.8
    f=np.array([np.power(alpha,i-j) for j in range(i+1)])
    I=df['rain_l6'][0:i+1]
    Ir=sum(list(I*f)[-account_t2:])
    delay_Ir.append(Ir)
df['dalay_rain']=delay_Ir
df['Rain']=df['dalay_rain']


df=df.dropna()
#----------------------------CAPI人员伤亡--------------------------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 12,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import seaborn as sns
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")
df=(df-df.mean())/df.std()
df['CAPI_l1']=df['CAPI'].shift(-1)
model = ols('d_CAPI~CAPI_l1+Rain+UAV+LSD+DW',data=df,missing='drop')
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
n_palette = ["#ff0000","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05
bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag
df_plot=df_plot.drop('Intercept', axis=0)
# import seaborn as sns
plt.figure(figsize=(16,10),dpi=1000)
plt.subplot(2,3,1)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')

ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.title('dCAPI:the change of CAPI',fontdict=title_font)
plt.tight_layout()

#--------------BPI建筑----------------------------------------------------------------------------------------
# import seaborn as sns
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")
model = ols('d_BPI~BPI_l1+Rain+UAV+DW',data=df,missing='drop')
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
n_palette = ["#ff0000","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05
bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag
df_plot=df_plot.drop('Intercept', axis=0)
# import seaborn as sns
# plt.figure()
plt.subplot(2,3,2)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.title('dBPI:the change of BPI',fontdict=title_font)
plt.tight_layout()


#----------------------------TPI交通-------------------------------------------------------------------------------
# import seaborn as sns
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")

# model = ols('d_TPI~TPI_l1+rain+wurenji+lnlongxishui+diaster+T_after',data=df,missing='drop')
model = ols('d_TPI~TPI_l1+Rain+AMap+DW+UAV',data=df,missing='drop')
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
n_palette = ["#ff0000","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05
bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag
df_plot=df_plot.drop('Intercept', axis=0)
# import seaborn as sns
# plt.figure()
plt.subplot(2,3,3)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.title('dTPI:the change of TPI',fontdict=title_font)
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.tight_layout()

#
#
#----------------------------WEPI水电--------------------------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 12,
         }
import seaborn as sns
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")
df['WPI_l1']=df['WPI'].shift(-1)
model = ols('d_WPI~WPI_l1+Rain+UAV+DW',data=df,missing='drop')
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
n_palette = ["#ff0000","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05
bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag
df_plot=df_plot.drop('Intercept', axis=0)
# import seaborn as sns
# plt.figure()
plt.subplot(2,3,4)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.title('dWEPI:the change of WEPI',fontdict=title_font)
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.tight_layout()

#----------------------------CPI通信------------------------------------------------------------------------
import seaborn as sns
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")
df['CPI_l1']=df['CPI'].shift(-1)
# X=df[['CPI_l1','rain','lnwurenji','lnlongxishui','zhihuishuiwu']]
# # X=sm.add_constant(X)
# y=df['d_CPI']
model= ols('d_CPI~CPI_l1+Rain+UAV+DW',data=df,missing='drop')
# res = sm.OLS(y, X).fit()
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
n_palette = ["#ff0000","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05
bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag

# import seaborn as sns
# plt.figure()
df_plot=df_plot.drop('Intercept', axis=0)
plt.subplot(2,3,5)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.title('dCPI:the change of CPI',fontdict=title_font)
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.tight_layout()

# out_predict=res.predict(X)
# MAPE和SMAPE需要自己实现
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
# mape(y, out_predict)


#----------------------------API农田-------------------------------------------------------------------------
import statsmodels.api as sm
corr=df.corr()
# sns.heatmap(corr, mask=None, vmax=0.3, annot=True,cmap="RdBu_r")
df['API_l1']=df['API'].shift(-1)
df['API_l2']=df['API'].shift(-2)
df['API_l3']=df['API'].shift(-3)
df['API_l4']=df['API'].shift(-4)
df['d_API_l1']=df['d_API'].shift(-1)
df['d_API_l2']=df['API'].shift(-2)
df['d_API_l3']=df['API'].shift(-3)
df['d_API_l4']=df['API'].shift(-4)
df['d_API_l5']=df['API'].shift(-5)
df['d_API_l6']=df['API'].shift(-6)

model = ols('d_API~API_l1+Rain+UAV+DW',data=df,missing='drop')
# res = sm.OLS(y, X).fit()
res = model.fit()
Bata = res.params
print(Bata)
res.summary()

coef=(res.conf_int()[0]+res.conf_int()[1])/2
variable=res.pvalues.index
pvalues=res.pvalues.values
df_plot=pd.DataFrame()
df_plot['coef']=coef
df_plot['variable']=variable
df_plot['pvalues']=pvalues
# n_palette = ["#ff0000","#ff0000","#00ff00"]
n_palette = ["#4E79A7","#ff0000","#00ff00"]
p_palette = ["#00ff00","#00ff00","#ff0000"]
df_plot['p-value']=pvalues>0.05

bag=[]
for i,e in enumerate(df_plot['p-value']):
    if e==False:
        v="p<0.05"
        bag.append(v)
    if e==True:
        v='p>0.05'
        bag.append(v)
df_plot['p-value']=bag
df_plot=df_plot.drop('Intercept', axis=0)
# import seaborn as sns
# plt.figure()
plt.subplot(2,3,6)
ax = sns.pointplot(x=df_plot['variable'], y=df_plot['coef'], hue = df_plot['p-value'], join=False,\
                   linestyle='',marker='o',palette={'p>0.05':"lightpink",'p<0.05':"skyblue"},dodge=False)
err_range=list(1.96*res.bse)[1:]
plt.errorbar(df_plot['variable'], y=df_plot['coef'], fmt="--", yerr=err_range,ecolor='lightcoral',\
             mfc=None,mev=None,capsize=5,elinewidth=2,ls = 'none')
# sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
ax.set(xlabel=None)
ax.set(ylabel=None)
plt.xticks(font=font1,rotation=30)
plt.yticks(font=font1)
plt.legend(prop=font1)
# #将说明图表放置在图表外
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0,xmin=-1,xmax=9,color="black")#横线
plt.title('dAPI:the change of API',fontdict=title_font)
plt.tight_layout()
plt.text(6.5,-0.05,"R-squared="+"%.2f" % res.rsquared,fontdict=font1)
print ("R-squared="+"%.2f" % res.rsquared)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.1, hspace=0.3)#wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
FIGdir="figdir\\"
plt.savefig(FIGdir+'Result_Multiple_regression.jpg')

# --------------BPI建筑----------------------------------------------------------------------------------------

df['UAV_l1'] = df['UAV'].shift(-1)
df['AMap_l1'] = df['AMap'].shift(-1)
df['DW_l1'] = df['DW'].shift(-1)
df['LSD_l1'] = df['LSD'].shift(-1)
df['CPI_l1'] = df['CPI'].shift(-1)

# ----------------------------通讯---------------------------------------
import semopy
import pandas as pd

desc = semopy.examples.political_democracy.get_model()
print(desc)
# data = df.iloc[100:250]
#7.14到7.28
data = df.iloc[96:432]
#7.14到7.26
data = df.iloc[96:384]
print(data.head())
data['e1'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e2'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e3'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e4'] = np.random.normal(size=len(data))  # 正态分布随机数
formula = """d_CPI~CPI_l1+Rain+UAV+LSD
            LSD~LSD_l1+Rain
            # LSD~LSD_l1+UAV+Rain
            # UAV~UAV_l1+Rain
            # LSD~UAV+Rain 
            # LSD~LSD_l1+Rain aic:15.96
            # UAV~UAV_l1+Rain
            # LSD~~UAV
            # LSD~~LSD_l1
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
# g = semopy.semplot(ins, "pd.png")
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)

# ---------------------------------------交通--------------------------------------------------
import semopy
import pandas as pd

desc = semopy.examples.political_democracy.get_model()
print(desc)
# data = df.iloc[100:350]
data = df.iloc[96:432]
#7.14到7.26
data = df.iloc[96:384]
print(data.head())
data['e1'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e2'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e3'] = np.random.normal(size=len(data))  # 正态分布随机数
data['e4'] = np.random.normal(size=len(data))  # 正态分布随机数
data['Rain_l1'] = data['Rain'].shift(-1)
formula = """d_TPI~TPI_l1+Rain+DW+AMap+UAV
            DW~DW_l1+UAV+Rain
            # AMap~AMap_l1+UAV+Rain
            # AMap~AMap_l1+UAV
            # AMap~AMap_l1+UAV_l1+Rain         
            # AMap~~AMap_l1
            # AMap~~UAV
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
# g = semopy.semplot(ins, "pd.png")
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)

# ----------------------------------------人员丧亡----------------------------------------------------------
import semopy
import pandas as pd
desc = semopy.examples.political_democracy.get_model()
print(desc)
# data = df.iloc[96:432]
#7.14到7.26
data = df.iloc[96:384]
# data = df.iloc[96:384]
print(data.head())
formula = """d_CAPI~CAPI_l1+Rain+LSD+UAV+DW
            # DW~DW_l1+UAV+Rain
            LSD~LSD_l1+UAV+Rain
            # AMap~AMap_l1+UAV+Rain
            # AMap~~UAV
            # LSD~~UAV
            # LSD~~LSD_l1
            # AMap~~LSD
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
# g = semopy.semplot(ins, "pd.png")
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)

# ----------------------------------------水电----------------------------------------------------------
import semopy
import pandas as pd

desc = semopy.examples.political_democracy.get_model()
print(desc)
data = df.iloc[96:384]
# data = df.iloc[100:400]
data = df
print(data.head())
formula = """d_WPI~WPI_l1+Rain+UAV+LSD
            LSD~LSD_l1+UAV+Rain
            # UAV~UAV_l1+Rain
            # LSD~~UAV
            # LSD~~LSD_l1

"""
mod = semopy.Model(formula)
res = mod.fit(data)
from semopy import Optimizer

opt = Optimizer(mod)
opt.optimize('DWLS')
# opt.optimize('MLW')
# opt = Optimizer(model)
# objective_function_value = opt.optimize()
print(res)
ins = mod.inspect()
# g = semopy.semplot(ins, "pd.png")
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)

# ----------------------------------------建筑----------------------------------------------------------
import semopy
import pandas as pd

desc = semopy.examples.political_democracy.get_model()
print(desc)
# data = df.iloc[100:400]
data = df.iloc[96:384]
print(data.head())
formula = """d_BPI~BPI_l1+Rain+DW+UAV
            DW~DW_l1+UAV+Rain
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
# g = semopy.semplot(ins, "pd.png")
print(ins)
stats = semopy.calc_stats(mod)
print(stats.T)

#---------------------total effect条形图-------------------------------------------------
#---------------------人员伤亡---------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import matplotlib.pyplot as plt
plt.figure(dpi=320,figsize=(6,4))
labels = ['UAV', 'DW', 'LSD', 'P','CAPI(-1)']
values = [-0.20, -0.05, -0.52, 0.10,-0.20]
plt.bar(labels, values,color='lightsalmon')
# f.set_color('orange')
plt.ylabel('Total effect',fontdict=font1)
plt.yticks(font=title_font)
plt.xticks(font=title_font)
plt.hlines(0,xmin=-1,xmax=6,color="black",linewidth=0.5)#横线
plt.tight_layout()
plt.show()
plt.savefig('.\\figdir\\CA_total_effect.jpg')


#---------------------通信---------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import matplotlib.pyplot as plt
plt.figure(dpi=320,figsize=(6,4))
labels = ['UAV', 'LSD', 'P','CPI(-1)']
values = [-0.32, -0.006, 0.24, 0.54]
plt.bar(labels, values,color='lightsalmon')
# f.set_color('orange')
plt.ylabel('Total effect',fontdict=font1)
plt.yticks(font=title_font)
plt.xticks(font=title_font)
plt.hlines(0,xmin=-1,xmax=5,color="black",linewidth=0.5)#横线
plt.tight_layout()
plt.show()
plt.savefig('.\\figdir\\C_total_effect.jpg')

#---------------------水电供应---------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import matplotlib.pyplot as plt
plt.figure(dpi=320,figsize=(6,4))
labels = ['UAV', 'LSD', 'P','WEPI(-1)']
values = [-0.34, -0.79, 0.09, 0.86]
plt.bar(labels, values,color='lightsalmon')
# f.set_color('orange')
plt.ylabel('Total effect',fontdict=font1)
plt.yticks(font=title_font)
plt.xticks(font=title_font)
plt.hlines(0,xmin=-1,xmax=5,color="black",linewidth=0.5)#横线
plt.tight_layout()
plt.show()
plt.savefig('.\\figdir\\WE_total_effect.jpg')

#---------------------交通---------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import matplotlib.pyplot as plt
plt.figure(dpi=320,figsize=(6,4))
labels = ['UAV', 'AMap', 'DW','P','TPI(-1)']
values = [-0.06, -0.19, -0.14, 0.19,-0.58]
plt.bar(labels, values,color='lightsalmon')
# f.set_color('orange')
plt.ylabel('Total effect',fontdict=font1)
plt.yticks(font=title_font)
plt.xticks(font=title_font)
plt.hlines(0,xmin=-1,xmax=5,color="black",linewidth=0.5)#横线
plt.tight_layout()
plt.show()
plt.savefig('.\\figdir\\T_total_effect.jpg')

#---------------------建筑---------------------------------------------------------
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
title_font = {'family': 'Times New Roman',
         'weight': 'semibold',
         'size': 14,
         }
import matplotlib.pyplot as plt
plt.figure(dpi=320,figsize=(6,4))
labels = ['UAV','DW','P','BPI(-1)']
values = [-0.34,-0.10,0.23,-0.35]
plt.bar(labels, values,color='lightsalmon')
# f.set_color('orange')
plt.ylabel('Total effect',fontdict=font1)
plt.yticks(font=title_font)
plt.xticks(font=title_font)
plt.hlines(0,xmin=-1,xmax=5,color="black",linewidth=0.5)#横线
plt.tight_layout()
plt.show()
plt.savefig('.\\figdir\\B_total_effect.jpg')

