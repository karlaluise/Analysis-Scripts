# -*- coding: utf-8 -*-
"""
This script plots mid-point titres, area under curve and average serum dilutions for 
serum ELISAs against a number of ELISA antigens. 

User input is required for number of mice total, number of mice/group and the number of 
antigens assayed against. 

Input files are assumed to be standard export .txt from IPD SpectraMax M3 with files 
numbered such that grouped data is continuous. Data should be named to allow sorting 
into correct order eg. 01_file.txt, 02_file.txt...Script will read ALL files in a folder. 

Created by Karla-Luise Herpoldt during the Great Plague of 2020 (04-28-20)
With Seaborn-wizardry-help from Ajasja Ljubetič

Written in Python 3.7
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import glob
import numpy as np
from sklearn import metrics
from numpy import trapz
#Will need to install statannot package. Use command "pip install statannot" in command line
from statannot import add_stat_annotation

"""Users must change values below"""

#number of mice in study
nmice = 30

#number of mice per group
micepergroup = 5

#number of groups
ngroups = 6

#number of antigens assayed against
nantigen = 9

#ELISA antigens
antigens = ['I53-50','I53-50A', 'I53-50B', 'VP8* Wa', 'VP8* DS1', 'VP8* 1076', 'VP8* L26', 'VP8* bovineP5', 'VP8* ST3']

#Directory containing input data
dirpath = '/Users/uadmin/Desktop/bleed2'

#define dilution series, "0" is blank ad is assumed to be in row H
dilutions = [100,500,2500,12500,62500,312500,1562500]

#chooseplot type from: AUC, IC50, titre
plottype = 'AUC'

""" End of required user input """

#read in data files from standard export from SpectraMax M3 and trim unnecessary columns/rows. 
dataframes = []

inputfiles = sorted(glob.glob(dirpath+'/*.txt'))


for file in inputfiles:
    df = pd.read_csv(file, encoding='latin1', skiprows=3,skipfooter=2, delimiter="\t", header=None)
    df.drop(df.columns[[0,1,-1,-2]], axis=1, inplace=True)
    df.drop(df.index[-1],axis=0, inplace=True)
    dataframes.append(df)
    
wholedata = pd.concat(dataframes, axis = 1, sort = False, ignore_index=True)

#remove empty columns on end of last plate
while len(wholedata.columns) % nmice != 0:
    wholedata.drop(wholedata.columns[-1], axis=1, inplace=True)

#calculate average blank value
avgblank = wholedata.iloc[-1].mean(axis=0)
 
#bs=blank subtracted values
bs=wholedata - wholedata.iloc[-1]
bs.drop(bs.index[-1],axis=0,inplace=True)

""" calculate EC50 """
# define 4pl logistic
def logistic4 ( x , A , B , C , D ):
    "" "4PL logistic equation." ""
    return (( A - D ) / ( 1.0 + (( x / C ) ** B ))) + D

#debug mode, use calculated set of EC50 values rather than recalculating new
#EC50values = [2.0, 2.0, 2.0, 2.0, 2.0, 3.2964078614349774, 3.4318553398300593, 3.1812684683393324, 3.295957878096406, 3.268203451228008, 2.0, 2.0, 2.0, 2.0, 2.0, 2.9605830239442272, 3.890151341521127, 3.471146629182961, 3.243681340692451, 3.6691118408054546, 3.2525715804437803, 3.229926267143195, 3.4406890607038534, 3.307163991057086, 3.31360393660697, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.7119619270197663, 2.4866051967112766, 2.9141242317996454, 2.5922189736616343, 3.169678584359545, 2.0, 2.0, 2.168039371524987, 2.0, 2.0, 2.1505277678932027, 2.907930931885667, 2.1004626838417098, 2.468641685768275, 2.5768170519697207, 3.1271933751849597, 2.0, 2.3767809993352995, 2.0, 3.36543710231556, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.6699428839489565, 2.0, 2.5288407466766887, 2.0, 2.0, 2.0, 2.630215928309467, 2.770896316905759, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4541763159551784, 2.1456185870221414, 2.1438235741299274, 2.6771309588031444, 2.4734999484986746, 2.0, 2.3087550591318986, 2.0, 2.438403468882444, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.230654746208462, 3.068215483392454, 2.0, 2.352426986232901, 2.690181104806591, 2.099094030362023, 2.2178345866265334, 3.0806054307988826, 2.5605066606263525, 2.235724910852017, 2.6575078092157245, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.163634051451198, 2.619274011444726, 2.514883769569247, 2.9649961084276155, 2.8015239629089175, 2.0, 2.0, 2.0, 2.0, 2.0, 3.1073919977428472, 3.544730873437062, 2.599974071327865, 2.848752643588523, 3.280006679586573, 2.189707926423051, 2.0, 2.6143605074500393, 2.7034784990888543, 3.0115606572563953, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.27336818340723, 4.04799221430951, 3.592743793658874, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.995975688273176, 3.5732550553568054, 3.39506084846069, 3.370822079821208, 3.401392313171533, 3.375013097368634, 3.394570103916625, 3.388868214368571, 3.657563714366117, 3.136304036407568, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.040508130602245, 2.0, 3.2116603860153, 2.0, 2.0, 2.6389430594036325, 2.0, 2.636897920833082, 2.0, 2.0, 3.0169098045868568, 2.793891348667605, 2.0, 2.0, 2.0, 2.5671137710565715, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.438693613300067, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
EC50values = []

xdata = dilutions
for i in range(0,(len(wholedata.columns))):
  ydata = bs[i]
  if bs[i][0] < avgblank:
      EC50values.append(np.log10(dilutions[0]))
  else:
      try:
          popt , pcov = curve_fit(logistic4, xdata , ydata, maxfev=10000)
          x_fit = np.linspace(100 , 2000000 , 2000000 )
          y_fit = logistic4(x_fit , *popt)
          if popt[2]<dilutions[0]:
              EC50values.append(np.log10(dilutions[0]))
          else:
              EC50values.append(np.log10(popt[2]))
      except RuntimeError:
          popt , pcov = curve_fit(logistic4, xdata , ydata, maxfev=1500000)
          x_fit = np.linspace(100 , 2000000 , 2000000 )
          y_fit = logistic4(x_fit , *popt)
          if popt[2]<dilutions[0]:
              EC50values.append(np.log10(dilutions[0]))
          else:
              EC50values.append(np.log10(popt[2]))
              
""" calculate AUC """
AUCvalues = []

for i in range(0,(len(wholedata.columns))):
    if bs[i][0] < avgblank:
        AUCvalues.append(np.log10(dilutions[0]))
    else:
        AUCvalues.append(np.log10(abs((trapz(xdata, bs[i])))))
        
#Add calculated values to dataframe

transp = bs.transpose()

transp['IC50'] = EC50values
transp['AUC'] = AUCvalues

#Add antigen names and group names to dataframe

groups = []
grouplabels = []
antigenlabels = []
j=0
while j <= ngroups-1:
    grouplabels.append(j+1)
    j=j+1

n=0
z=0
while len(groups) < nmice*nantigen:
    i=0
    while i < (len(grouplabels)):
            if n < micepergroup:
                groups.append('Group '+str(grouplabels[i]))
                antigenlabels.append(str(antigens[z]))
                n=n+1 
            else:
                n=0
                i=i+1
    z=z+1     
    
transp.insert(0, 'Experimental Groups', groups)
transp.insert(0, 'antigen', antigenlabels)

transp.set_index('antigen')

#plot multipanel figure depending on plot-type chosen above

sns.set_palette("husl")
fg = sns.catplot(x='Experimental Groups', y=plottype, col = 'antigen', data=transp, kind='swarm',col_wrap=nantigen/3)

median_width=0.5

def plot_means(x,y, color, data):
    #print(x,y, color, data)
    means = data.groupby(['Experimental Groups'])[plottype].mean()
    
    for n, mean in enumerate(means):
        plt.plot([n-median_width/2, n+median_width/2], [mean,mean], lw=2.5, color='darkgrey')
        
fg.map_dataframe(plot_means, x='Experimental Groups', y=plottype)

def plot_stats(x,y,color, data):
    #Introduce minor test variance inserted here to avoid zero variance test errors:
    for j in range(0,len(transp)):
        if transp[plottype][j] == 2.0:
            transp.at[j, plottype] = 2.0+ np.random.uniform(-0.005,0.005)
    transp.set_index('antigen', inplace=True)
    for i in range(0, len(antigens)):
        Test_Data = transp.loc[antigens[i]]
        print (Test_Data)
        test_results = add_stat_annotation(fg.axes.flat[i], data=Test_Data, x='Experimental Groups', y=plottype, perform_stat_test=True,
                                   box_pairs=[("Group 1", "Group 2"), ("Group 3", "Group 4"),("Group 1", "Group 3")],
                                   test='Mann-Whitney', text_format='star',
                                   loc='inside', verbose=1)
    

#Mann-Whitney
fg.map_dataframe(plot_stats, x='Experimental Groups', y=plottype)

plt.savefig(dirpath+'/'+plottype+'.png')