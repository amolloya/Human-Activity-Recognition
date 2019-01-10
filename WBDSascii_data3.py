import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import impute
from sklearn.decomposition import PCA

imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
pca = PCA(n_components=9)

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS01walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS01walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS01walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS01walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS01walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS01walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X1 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS02walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,22):
    filename = 'WBDSascii\WBDS02walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS02walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS02walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS02walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,23):
    filename = 'WBDSascii\WBDS02walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X2 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,9):
    filename = 'WBDSascii\WBDS03walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS03walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS03walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS03walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS03walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X3 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS04walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS04walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS04walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS04walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS04walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS04walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X4 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS05walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS05walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS05walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

filename = 'WBDSascii\WBDS05walkO10Smkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = x_[0:250,:]

xS = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS05walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS05walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X5 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS06walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS06walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS06walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

filename = 'WBDSascii\WBDS06walkO10Smkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = x_[0:250,:]

xS = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS06walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS06walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X6 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS07walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS07walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS07walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS07walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

filename = 'WBDSascii\WBDS07walkO10Fmkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = x_[0:250,:]

xF = np.dstack((x1,x2))

X7 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS08walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS08walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,7):
    filename = 'WBDSascii\WBDS08walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS08walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,24):
    filename = 'WBDSascii\WBDS08walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X8 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,8):
    filename = 'WBDSascii\WBDS09walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS09walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS09walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS09walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS09walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X9 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS10walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS10walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS10walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS10walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS10walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS10walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X10 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS11walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,19):
    filename = 'WBDSascii\WBDS11walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS11walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS11walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS11walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS11walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X11 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS12walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

df_mkr = pd.read_csv('WBDSascii\WBDS12walkO10Cmkr.txt',delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS12walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,19):
    filename = 'WBDSascii\WBDS12walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS12walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xF = np.dstack((x1,x2))

X12 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,7):
    filename = 'WBDSascii\WBDS13walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS13walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS13walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS13walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS13walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X13 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS14walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

df_mkr = pd.read_csv('WBDSascii\WBDS14walkO10Cmkr.txt',delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS14walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS14walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS14walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS14walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X14 = np.dstack((xS,xC,xF))


###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS15walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,21):
    filename = 'WBDSascii\WBDS15walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS15walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS15walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS15walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X15 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS16walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

df_mkr = pd.read_csv('WBDSascii\WBDS16walkO10Cmkr.txt',delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS16walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS16walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS16walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS16walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X16 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS17walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS17walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS17walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS17walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS17walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS17walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X17 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS18walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS18walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS18walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS18walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS18walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X18 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS19walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS19walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS19walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS19walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS19walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS19walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X19 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS20walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS20walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS20walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS20walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS20walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X20 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS21walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS21walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,8):
    filename = 'WBDSascii\WBDS21walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS21walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS21walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X21 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS22walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS22walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,8):
    filename = 'WBDSascii\WBDS22walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS22walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS22walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X22 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS23walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS23walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

filename = 'WBDSascii\WBDS23walkO10Smkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,3):
    filename = 'WBDSascii\WBDS23walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(7,9):
    filename = 'WBDSascii\WBDS23walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))
    
x3 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS23walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x3 = np.dstack((x3,x_))

xF = np.dstack((x1,x2, x3))

X23 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS24walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS24walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS24walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS24walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS24walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xF = np.dstack((x1,x2))

X24 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS25walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,20):
    filename = 'WBDSascii\WBDS25walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,8):
    filename = 'WBDSascii\WBDS25walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS25walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS25walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X25 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS26walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,18):
    filename = 'WBDSascii\WBDS26walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS26walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS26walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS26walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(11,16):
    filename = 'WBDSascii\WBDS26walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X26 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS27walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS27walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS27walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS27walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS27walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS27walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X27 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS28walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS28walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS28walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS28walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS28walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS28walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X28 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS29walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS29walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS29walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS29walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS29walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X29 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS30walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS30walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS30walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,21):
    filename = 'WBDSascii\WBDS30walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS30walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS30walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X30 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS31walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS31walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS31walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,20):
    filename = 'WBDSascii\WBDS31walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS31walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xF = np.dstack((x1,x2))

X31 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS32walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS32walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS32walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS32walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS32walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS32walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X32 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS33walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS33walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS33walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

filename = 'WBDSascii\WBDS33walkO10Smkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS33walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS33walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X33 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS34walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS34walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS34walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS34walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

filename = 'WBDSascii\WBDS34walkO10Fmkr.txt'
df_mkr = pd.read_csv(filename,delimiter="\t")
x_ = df_mkr.iloc[:, 1:].values
x_ = x_[0:250,:]
x_ = imputer.fit_transform(x_)
x_ = pca.fit_transform(x_)
x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X34 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS35walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS35walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS35walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS35walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS35walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS35walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X35 = np.dstack((xS,xC,xF))

###############################################################################

#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,10):
#    filename = 'WBDSascii\WBDS36walkO0%sCmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(10,18):
#    filename = 'WBDSascii\WBDS36walkO%sCmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x2 = np.dstack((x2,x_))
#
#xC = np.dstack((x1,x2))
#
#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,9):
#    filename = 'WBDSascii\WBDS36walkO0%sSmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#xS = np.dstack((x1,x2))
#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,10):
#    filename = 'WBDSascii\WBDS36walkO0%sFmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(10,13):
#    filename = 'WBDSascii\WBDS36walkO%sFmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x2 = np.dstack((x2,x_))
#
#xF = np.dstack((x1,x2))
#
#X36 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS37walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS37walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS37walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS37walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,13):
    filename = 'WBDSascii\WBDS37walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xF = np.dstack((x1,x2))

X37 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS38walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,16):
    filename = 'WBDSascii\WBDS38walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS38walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,17):
    filename = 'WBDSascii\WBDS38walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,9):
    filename = 'WBDSascii\WBDS38walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(11,13):
    filename = 'WBDSascii\WBDS38walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

x3 = [[[] for i in range(9)] for j in range(250)]

for i in range(14,21):
    filename = 'WBDSascii\WBDS38walkO%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x3 = np.dstack((x3,x_))
    
xF = np.dstack((x1,x2,x3))

X38 = np.dstack((xS,xC,xF))

###############################################################################

#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,10):
#    filename = 'WBDSascii\WBDS39walkO0%sCmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(10,13):
#    filename = 'WBDSascii\WBDS39walkO%sCmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x2 = np.dstack((x2,x_))
#
#xC = np.dstack((x1,x2))
#
#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,10):
#    filename = 'WBDSascii\WBDS39walkO0%sSmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
##    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
##    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(10,19):
#    filename = 'WBDSascii\WBDS39walkO%sSmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x2 = np.dstack((x2,x_))
#
#xS = np.dstack((x1,x2))
#x1 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(1,10):
#    filename = 'WBDSascii\WBDS39walkO0%sFmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x1 = np.dstack((x1,x_))
#
#x2 = [[[] for i in range(9)] for j in range(250)]
#
#for i in range(10,14):
#    filename = 'WBDSascii\WBDS39walkO%sFmkr.txt' % str(i)
#    df_mkr = pd.read_csv(filename,delimiter="\t")
#    x_ = df_mkr.iloc[:, 1:].values
#    x_ = x_[0:250,:]
#    imputer = impute.SimpleImputer(missing_values=np.nan, strategy='median')
#    x_ = imputer.fit_transform(x_)
#    x_ = pca.fit_transform(x_)
#    x2 = np.dstack((x2,x_))
#
#xF = np.dstack((x1,x2))
#
#X39 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS40walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS40walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,14):
    filename = 'WBDSascii\WBDS40walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,5):
    filename = 'WBDSascii\WBDS40walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(6,7):
    filename = 'WBDSascii\WBDS40walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

x3 = [[[] for i in range(9)] for j in range(250)]

for i in range(8,10):
    filename = 'WBDSascii\WBDS40walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x3 = np.dstack((x3,x_))
    
xF = np.dstack((x1,x2,x3))

X40 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS41walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,11):
    filename = 'WBDSascii\WBDS41walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))
    
xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS41walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS41walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,8):
    filename = 'WBDSascii\WBDS41walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]
    
xF = np.dstack((x1,x2))

X41 = np.dstack((xS,xC,xF))

###############################################################################

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS42walkO0%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,12):
    filename = 'WBDSascii\WBDS42walkO%sCmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xC = np.dstack((x1,x2))

x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS42walkO0%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

for i in range(10,15):
    filename = 'WBDSascii\WBDS42walkO%sSmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x2 = np.dstack((x2,x_))

xS = np.dstack((x1,x2))
x1 = [[[] for i in range(9)] for j in range(250)]

for i in range(1,10):
    filename = 'WBDSascii\WBDS42walkO0%sFmkr.txt' % str(i)
    df_mkr = pd.read_csv(filename,delimiter="\t")
    x_ = df_mkr.iloc[:, 1:].values
    x_ = x_[0:250,:]
    x_ = imputer.fit_transform(x_)
    x_ = pca.fit_transform(x_)
    x1 = np.dstack((x1,x_))

x2 = [[[] for i in range(9)] for j in range(250)]

xF = np.dstack((x1,x2))

X42 = np.dstack((xS,xC,xF))

###############################################################################

# nan values 3rd metafile

Y1 = np.concatenate((np.zeros(11),np.ones(17),2*np.ones(15)))
Y2 = np.concatenate((np.zeros(13),np.ones(21),2*np.ones(22)))
Y3 = np.concatenate((np.zeros(14),np.ones(8),2*np.ones(13)))
Y4 = np.concatenate((np.zeros(14),np.ones(12),2*np.ones(14)))
Y5 = np.concatenate((np.zeros(10),np.ones(14),2*np.ones(16)))
Y6 = np.concatenate((np.zeros(10),np.ones(14),2*np.ones(14)))
Y7 = np.concatenate((np.zeros(17),np.ones(9),2*np.ones(10)))
Y8 = np.concatenate((np.zeros(6),np.ones(12),2*np.ones(23)))
Y9 = np.concatenate((np.zeros(15),np.ones(7),2*np.ones(15)))
Y10 = np.concatenate((np.zeros(13),np.ones(16),2*np.ones(12)))
Y11 = np.concatenate((np.zeros(11),np.ones(18),2*np.ones(15)))
Y12 = np.concatenate((np.zeros(18),np.ones(10),2*np.ones(9)))
Y13 = np.concatenate((np.zeros(12),np.ones(6),2*np.ones(14)))
Y14 = np.concatenate((np.zeros(17),np.ones(10),2*np.ones(11)))
Y15 = np.concatenate((np.zeros(9),np.ones(20),2*np.ones(14)))
Y16 = np.concatenate((np.zeros(16),np.ones(10),2*np.ones(14)))
Y17 = np.concatenate((np.zeros(15),np.ones(16),2*np.ones(16)))
Y18 = np.concatenate((np.zeros(12),np.ones(9),2*np.ones(14)))
Y19 = np.concatenate((np.zeros(11),np.ones(11),2*np.ones(15)))
Y20 = np.concatenate((np.zeros(13),np.ones(9),2*np.ones(11)))
Y21 = np.concatenate((np.zeros(7),np.ones(15),2*np.ones(17)))
Y22 = np.concatenate((np.zeros(7),np.ones(14),2*np.ones(17)))
Y23 = np.concatenate((np.zeros(10),np.ones(9),2*np.ones(7)))
Y24 = np.concatenate((np.zeros(14),np.ones(14),2*np.ones(9)))
Y25 = np.concatenate((np.zeros(7),np.ones(19),2*np.ones(15)))
Y26 = np.concatenate((np.zeros(11),np.ones(17),2*np.ones(14)))
Y27 = np.concatenate((np.zeros(13),np.ones(12),2*np.ones(12)))
Y28 = np.concatenate((np.zeros(13),np.ones(13),2*np.ones(13)))
Y29 = np.concatenate((np.zeros(9),np.ones(16),2*np.ones(11)))
Y30 = np.concatenate((np.zeros(20),np.ones(14),2*np.ones(11)))
Y31 = np.concatenate((np.zeros(19),np.ones(11),2*np.ones(9)))
Y32 = np.concatenate((np.zeros(15),np.ones(11),2*np.ones(15)))
Y33 = np.concatenate((np.zeros(10),np.ones(14),2*np.ones(13)))
Y34 = np.concatenate((np.zeros(13),np.ones(9),2*np.ones(10)))
Y35 = np.concatenate((np.zeros(14),np.ones(13),2*np.ones(13)))
#Y36 = np.concatenate((np.zeros(8),np.ones(17),2*np.ones(12)))
Y37 = np.concatenate((np.zeros(16),np.ones(9),2*np.ones(12)))
Y38 = np.concatenate((np.zeros(16),np.ones(15),2*np.ones(17)))
#Y39 = np.concatenate((np.zeros(18),np.ones(12),2*np.ones(13)))
Y40 = np.concatenate((np.zeros(13),np.ones(9),2*np.ones(7)))
Y41 = np.concatenate((np.zeros(11),np.ones(10),2*np.ones(7)))
Y42 = np.concatenate((np.zeros(14),np.ones(11),2*np.ones(9)))

###############################################################################

X1 = tf.keras.utils.normalize(X1,axis=-1,order=2)
X1 = X1.transpose(2,0,1)
X2 = tf.keras.utils.normalize(X2,axis=-1,order=2)
X2 = X2.transpose(2,0,1)
X3 = tf.keras.utils.normalize(X3,axis=-1,order=2)
X3 = X3.transpose(2,0,1)
X4 = tf.keras.utils.normalize(X4,axis=-1,order=2)
X4 = X4.transpose(2,0,1)
X5 = tf.keras.utils.normalize(X5,axis=-1,order=2)
X5 = X5.transpose(2,0,1)
X6 = tf.keras.utils.normalize(X6,axis=-1,order=2)
X6 = X6.transpose(2,0,1)
X7 = tf.keras.utils.normalize(X7,axis=-1,order=2)
X7 = X7.transpose(2,0,1)
X8 = tf.keras.utils.normalize(X8,axis=-1,order=2)
X8 = X8.transpose(2,0,1)
X9 = tf.keras.utils.normalize(X9,axis=-1,order=2)
X9 = X9.transpose(2,0,1)
X10 = tf.keras.utils.normalize(X10,axis=-1,order=2)
X10 = X10.transpose(2,0,1)
X11 = tf.keras.utils.normalize(X11,axis=-1,order=2)
X11 = X11.transpose(2,0,1)
X12 = tf.keras.utils.normalize(X12,axis=-1,order=2)
X12 = X12.transpose(2,0,1)
X13 = tf.keras.utils.normalize(X13,axis=-1,order=2)
X13 = X13.transpose(2,0,1)
X14 = tf.keras.utils.normalize(X14,axis=-1,order=2)
X14 = X14.transpose(2,0,1)
X15 = tf.keras.utils.normalize(X15,axis=-1,order=2)
X15 = X15.transpose(2,0,1)
X16 = tf.keras.utils.normalize(X16,axis=-1,order=2)
X16 = X16.transpose(2,0,1)
X17 = tf.keras.utils.normalize(X17,axis=-1,order=2)
X17 = X17.transpose(2,0,1)
X18 = tf.keras.utils.normalize(X18,axis=-1,order=2)
X18 = X18.transpose(2,0,1)
X19 = tf.keras.utils.normalize(X19,axis=-1,order=2)
X19 = X19.transpose(2,0,1)
X20 = tf.keras.utils.normalize(X20,axis=-1,order=2)
X20 = X20.transpose(2,0,1)
X21 = tf.keras.utils.normalize(X21,axis=-1,order=2)
X21 = X21.transpose(2,0,1)
X22 = tf.keras.utils.normalize(X22,axis=-1,order=2)
X22 = X22.transpose(2,0,1)
X23 = tf.keras.utils.normalize(X23,axis=-1,order=2)
X23 = X23.transpose(2,0,1)
X24 = tf.keras.utils.normalize(X24,axis=-1,order=2)
X24 = X24.transpose(2,0,1)
X25 = tf.keras.utils.normalize(X25,axis=-1,order=2)
X25 = X25.transpose(2,0,1)
X26 = tf.keras.utils.normalize(X26,axis=-1,order=2)
X26 = X26.transpose(2,0,1)
X27 = tf.keras.utils.normalize(X27,axis=-1,order=2)
X27 = X27.transpose(2,0,1)
X28 = tf.keras.utils.normalize(X28,axis=-1,order=2)
X28 = X28.transpose(2,0,1)
X29 = tf.keras.utils.normalize(X29,axis=-1,order=2)
X29 = X29.transpose(2,0,1)
X30 = tf.keras.utils.normalize(X30,axis=-1,order=2)
X30 = X30.transpose(2,0,1)
X31 = tf.keras.utils.normalize(X31,axis=-1,order=2)
X31 = X31.transpose(2,0,1)
X32 = tf.keras.utils.normalize(X32,axis=-1,order=2)
X32 = X32.transpose(2,0,1)
X33 = tf.keras.utils.normalize(X33,axis=-1,order=2)
X33 = X33.transpose(2,0,1)
X34 = tf.keras.utils.normalize(X34,axis=-1,order=2)
X34 = X34.transpose(2,0,1)
X35 = tf.keras.utils.normalize(X35,axis=-1,order=2)
X35 = X35.transpose(2,0,1)
#X36 = tf.keras.utils.normalize(X36,axis=-1,order=2)
#X36 = X36.transpose(2,0,1)
X37 = tf.keras.utils.normalize(X37,axis=-1,order=2)
X37 = X37.transpose(2,0,1)
X38 = tf.keras.utils.normalize(X38,axis=-1,order=2)
X38 = X38.transpose(2,0,1)
#X39 = tf.keras.utils.normalize(X39,axis=-1,order=2)
#X39 = X39.transpose(2,0,1)
X40 = tf.keras.utils.normalize(X40,axis=-1,order=2)
X40 = X40.transpose(2,0,1)
X41 = tf.keras.utils.normalize(X41,axis=-1,order=2)
X41 = X41.transpose(2,0,1)
X42 = tf.keras.utils.normalize(X42,axis=-1,order=2)
X42 = X42.transpose(2,0,1)

###############################################################################

X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X37, X38, X40, X41, X42), axis= 0)
X = X.reshape(X.shape[0], 250, 9, 1)

###############################################################################

Y1 = to_categorical(Y1)
Y2 = to_categorical(Y2)
Y3 = to_categorical(Y3)
Y4 = to_categorical(Y4)
Y5 = to_categorical(Y5)
Y6 = to_categorical(Y6)
Y7 = to_categorical(Y7)
Y8 = to_categorical(Y8)
Y9 = to_categorical(Y9)
Y10 = to_categorical(Y10)
Y11 = to_categorical(Y11)
Y12 = to_categorical(Y12)
Y13 = to_categorical(Y13)
Y14 = to_categorical(Y14)
Y15 = to_categorical(Y15)
Y16 = to_categorical(Y16)
Y17 = to_categorical(Y17)
Y18 = to_categorical(Y18)
Y19 = to_categorical(Y19)
Y20 = to_categorical(Y20)
Y21 = to_categorical(Y21)
Y22 = to_categorical(Y22)
Y23 = to_categorical(Y23)
Y24 = to_categorical(Y24)
Y25 = to_categorical(Y25)
Y26 = to_categorical(Y26)
Y27 = to_categorical(Y27)
Y28 = to_categorical(Y28)
Y29 = to_categorical(Y29)
Y30 = to_categorical(Y30)
Y31 = to_categorical(Y31)
Y32 = to_categorical(Y32)
Y33 = to_categorical(Y33)
Y34 = to_categorical(Y34)
Y35 = to_categorical(Y35)
#Y36 = to_categorical(Y36)
Y37 = to_categorical(Y37)
Y38 = to_categorical(Y38)
#Y39 = to_categorical(Y39)
Y40 = to_categorical(Y40)
Y41 = to_categorical(Y41)
Y42 = to_categorical(Y42)

Y = np.vstack((Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15, Y16, Y17, Y18, Y19, Y20, Y21, Y22, Y23, Y24, Y25, Y26, Y27, Y28, Y29, Y30, Y31, Y32, Y33, Y34, Y35, Y37, Y38, Y40, Y41, Y42))
