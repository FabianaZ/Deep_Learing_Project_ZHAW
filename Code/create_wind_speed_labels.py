# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:57:38 2022

@author: fabia
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime


###############################################################################
def is_dst(dt=None, timezone="UTC"):
    if dt is None:
        dt = datetime.utcnow()
    timezone = pytz.timezone(timezone)
    timezone_aware_date = timezone.localize(dt, is_dst=None)
    return timezone_aware_date.tzinfo._dst.seconds != 0
###############################################################################



filepath = 'D:\\DeepLearning\\500hPaGeopotential\\DWD_Windspeed\\final_dataset\\wind_speed_luebeck.csv'
destpath = 'D:\\DeepLearning\\500hPaGeopotential\\DWD_Windspeed\\final_dataset\\wind_speed_labels_luebeck.csv'

df = pd.read_csv(filepath,sep=';')

df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'],format = '%Y%m%d%H')

df.rename(columns={'MESS_DATUM':'TimeStamp',df.columns[3]:'WindSpeed'},inplace=True)

df.set_index('TimeStamp', inplace=True)

df.dropna(inplace=True)

# we are only interested in wind speed
df = df.iloc[:,2].to_frame()



# we need 00 UTC hour equivalent in Germany





df = df.loc[df.index.hour ==0]

# change to daily granularity (take max wind speed)
df = df.resample('D').mean() # not correct way to do it


# daylightsavings time
#df['dls'] =0


#dfnew = df.copy(deep=True)
#dfnew = dfnew[0:0]




#entgegen der Angabe im File scheint es zu Beginn des Zeithorizonts 1985 nicht MEZ-Daten zu sein

#for i in range(len(df)):
    
 #   print(i)
    
    #df['dls'].iloc[i] = is_dst(df.index[i],'Europe/Berlin')
    
  #  if df.index[i].hour == 0:
   #     print("added")
    #    dfnew.append(df[i,:], ignore_index = True)
    
    
    # Sommerzeit liefert einen True-Wert zurück
    
    #if df['dls'].iloc[i] == 1:
    #    print("summer time")
    #    print(df.index[i].strftime('%Y%m%D'))
    #    if df.index[i].hour == 2:
    #        dfnew.append(df[i,:], ignore_index = True)
            
    #if df['dls'].iloc[i] == 0:
    #    print("winter")
    #    if df.index[i].hour ==1:
    # dfnew.append(df.iloc[i], ignore_index = True)
    
    # Winterzeit liefert einen False-Wert zurück
    
median = df['WindSpeed'].quantile(0.5)
    

df['WindLabel'] = 0
df['WindLabel'].loc[df['WindSpeed']>=median] = 1

# save to csv file
df.to_csv(destpath)