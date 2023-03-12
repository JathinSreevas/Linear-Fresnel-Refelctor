#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('view.csv')


# In[3]:


df


# ## Data Cleaning

# In[4]:


df.drop(df.loc[:, 'Latitude':'Wind Speed Units'].columns, axis=1, inplace=True)
#Dropping unnecessary columns


# In[5]:


df.dtypes
#Checking if the columns are iun the ideal type or not


# In[6]:


df.rename(columns={'Source':'Hour', 'Location ID':'GHI', 'City':'DHI', 'State':'Temperature', 'Country':'Wind Speed'}, inplace=True)
#Renaming the columns to our need


# In[7]:


df.drop(df.index[:3], inplace=True)
df = df.reset_index(drop=True)
#Dropping first three unnecessary rows and resetting the index


# In[8]:


for i in df.columns:
    df[i]=df[i].astype('float32')
#Changing the data types to float to ease calculations in the future


# In[9]:


df


# ## Finding BHI

# In[10]:


df['BHI']=df['GHI']-df['DHI']
#Finding beam radiation I_b from I_g and I_d


# In[11]:


Ib=df['BHI'].to_numpy()
Id=df['DHI'].to_numpy()
Ig=df['GHI'].to_numpy()
#Creating arrays as numpy arrays are computationally faster than pandas


# ## Finding I_sc'

# In[12]:


day=np.arange(1,366)
df['day']=np.repeat(day,24)
#Creating the days of an year and saving them to the dataframe


# In[13]:


I_sc_day = 1367*(1+0.033*np.cos(np.deg2rad(360*day/365)))
#Finding the daywise solar irradiance


# In[14]:


I_sc_day=np.repeat(I_sc_day, 24)
df['I_sc\'']=I_sc_day
#As solar irradiance is constant for a day, we repeat the value for 24 hours and save it to the dataframe


# ## Finding Solar Zenith Angle and incidence angle

# θ = f(β, γ, δ, ω, φ)<br>
# θz = f(β=0, γ, δ, ω, φ)<br>
# where β=collector tilt angle, γ=surface azimuth angle, δ=declination angle, ω=hour angle, φ=latitude<br>
# Latitude is given as 26.35<br>Assuming that the panels are facing south<br>Assuming an average value of tilt angle (for now)

# In[15]:


φ=26+35/60
γ=0
β=15


# ### Declination Angle

# In[16]:


δ=23.45*np.sin(np.deg2rad(360*(284+day)/365))
#Finding the declination angle using the formula


# In[17]:


δ=np.repeat(δ, 24)
df['δ']=δ
#As solar declination angle is constant for a day, we repeat the value for 24 hours and save it to the dataframe


# In[18]:


hr = np.arange(0,24*365)%24+1
df['daywise_hours'] = hr
#We would want daywise hours from 1-24 to be clear about our daywise radiation


# ### Hour Angle

# In[19]:


E=[]
for i in day:
    B=(i-1)*360/365
    E.append(229.18*(0.000075+0.001868*np.cos(np.deg2rad(B))-0.032077*np.sin(np.deg2rad(B))-0.014615*np.cos(np.deg2rad(2*B))-0.04089*np.sin(np.deg2rad(2*B))))
E=np.repeat(E,24)
df['Error in min']=E
#Finding daywise error in time and saving it to the dataframe


# In[20]:


LAT=[]
for i,j in enumerate(hr):
    LAT.append(j*60-37.8+E[i])
LAT=np.asarray(LAT)
df['LAT_minutes']=LAT
#Finding LAT in minutes and saving it to the dataframe


# In[21]:


df['LAT']=pd.to_datetime(df.LAT_minutes, unit='m').dt.strftime('%H:%M')
#Finding LAT in time format and saving it to the dataframe


# In[22]:


ω=(LAT-720)/60*15
df['ω']=ω
#Finding hour angle in time format and saving it to the dataframe


# In[23]:


df


# In[24]:


def value(β):
    first = np.sin(np.deg2rad(β))*np.cos(np.deg2rad(γ))*(np.cos(np.deg2rad(δ))*np.cos(np.deg2rad(ω))*np.sin(np.deg2rad(φ))-np.sin(np.deg2rad(δ))*np.cos(np.deg2rad(φ)))
    second = np.sin(np.deg2rad(β))*np.sin(np.deg2rad(γ))*np.cos(np.deg2rad(δ))*np.sin(np.deg2rad(ω))
    third = np.cos(np.deg2rad(β))*(np.cos(np.deg2rad(δ))*np.cos(np.deg2rad(ω))*np.cos(np.deg2rad(φ))+np.sin(np.deg2rad(δ))*np.sin(np.deg2rad(φ)))                                                                                                    
    return first+second+third
#Formula for cosθ and cosθz with β varying


# In[25]:


cosθ=value(15)
df['cosθ']=cosθ
#Incidence angles and saving to the dataframe


# In[26]:


cosθz=value(0)
df['cosθz']=cosθz
#Solar zenith angles and saving to the dataframe


# ### Radiation on Tilted Surface - Isotropic Model (liu Jordan)

# In[27]:


rb=cosθ/cosθz
df['rb']=rb
#Finding the tilt factor for beam radiation and saving to the dataframe


# In[28]:


max(rb)
#Weird value noted meansd more data cleaning needed, will do after finding It


# In[29]:


rd=(1+np.cos(np.deg2rad(β)))/2
#Finding the tilt factor for diffused radiation


# In[30]:


rr=0.2*(1-np.cos(np.deg2rad(β)))/2
#Finding the tilt factor for reflected radiation


# In[31]:


It_LJ = np.multiply(Ib,rb) + Id*rd + Ig*rr
df['It_LJ']=It_LJ
#Finding the tilted radiation and saving to the dataframe


# ### Radiation on Tilted Surface - Anisotropic Model (HDKR model)

# In[32]:


Io=I_sc_day*cosθz


# In[33]:


Ai=df['BHI']/Io


# In[34]:


It_HDRK=(Ib+Id*Ai)*rb+Id*(1-Ai)*rd*(1+np.sqrt(np.divide(Ib, Ig, out=np.zeros_like(Ib), where=Ig!=0))*(np.sin(np.deg2rad(β/2)))**3)+Ig*rr


# In[35]:


df['It_HDRK']=It_HDRK


# In[36]:


df[:50]


# The difference between It_LJ and It_HDRK can be seen to be less than 10%

# In[37]:


df['Ibn']=df['BHI']/df['cosθz']
#Finding the beam normal radiation


# In[38]:


df


# ## More Data Cleaning

# In[39]:


df.sort_values(by='It_LJ', ascending=False)


# Clearly rb max value is 45025 and It max value is coming out to be 315218 W/m2 which is way out of bound, so we replace it with zero. Negative values of It are also replaced with zero

# In[40]:


It_LJ[It_LJ<0]=0
It_LJ[It_LJ>1100]=0
df['It_LJ']=It_LJ


# In[41]:


It_HDRK[It_HDRK<0]=0
It_HDRK[It_HDRK>1100]=0
df['It_HDRK']=It_HDRK


# I_t is cleaned now<br>
# Let's clean I_bn

# In[42]:


Ibn=np.asarray(df['Ibn'])
Ibn[Ibn<0]=0
df['Ibn']=Ibn


# In[43]:


df.sort_values(by='Ibn', ascending=False)


# In[44]:


df
#Final dataframe


# In[45]:


df.to_excel('data.xlsx')


# # Modelling

# In[46]:


from CoolProp.CoolProp import PhaseSI, PropsSI, get_global_param_string
import CoolProp.CoolProp as cp


# In[47]:


h1= cp.PropsSI('H', 'T',10+273,'P', 400000,'Water')
h2= cp.PropsSI('H', 'T',90+273,'P', 100000,'Water')
h4= cp.PropsSI('H', 'T',90+273,'P', 150000,'Water')
h5= cp.PropsSI('H', 'T',10+273,'P', 100000,'Water')


# In[48]:


T1=10+273
T2=90+273
T3=T2
T4=T2
T5=T1
T6=5+273
T7=T6
T8=85+273


# Assuming ideal heat exchanger <br>
# Using m2Cpw(T7-T6)=m3Cpm(T4-T5) we find m2

# In[49]:


from sympy import symbols, solve


# In[50]:


m = symbols('x')


# In[51]:


m3=20*1032


# In[52]:


expr=m*(h4-h5)-m3*3.93*(T8-T7)*1000


# In[53]:


print("The value of the volume flow rate of water in the heat exchanger in m3/hr is", solve(expr)[0]/1000)


# In[54]:


print("Useful energy needed for pasteurization is", m3*3.93*(T8-T7), "in kJ/hr")
print("Useful energy needed for pasteurization is", m3*3.93*(T8-T7)/(3600*1000), "in MW")


# We'll model for June 21st

# In[55]:


n=173


# In[56]:


dff=df.iloc[(n-1)*24:n*24]
#Selecting the values for June 21st


# In[57]:


dff


# At LAT close to 12 noon, 12:20PM, Ibn is maximum which is 687.3W/m2

# In[58]:


I_bn=687.3
A=19.373*1000*(h2-h1)/(I_bn*0.85*3600)
print(A)
# Returns area needed for the solar field for that time instant


# In[59]:


mw1=19.313*Ibn/I_bn
df['mw1']=mw1
#Finding the mass flow rae for all the other hours based on these values


# In[60]:


df


# In[61]:


SM=1.8
print("The area of solar field needed is", SM*A)
#Final area requirement for the whole solar field

