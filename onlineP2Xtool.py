# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:34:45 2022

@author: Tomás Wierzba
"""
pip install numpy_financial as npf
import streamlit as st
import requests
import json
import pandas as pd
import numpy_financial as npf
import numpy as np
import math
import altair as alt



#OUTPUTS VANILLA VERSION
#Optimal size of electrolyzer 
#Optimal Operational mode
#LCoH
#IRR

st.write(""" # Welcome to Hybrid Greentech's Power-to-X Sizing Platform""")
st.write("This simple model provides valuable information for investors assesing to participate in the P2X sector. In order to give a general understanding, this model assumes a 1 MW electrolyzer, i.e. CAPEX reduction due to economics of scale are not considered. Other assumptions are that the SOEC produces 23.3 kg of Hydrogen per MWh of electricity and that the stacks replacement capital cost is 20% of the CAPEX. The main variables for this business case-study can be changed in the left pane.")

#Explain assumptions here

electrolyser_nom_cap = 1000 #kW

st.sidebar.write(""" # Key variables""")

#Decide averageelectricity spot price
Electricity_spot_MWh = st.sidebar.slider('What is the average electricity spot price in €/MWh? ', 0, 200, 60)
Electricity_spot = Electricity_spot_MWh/1000

#Decide Specific capex
electrolyser_specific_invest= st.sidebar.slider('What is the electrolyzer capital investment in €/kW? ', 0, 5000, 3050,50)

#Decide OPEX % of CAPEX
electrolyser_OPEX_percentage2= st.sidebar.slider('What is the % CAPEX spent in OPEX yearly? ', 0, 20, 5)
electrolyser_OPEX_percentage = electrolyser_OPEX_percentage2/100

#Decide how many full-load hours of operation will the electrolyzer run in a year
full_load_hours= st.sidebar.slider('Full-load hours of operation in a year: ', 0, 8760, 7500,100)

#Decide technical lifetime of stack
technical_lifetime_stacks= st.sidebar.slider('Technical lifetime of stacks in full-load hours of operation: ', 0, 150000,20000 ,1000)

#Decide future Hydrogen price
Hydrogen_price = st.sidebar.slider('What is the future average Hydrogen sale price in €/kg?', 0, 15,6)

#Decide future Hydrogen price
lifetime = st.sidebar.slider('What is the project lifetime in years?', 0, 30,25)



# Recommended values from https://ens.dk/sites/ens.dk/files/Analyser/technology_data_for_renewable_fuels.pdf pg. 107 & 115

H2_SOEC_input = 0.0233     # Kg / kWh input
SOEC_STACK_replacement   = 0.2       # 20% of SOEC CAPEX according to Ceres Power


H2_electrolyser_input = H2_SOEC_input
electrolyser_STACK_replacement = SOEC_STACK_replacement
  

#-----------------------------------OPEX---------------------------------------------------------------------------------------
OPEX_electrolyser_yearly = electrolyser_OPEX_percentage * electrolyser_specific_invest * electrolyser_nom_cap #€/year this value must be changed when changing electrolyser specific investment
Electricity_cost_yearly  = Electricity_spot * full_load_hours * electrolyser_nom_cap
OPEX_yearly              = OPEX_electrolyser_yearly + Electricity_cost_yearly #€/year this is the yearly OPEX
#Replacement cost for stacks when technical lifetime is achieved included in cash flow
#------------------------------------CAPEX--------------------------------------------------------------------------------------------
CAPEX_electrolyser      = electrolyser_nom_cap * electrolyser_specific_invest #€ this value must be changed when changing electrolyser specific investment
CAPEX                   = CAPEX_electrolyser 
#------------------------------------Income-----------------------------------------------------------------------------------------
Hydrogen_production_yearly = np.zeros(lifetime +1)
for t in range(1,lifetime+1):
    Hydrogen_production_yearly[t] = H2_SOEC_input * full_load_hours * electrolyser_nom_cap
Hydrogen_income_yearly = np.zeros(lifetime +1)
for t in range(1,lifetime+1):
    Hydrogen_income_yearly[t] = Hydrogen_price * Hydrogen_production_yearly[t] #€/year
#------------------------------------CashFlow-----------------------------------------------------------------------------------------
cf = np.zeros(lifetime+1) #cashflow in M€
for t in range (1, len(cf)):
    cf[t] = (- OPEX_yearly + Hydrogen_income_yearly[t])/1e+6
cf[0] = -CAPEX/1e+6
electrolyser_exact_replacement_period = technical_lifetime_stacks/full_load_hours #years
years_of_stack_replacement = np.zeros(lifetime+1)
years_of_stack_replacement[0]=0
for i in range(1, lifetime+1):
    if math.ceil(electrolyser_exact_replacement_period*i)<=lifetime:
        years_of_stack_replacement[math.ceil(electrolyser_exact_replacement_period*i)]=1
for t in range (1,len(cf)):
    cf[t] = -CAPEX_electrolyser * electrolyser_STACK_replacement * years_of_stack_replacement[t] /1e+6 + (- OPEX_yearly + Hydrogen_income_yearly[t])/1e+6
year=np.linspace(0, lifetime,lifetime+1)
chart_data2 = pd.DataFrame({'Year':year,'Cash Flow in Million €':cf})
d = alt.Chart(chart_data2).mark_bar().encode(
     x='Year:O',y='Cash Flow in Million €',color=alt.value('#ffe300')).configure(background='#142330')
st.altair_chart(d, use_container_width=True)
#------------------------------------NPV--------------------------------------------------------------------------------------
discountRate    = 0.05; # Five percent per annum
discountRate2 = round(discountRate*100,1)
npv             = npf.npv(discountRate, cf)
npv2 = round(npv,2)
NPV = np.zeros(len(cf))
for i in range(0,len(cf)):
    NPV[i] = npf.npv(discountRate, cf[0:(i+1)])
for i in range(1,len(cf)):
    if NPV[i]>=0:
        if NPV[i-1]<=0:
            a1010=i
if all(e <= 0 for e in NPV):
    st.write('Project is not profitable in 27 years')
else:
    st.write('Payback time is approximately', a1010 ,'years')
chart_data3 = pd.DataFrame({'Year':year,'Net Present value in Million €':NPV})
c = alt.Chart(chart_data3).mark_bar().encode(
     x='Year:O',y='Net Present value in Million €',color=alt.value('#ffe300')).configure(background='#142330')
st.altair_chart(c, use_container_width=True)

st.write('Net present value of the investment considering a ',discountRate2,'% discount rate is: ', npv2, 'M€')
#------------------------------------IRR---------------------------------------------------------------------------------------------
IRR = npf.irr(cf)
IRR2 = round(IRR*100,1)
st.write('The IRR is ',IRR2,'%')
#Hydrogen Price independent
#------------------------------------LCoH-----------------------------------------------------------------------------------
Expenses = np.zeros(lifetime +1) #expenses plus electricity income in €/year
for t in range(1,len(cf)):
    Expenses[t] = -cf[t]*1e+6 + (Hydrogen_income_yearly[t])
Expenses[0] = -cf[0]*1e+6
LCoH = npf.npv(discountRate,Expenses)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH2 = round(LCoH,1)
st.write('The levelized cost of Hydrogen, considering a ',discountRate2,'% discount rate, is: ',LCoH2,'€/kg')

#------------------------------------LCoH per expense-----------------------------------------------------------------
OPEX_electrolyser_yearly_v = np.zeros(lifetime+1)
OPEX_electrolyser_yearly_v[0] = 0
for i in range(1,lifetime+1):
    OPEX_electrolyser_yearly_v[i] = OPEX_electrolyser_yearly
LCoH_opex_electrolyser = npf.npv(discountRate,OPEX_electrolyser_yearly_v)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH_opex_electrolyser2 = round(LCoH_opex_electrolyser,1)


Electricity_cost_yearly_v = np.zeros(lifetime+1)
Electricity_cost_yearly_v[0] = 0
for i in range(1,lifetime+1):
    Electricity_cost_yearly_v[i] = Electricity_cost_yearly
LCoH_electricity_cost = npf.npv(discountRate,Electricity_cost_yearly_v)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH_electricity_cost2 = round(LCoH_electricity_cost,1)


CAPEX_v = np.zeros(lifetime+1)
CAPEX_v[0] = CAPEX
LCoH_capex = npf.npv(discountRate,CAPEX_v)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH_capex2 = round(LCoH_capex,1)


stack_replacement_cost_v = np.zeros(lifetime+1)
stack_replacement_cost_v[0] = 0
for t in range(1,len(cf)):
    stack_replacement_cost_v[t] = CAPEX_electrolyser * electrolyser_STACK_replacement * years_of_stack_replacement[t]
LCoH_stack_rep_cost = npf.npv(discountRate,stack_replacement_cost_v)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH_stack_rep_cost2 = round(LCoH_stack_rep_cost,1)


source = pd.DataFrame({"Values": [LCoH_electricity_cost2,LCoH_capex2,LCoH_stack_rep_cost2, LCoH_opex_electrolyser2],"Cost contribution": ['Electricity','CAPEX','Stack Replacement','OPEX Electrolyzer']})

base = alt.Chart(source).encode(
    theta=alt.Theta("Values:Q", stack=True),
    radius=alt.Radius("Values", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
    color='Cost contribution:N'
)

c1 = base.mark_arc(innerRadius=20, stroke="#fff")

c2 = base.mark_text(radiusOffset=10).encode(text="Values:Q")
#c2 = base.mark_text(radiusOffset=10, align='left',
 #   baseline='middle').encode(text="Values:Q", color=alt.value('white'))

c1 + c2

