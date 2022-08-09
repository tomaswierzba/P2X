# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:34:45 2022

@author: Tomás Wierzba
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import numpy_financial as npf
import math
import altair as alt



#OUTPUTS VANILLA VERSION
#Optimal size of electrolyzer 
#Optimal Operational mode
#LCoH
#IRR

st.write(""" # Welcome to Hybrid Greentech's user-friendly SOEC Business-case evaluation tool""")
st.write("The simple model employed in this platform provides valuable information for investors assesing to participate in the P2X sector. In order to give a general understanding, this model assumes initially a 1 MW electrolyzer. The main variables for this business case-study can be changed in the left pane and their initial values repesent SOEC technology.")

#Explain assumptions here

electrolyser_nom_cap = 1000 #kW

st.sidebar.write(""" # Key variables""")

#Decide average electricity spot price
Electricity_spot_MWh = st.sidebar.slider('What is the average electricity spot price in €/MWh? ', 0, 200, 60)
Electricity_spot = Electricity_spot_MWh/1000

#Decide power-to-hydrogen production ratio
H2_electrolyser_input_1000 = st.sidebar.slider('What is the power-to-Hydrogen production ratio kg/MWh? ', 10.0, 30.0, 23.3,0.1)
H2_electrolyser_input = H2_electrolyser_input_1000/1000

#Decide Specific capex
electrolyser_specific_invest= st.sidebar.slider('What is the electrolyzer capital investment in €/kW? ', 0, 5000, 3050,50)

#Decide OPEX % of CAPEX
electrolyser_OPEX_percentage2= st.sidebar.slider('What is the % CAPEX spent in O&M yearly? ', 0, 20, 5)
electrolyser_OPEX_percentage = electrolyser_OPEX_percentage2/100

#Decide how many full-load hours of operation will the electrolyzer run in a year
full_load_hours= st.sidebar.slider('Full-load hours of operation in a year: ', 0, 8760, 7500,100)

#Decide technical lifetime of stack
technical_lifetime_stacks= st.sidebar.slider('Technical lifetime of stacks in full-load hours of operation: ', 0, 150000,20000 ,1000)

#Decide stack replacement cost
electrolyser_STACK_replacement_100 = st.sidebar.slider('Stack replacement cost as % of CAPEX: ', 0, 100,20 ,1)
electrolyser_STACK_replacement = electrolyser_STACK_replacement_100/100

#Decide future Hydrogen price
Hydrogen_price = st.sidebar.slider('What is the future average Hydrogen sales price in €/kg?', 0, 15,6)

#Decide future Hydrogen price
lifetime = st.sidebar.slider('What is the project lifetime in years?', 0, 30,25)

#Decide discount rate
discountRate_100 = st.sidebar.slider('What is the desired discount rate?', 0, 50, 5,1)
discountRate = discountRate_100/100

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
    Hydrogen_production_yearly[t] = H2_electrolyser_input * full_load_hours * electrolyser_nom_cap
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
chart_data2 = pd.DataFrame({'Year':year,'Non-discounted Cash Flow in Million €':cf})
d = alt.Chart(chart_data2).mark_bar().encode(
     x='Year:O',y='Non-discounted Cash Flow in Million €:Q',color=alt.value('#ffe300')).configure(background='#193047').properties(
    title='Non-discounted Cash Flows'
)
d.configure_title(
    fontSize=20,
    font='Courier',
    anchor='start',
)
st.altair_chart(d, use_container_width=True)
#------------------------------------NPV--------------------------------------------------------------------------------------
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
    st.write('Payback time is approximately %s years' % (a1010))
chart_data3 = pd.DataFrame({'Year':year,'Accumulated Discounted Cash Flows in Million €':NPV})
c = alt.Chart(chart_data3).mark_bar().encode(
     x='Year:O',y='Accumulated Discounted Cash Flows in Million €',color=alt.value('#ffe300')).configure(background='#193047').properties(
    title='Accumulated Discounted Cash Flows'
)
d.configure_title(
    fontSize=20,
    font='Courier',
    anchor='start',
)
st.altair_chart(c, use_container_width=True)

st.write('Net present value: %s M€ (%s %% discount rate)' % (npv2,discountRate2))
#------------------------------------IRR---------------------------------------------------------------------------------------------
IRR = npf.irr(cf)
IRR2 = round(IRR*100,1)
st.write('IRR: %s %%' % (IRR2))
#Hydrogen Price independent
#------------------------------------LCoH-----------------------------------------------------------------------------------
Expenses = np.zeros(lifetime +1) #expenses plus electricity income in €/year
for t in range(1,len(cf)):
    Expenses[t] = -cf[t]*1e+6 + (Hydrogen_income_yearly[t])
Expenses[0] = -cf[0]*1e+6
LCoH = npf.npv(discountRate,Expenses)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH2 = round(LCoH,1)
st.write('Levelized Cost of Hydrogen: %s €/kg (%s %% discount rate)' % (LCoH2,discountRate2))

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

st.write("The radial plot below shows the levelised cost contributions of the main cost-drivers for Hydrogen:")
source = pd.DataFrame({"Values": [LCoH_electricity_cost2,LCoH_capex2,LCoH_stack_rep_cost2, LCoH_opex_electrolyser2],"Cost contribution": ['Electricity','CAPEX','Stack Replacement','O&M Electrolyzer'],"labels":["%s €/kg" % (LCoH_electricity_cost2),"%s €/kg" % (LCoH_capex2),"%s €/kg" % (LCoH_stack_rep_cost2),"%s €/kg" % (LCoH_opex_electrolyser2)]})
domain = ['Electricity','CAPEX','Stack Replacement','O&M Electrolyzer']
range_ = ['#142330', '#194366', '#f0f2f6', '#ffe300']
base = alt.Chart(source).encode(
    theta=alt.Theta("Values:Q", stack=True), color=alt.Color('Cost contribution:N', scale=alt.Scale(domain=domain, range=range_)),
    radius=alt.Radius("Values:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
)

c1 = base.mark_arc(innerRadius=20)

c2 = base.mark_text(radiusOffset=20).encode(text="labels:N")
#c2 = base.mark_text(radiusOffset=10, align='left',
 #   baseline='middle').encode(text="Values:Q", color=alt.value('white'))

c1 + c2

data = {
'Electricity':LCoH_electricity_cost2,'CAPEX':LCoH_capex2,'Stack Replacement':LCoH_stack_rep_cost2, 'OPEX':LCoH_opex_electrolyser2
}
a20 = max(data, key=data.get)
per_main_costdriver = round(data[a20] / LCoH * 100 , 1)
st.write("The main cost-driver for the Levelized Cost of Hydrogen is found to be %s, accounting for %s %% of the cost." % (a20, per_main_costdriver))
st.write("What's your next action towards 100% renewables?")
st.write("Let's create more value together, send us an e-mail to info@hybridgreentech.com")
