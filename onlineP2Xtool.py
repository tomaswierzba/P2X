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


from IPython.display import Image
import base64, io, IPython
from PIL import Image 
image = Image.open('HG_Flat_hori.png')
image2 = Image.open('HG_Yellow_hori.png')


st.image(image, caption=None)

#new_title0 = '<p style="font-size:45px;font-weight:700;color:black;text-align:center;">NEXP2X Business-Case Tool</p>' 
#st.write(new_title0, unsafe_allow_html=True)
st.write(""" # Electrolyzer Business-Case Tool """)

st.write(""" Created in the NEXP2X Project - Funded by Innovation Fund Denmark """)




#Explain assumptions here

electrolyser_nom_cap = 1000 #kW

st.sidebar.image(image2)

new_title1 = '<p style="font-size:25px;font-weight:600;color:#f0f2f6">Key variables</p>'
st.sidebar.write(new_title1, unsafe_allow_html=True)
new_title3 = '<p style="font-size:15px;font-weight:500;color:#f0f2f6">This tool assumes a 1 MW electrolyzer. Main variables for this business case-study can be changed in the left pane and their initial values repesent SOEC technology.</p>'
st.sidebar.write(new_title3, unsafe_allow_html=True)

new_title2 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Market prices</p>'
st.sidebar.markdown(new_title2, unsafe_allow_html=True)
#Decide future Hydrogen price
Hydrogen_price = st.sidebar.slider('Average Hydrogen sales price in €/kg:',0, 15,6,1)


#Decide average electricity spot price
Electricity_spot_MWh = st.sidebar.slider('Average electricity spot price in €/MWh: ', 0, 200, 60,10)
Electricity_spot = Electricity_spot_MWh/1000

new_title4 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Technical</p>'
st.sidebar.markdown(new_title4, unsafe_allow_html=True)
#Decide how many full-load hours of operation will the electrolyzer run in a year
full_load_hours= st.sidebar.slider('Full-load hours of operation in a year: ', 0, 8760, 7500,100)

#Decide power-to-hydrogen production ratio
H2_electrolyser_input_1000 = st.sidebar.slider('Power-to-Hydrogen production ratio in kg/MWh: ', 10, 30, 23,1)
H2_electrolyser_input = H2_electrolyser_input_1000/1000

#Decide technical lifetime of stack
technical_lifetime_stacks= st.sidebar.slider('Stacks lifetime in full-load hours of operation: ', 0, 100000,20000 ,5000)

new_title5 = '<p style="font-size:20px;font-weight:600;color:#f0f2f6">Financial</p>'
st.sidebar.markdown(new_title5, unsafe_allow_html=True)
#Decide Specific capex
electrolyser_specific_invest= st.sidebar.slider('Electrolyzer capital investment in €/kW: ', 0, 5000, 3000,250)

#Decide OPEX % of CAPEX
electrolyser_OPEX_percentage2= st.sidebar.slider('O&M yearly in % CAPEX (Stack replacement cost not included): ', 0, 20, 5,1)
electrolyser_OPEX_percentage = electrolyser_OPEX_percentage2/100

#Decide stack replacement cost
electrolyser_STACK_replacement_100 = st.sidebar.slider('Stack replacement cost as % of CAPEX: ', 0, 50,20 ,1)
electrolyser_STACK_replacement = electrolyser_STACK_replacement_100/100

#Decide project lifetime
lifetime = st.sidebar.slider('Project lifetime in years:', 0, 30,25,1)

#Decide discount rate
discountRate_100 = st.sidebar.slider('Desired discount rate in %:', 0, 50, 5,1)
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

#------------------------------------NPV--------------------------------------------------------------------------------------
discountRate2 = round(discountRate*100,1)
npv             = npf.npv(discountRate, cf)
npv2 = round(npv,1)
NPV = np.zeros(len(cf))
for i in range(0,len(cf)):
    NPV[i] = npf.npv(discountRate, cf[0:(i+1)])
for i in range(1,len(cf)):
    if NPV[i]>=0:
        if NPV[i-1]<=0:
            a1010=i

if all(e <= 0 for e in NPV):
    a101="N/A"
    #st.write('Project is not profitable in 27 years')
else:
    a101 = "%s years" % (a1010)
    #st.write('Payback time: %s years' % (a1010))

#st.write('Net present value: %s M€ (%s %% discount rate)' % (npv2,discountRate2))
#------------------------------------IRR---------------------------------------------------------------------------------------------
IRR = npf.irr(cf)
IRR2 = round(IRR*100)
#st.write('IRR: %s %%' % (IRR2))
#Hydrogen Price independent
#------------------------------------LCoH-----------------------------------------------------------------------------------
Expenses = np.zeros(lifetime +1) #expenses plus electricity income in €/year
for t in range(1,len(cf)):
    Expenses[t] = -cf[t]*1e+6 + (Hydrogen_income_yearly[t])
Expenses[0] = -cf[0]*1e+6
LCoH = npf.npv(discountRate,Expenses)/npf.npv(discountRate, Hydrogen_production_yearly)
LCoH2 = round(LCoH,1)
#st.write('Levelized Cost of Hydrogen: %s €/kg (%s %% discount rate)' % (LCoH2,discountRate2))

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

data = {
'Electricity':LCoH_electricity_cost2,'CAPEX':LCoH_capex2,'Stack Replacement':LCoH_stack_rep_cost2, 'OPEX':LCoH_opex_electrolyser2
}
a20 = max(data, key=data.get)
per_main_costdriver = round(data[a20] / LCoH * 100 )

#------------------------------------Show results-----------------------------------------------------------------
#new_title7 = '<p style="font-size:45px;font-weight:700;color:black;text-align:center;">Results</p>'
#st.write(new_title7, unsafe_allow_html=True)
st.write(""" # Results """)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Payback time", '%s' % (a101))
col2.metric("NPV", "%s M€"  % (npv2))
col3.metric("IRR", "%s %%" % (IRR2))
col4.metric("LCoH", "%s €/kg" % (LCoH2))
st.metric("Cost-driver","%s (%s %% of cost)" % (a20, per_main_costdriver))
#st.write("The main cost-driver for the Levelized Cost of Hydrogen is found to be %s, accounting for %s %% of the cost." % (a20, per_main_costdriver))

st.write(" # Levelised cost contributions for Hydrogen")
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

brush2 = alt.selection_interval()
st.write(" # Cash flow plots")
year=np.linspace(0, lifetime,lifetime+1)
chart_data2 = pd.DataFrame({'Year':year,'Non-discounted Cash Flows in Million €':cf})
d = alt.Chart(chart_data2).mark_bar().encode(
     x='Year:O',y='Non-discounted Cash Flows in Million €:Q',color=alt.value('#ffe300')).interactive().properties(    #color=alt.condition(brush2, alt.value('#ffe300'), alt.value('lightgray'))
    title='Non-discounted Cash Flows',width= 600, height= 400
).configure_title(
    fontSize=25,
    fontWeight=900,
    anchor='middle',
    color='#f0f2f6'
).configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black') 
#.add_selection(brush2)
#.configure(background='white')

st.altair_chart(d, use_container_width=True) 

brush = alt.selection_interval()
chart_data3 = pd.DataFrame({'Year':year,"Acc Disc Cash Flows in Million €":NPV})

c = alt.Chart(chart_data3).mark_bar().encode(
     x='Year:O',y="Acc Disc Cash Flows in Million €", color=alt.value('#ffe300') )
#.add_selection(brush)

imgCode = []

#image = Image.open('HG_Flat_hori.png')
#output = io.BytesIO()    
#image.save(output, format='PNG')
#encoded_string = "data:image/png;base64,"+base64.b64encode(output.getvalue()).decode()
#imgCode.append(encoded_string)

#img_source = pd.DataFrame({"x": 10, "y": -2,"value": imgCode})
#img = alt.Chart(img_source).mark_image().encode( x='x',y='y',url='value')

if all(e <= 0 for e in NPV):
    g=(c).interactive().properties(
        title='Accumulated Discounted Cash Flows',width= 600, height= 400).configure_title(fontSize=25,fontWeight=900,anchor='middle',color='#f0f2f6').configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black').configure_line(fontStyle='dash', fontWeight=900).configure_text(fontSize=15,fontWeight='bold')
    st.altair_chart(g, use_container_width=True) 
else:
    for i in range(0,len(cf)):
        year[i]=a1010

    label=['']*len(cf)
    label[1]='Payback Time'
    chart_data4 = pd.DataFrame({'Year':year,"Acc Disc Cash Flows in Million €":NPV, "Label":label})    

    line = alt.Chart(chart_data4).mark_rule(color='red').encode( x='Year:O',y="Acc Disc Cash Flows in Million €")

    text = line.mark_text(
        align='right',
        baseline='middle',
        dx=-10
    , color= 'red').encode(
        text='Label'
    )

    g=(c+line+text).interactive().properties(
        title='Accumulated Discounted Cash Flows',width= 600, height= 400).configure_title(fontSize=25,fontWeight=900,anchor='middle',color='#f0f2f6').configure_axis(titleColor='#f0f2f6',labelColor='#f0f2f6',labelAngle=0,labelFontSize=15,titleFontSize=15, gridColor='black').configure_line(fontStyle='dash', fontWeight=900).configure_text(fontSize=15,fontWeight='bold') #.configure_image(opacity=0.5,width=50,height=50)
    st.altair_chart(g, use_container_width=True) 
    
new_title100 = '<p style="font-size:40px;font-weight:600;color:#ffe300">What is your next action towards 100% renewables?</p>'
st.write(new_title100, unsafe_allow_html=True)
st.write(" #  Let's create more value together, send us an e-mail to info@hybridgreentech.com")
