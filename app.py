import streamlit as st
import pickle
import pandas as pd
import os
import pickle

teams = [
    'Royal Challengers Bangalore', 'Delhi Capitals',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Mumbai Indians',
    'Chennai Super Kings', 'Sunrisers Hyderabad', 'Rising Pune Supergiants',
    'Gujarat Titans', 'Punjab Kings', 'Lucknow Super Giants'
]

cities = [
    'Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad',
    'Chennai', 'Johannesburg', 'Kimberley', 'Ahmedabad', 'Cuttack', 'Nagpur',
    'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Rajkot', 'Kanpur', 
    'Indore', 'Lucknow', 'Guwahati', 'Mohali', 'New Chandigarh'
]

file_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")

with open(os.path.join("pipe.pkl"), "rb") as f:
    pipe = pickle.load(f)

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
