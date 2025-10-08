
# streamlit_app_single.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import os, warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    paths=['data/raw/','./']
    files=['users_dataset.csv','exercises_dataset.csv','foods_dataset.csv','user_exercise_interactions.csv','user_food_preferences.csv']
    for p in paths:
        try:
            users=pd.read_csv(os.path.join(p,files[0]))
            ex=pd.read_csv(os.path.join(p,files[1]))
            foods=pd.read_csv(os.path.join(p,files[2]))
            uex=pd.read_csv(os.path.join(p,files[3]))
            ufood=pd.read_csv(os.path.join(p,files[4]))
            return users,ex,foods,uex,ufood
        except FileNotFoundError:
            continue
    return None,None,None,None,None

class MiniPlanner:
    def __init__(self, users, ex, foods, uex, ufood):
        self.users=users; self.ex=ex.copy(); self.foods=foods.copy(); self.uex=uex; self.ufood=ufood
        self._prep()
    def _prep(self):
        ef=self.ex.copy()
        for col in ['muscle_group','equipment','difficulty_level']:
            ef[col]=ef[col].astype(str)
            ef[col+'_enc']=LabelEncoder().fit_transform(ef[col])
        feats=['muscle_group_enc','equipment_enc','difficulty_level_enc','duration_minutes','calories_per_minute']
        self.scaler=StandardScaler(); self.Xex=self.scaler.fit_transform(ef[feats])
        self.sim=cosine_similarity(self.Xex)
        self.umat=self.uex.pivot_table(index='user_id',columns='exercise_id',values='rating',fill_value=0)
        if len(self.umat)>0:
            self.knn=NearestNeighbors(n_neighbors=min(10,max(2,len(self.umat))),metric='cosine').fit(self.umat.values)
    def recommend_exercises_content(self, profile, k=5):
        goal=profile.get('fitness_goal','Maintenance')
        if goal=='Weight Loss':
            mask=(self.ex['muscle_group']=='Cardio') | (self.ex['calories_per_minute']>6)
        elif goal=='Muscle Gain':
            mask=self.ex['muscle_group'].isin(['Chest','Back','Legs','Arms','Shoulders'])
        else:
            mask=self.ex['muscle_group'].isin(['Full Body','Core'])
        base=self.ex[mask]
        if base.empty: base=self.ex.head(5)
        idx=base['exercise_id'].values-1
        score=self.sim[idx].mean(axis=0)
        top=np.argsort(score)[::-1]
        rec=self.ex.iloc[top]
        rec=rec[~rec['exercise_id'].isin(base['exercise_id'])].head(k)
        return rec[['exercise_name','muscle_group','equipment','difficulty_level','duration_minutes']]
    def recommend_exercises_cf(self, user_id, k=5):
        if user_id not in self.umat.index:
            pop=self.uex.groupby('exercise_id')['rating'].mean().sort_values(ascending=False)
            return self.ex[self.ex['exercise_id'].isin(pop.head(k).index)][['exercise_name','muscle_group','equipment','difficulty_level','duration_minutes']]
        uidx=self.umat.index.get_loc(user_id)
        dist,ind=self.knn.kneighbors(self.umat.values[uidx].reshape(1,-1))
        sims=[self.umat.index[i] for i in ind[0][1:]]
        liked=self.uex[(self.uex.user_id.isin(sims)) & (self.uex.rating>=4)]
        seen=set(self.uex[self.uex.user_id==user_id].exercise_id)
        cand=liked[~liked.exercise_id.isin(seen)]
        ids=cand.groupby('exercise_id').rating.mean().sort_values(ascending=False).head(k).index
        return self.ex[self.ex.exercise_id.isin(ids)][['exercise_name','muscle_group','equipment','difficulty_level','duration_minutes']]
    def daily_needs(self, p):
        bmr=10*p['weight_kg']+6.25*p['height_cm']-5*p['age']+(5 if p['gender']=='Male' else -161)
        mult={'Sedentary':1.2,'Lightly Active':1.375,'Moderately Active':1.55,'Very Active':1.725}
        cal=bmr*mult.get(p['activity_level'],1.55)
        if p['fitness_goal']=='Weight Loss': pr,cr,fr=0.30,0.40,0.30
        elif p['fitness_goal']=='Muscle Gain': pr,cr,fr=0.25,0.45,0.30
        else: pr,cr,fr=0.20,0.50,0.30
        return {'calories':cal,'protein_g':cal*pr/4,'carbs_g':cal*cr/4,'fat_g':cal*fr/9}
    def recommend_foods(self, p, needs, k=8):
        df=self.foods.copy(); pref=p.get('dietary_preferences','None')
        if pref=='Vegetarian': df=df[df.vegetarian==True]
        elif pref=='Vegan': df=df[df.vegan==True]
        elif pref=='Gluten-Free': df=df[df.gluten_free==True]
        elif pref=='Keto': df=df[df.keto_friendly==True]
        def score(r):
            cm=1-abs(r.calories_per_100g-needs['calories']/10)/max(needs['calories'],1)
            pm=1-abs(r.protein_g-needs['protein_g']/10)/max(needs['protein_g'],1)
            return max(0,0.4*cm+0.4*pm+0.2*(r.fiber_g/10))
        df['nutrition_score']=df.apply(score,axis=1)
        return df.nlargest(k,'nutrition_score')[['food_name','category','calories_per_100g','protein_g','carbs_g','fat_g','fiber_g']]
    def weekly_plan(self,p):
        rec=self.recommend_exercises_content(p,10); needs=self.daily_needs(p); foods=self.recommend_foods(p,needs,15)
        days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']; plan={}
        for i,d in enumerate(days):
            if i%2==0 and not rec.empty:
                w=rec.iloc[i%len(rec)]; workout={'exercise':w['exercise_name'],'muscle_group':w['muscle_group'],'duration':int(w['duration_minutes']),'equipment':w['equipment']}
            else: workout='Rest day or light activity'
            m=foods.sample(3) if len(foods)>=3 else foods
            meals={'breakfast':m.iloc[0]['food_name'],'lunch':m.iloc[1]['food_name'] if len(m)>1 else 'Salad','dinner':m.iloc[2]['food_name'] if len(m)>2 else 'Protein + veggies'}
            plan[d]={'workout':workout,'meals':meals}
        return plan

st.set_page_config(page_title='AI Fitness & Diet Planner', page_icon='🏋️‍♂️', layout='wide')
st.title('🏋️‍♂️ AI-Powered Fitness & Diet Planner')
st.caption('Personalized workout and nutrition recommendations powered by machine learning')

users,ex,foods,uex,ufood=load_data()
if users is None:
    st.error('Datasets not found. Place CSVs in data/raw/ or project root.')
    st.stop()
planner=MiniPlanner(users,ex,foods,uex,ufood)

page=st.sidebar.radio('Navigate',['Get Recommendations','Data Dashboard','About'])

def profile_form():
    c1,c2=st.columns(2)
    with c1:
        age=st.slider('Age',18,80,30); gender=st.selectbox('Gender',['Male','Female'])
        height=st.slider('Height (cm)',140,210,170); weight=st.slider('Weight (kg)',40,150,70)
    with c2:
        activity=st.selectbox('Activity Level',['Sedentary','Lightly Active','Moderately Active','Very Active'])
        goal=st.selectbox('Fitness Goal',['Weight Loss','Muscle Gain','Maintenance','Endurance'])
        diet=st.selectbox('Dietary Preferences',['None','Vegetarian','Vegan','Gluten-Free','Keto'])
        condition=st.selectbox('Medical Conditions',['None','Diabetes','Hypertension','Heart Disease'])
    bmi=weight/((height/100)**2); bmr=10*weight+6.25*height-5*age+(5 if gender=='Male' else -161)
    mult={'Sedentary':1.2,'Lightly Active':1.375,'Moderately Active':1.55,'Very Active':1.725}
    daily=int(bmr*mult[activity])
    m1,m2,m3,m4=st.columns(4); m1.metric('BMI',f'{bmi:.1f}'); m2.metric('BMR',f'{int(bmr)} cal/day'); m3.metric('Daily Calories',f'{daily}'); m4.metric('Category','Normal' if 18.5<=bmi<25 else 'Check')
    return {'user_id':9999,'age':age,'gender':gender,'height_cm':height,'weight_kg':weight,'activity_level':activity,'fitness_goal':goal,'dietary_preferences':diet,'medical_conditions':condition,'bmi':round(bmi,1),'bmr':int(bmr),'daily_calories':daily}

if page=='Get Recommendations':
    profile=profile_form()
    if st.button('Generate My Plan'):
        needs=planner.daily_needs(profile)
        ex_c=planner.recommend_exercises_content(profile,5)
        ex_cf=planner.recommend_exercises_cf(profile['user_id'],5)
        foods_rec=planner.recommend_foods(profile,needs,8)
        tabs=st.tabs(['Exercises','Nutrition','Weekly Plan'])
        with tabs[0]:
            st.subheader('Content-Based Recommendations'); st.dataframe(ex_c)
            st.subheader('Collaborative Filtering Recommendations'); st.dataframe(ex_cf)
        with tabs[1]:
            l,r=st.columns([1,1])
            with l:
                st.metric('Calories',f"{int(needs['calories'])}"); st.metric('Protein',f"{int(needs['protein_g'])} g"); st.metric('Carbs',f"{int(needs['carbs_g'])} g"); st.metric('Fat',f"{int(needs['fat_g'])} g")
            with r:
                fig=go.Figure(data=[go.Pie(labels=['Protein','Carbs','Fat'],values=[needs['protein_g'],needs['carbs_g'],needs['fat_g']],hole=.3)])
                st.plotly_chart(fig, use_container_width=True)
            st.subheader('Recommended Foods'); st.dataframe(foods_rec)
        with tabs[2]:
            plan=planner.weekly_plan(profile)
            for day,d in plan.items():
                with st.expander(day):
                    st.write('Workout:',d['workout']); st.write('Meals:',d['meals'])
elif page=='Data Dashboard':
    t1,t2,t3=st.tabs(['Users','Exercises','Interactions'])
    with t1:
        a,b=st.columns(2)
        a.plotly_chart(px.histogram(users,x='age',nbins=20,title='Age Distribution'), use_container_width=True)
        b.plotly_chart(px.histogram(users,x='bmi',nbins=20,title='BMI Distribution'), use_container_width=True)
        st.plotly_chart(px.bar(users['fitness_goal'].value_counts(), title='Fitness Goals'), use_container_width=True)
    with t2:
        st.plotly_chart(px.bar(ex['muscle_group'].value_counts(), title='Muscle Groups'), use_container_width=True)
        st.plotly_chart(px.pie(values=ex['difficulty_level'].value_counts().values, names=ex['difficulty_level'].value_counts().index, title='Difficulty Levels'), use_container_width=True)
    with t3:
        st.plotly_chart(px.histogram(uex,x='rating',nbins=5,title='Exercise Ratings'), use_container_width=True)
        st.plotly_chart(px.histogram(uex,x='completion_rate',nbins=20,title='Completion Rates'), use_container_width=True)
else:
    st.write('This Streamlit app wraps a compact AI system that generates personalized workout and diet plans using content-based and collaborative filtering plus nutrition analysis.')
