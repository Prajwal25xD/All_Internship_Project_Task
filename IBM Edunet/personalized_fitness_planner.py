
"""
PERSONALIZED WORKOUT & DIET PLANNER WITH AI
============================================

A comprehensive machine learning solution for personalized fitness and nutrition recommendations.
This system combines multiple recommendation techniques including:
- Content-based filtering for exercises and nutrition
- Collaborative filtering for exercise recommendations  
- Hybrid recommendation system
- Nutritional analysis and meal planning
- User profiling and preference learning

Libraries Used:
- pandas, numpy: Data manipulation and analysis
- scikit-learn: Machine learning algorithms
- matplotlib, seaborn: Data visualization
- TensorFlow/Keras: Deep learning models (optional)

Datasets:
1. users_dataset.csv - User profiles with demographics and fitness goals
2. exercises_dataset.csv - Exercise database with features
3. foods_dataset.csv - Food nutrition database
4. user_exercise_interactions.csv - User-exercise rating interactions
5. user_food_preferences.csv - User food preferences

Author: AI Assistant
Date: September 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class PersonalizedFitnessPlanner:
    """
    Main class for the personalized fitness and diet planning system
    """
    
    def __init__(self, users_df, exercises_df, foods_df, user_exercise_df, user_food_df):
        self.users_df = users_df
        self.exercises_df = exercises_df
        self.foods_df = foods_df
        self.user_exercise_df = user_exercise_df
        self.user_food_df = user_food_df
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.exercise_features = None
        self.food_features = None
        self.user_exercise_matrix = None
        self.collaborative_model = None
        
    def perform_eda(self):
        """Comprehensive Exploratory Data Analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        # User demographics
        print("USER DEMOGRAPHICS:")
        print(f"Age: {self.users_df['age'].describe()}")
        print(f"BMI: {self.users_df['bmi'].describe()}")
        print(f"Gender distribution:\n{self.users_df['gender'].value_counts()}")
        
        # Exercise analysis
        print("\nEXERCISE ANALYSIS:")
        print(f"Muscle groups:\n{self.exercises_df['muscle_group'].value_counts()}")
        print(f"Difficulty levels:\n{self.exercises_df['difficulty_level'].value_counts()}")
        
        # Food analysis
        print("\nFOOD ANALYSIS:")
        print(f"Categories:\n{self.foods_df['category'].value_counts()}")
        print(f"Average calories per 100g: {self.foods_df['calories_per_100g'].mean():.1f}")
        
        return self.users_df.describe(), self.exercises_df.describe(), self.foods_df.describe()
    
    def preprocess_data(self):
        """Data preprocessing for machine learning models"""
        # Encode exercise features
        exercise_features = self.exercises_df.copy()
        categorical_cols = ['muscle_group', 'equipment', 'difficulty_level']
        
        for col in categorical_cols:
            le = LabelEncoder()
            exercise_features[f'{col}_encoded'] = le.fit_transform(exercise_features[col])
            self.label_encoders[f'exercise_{col}'] = le
            
        feature_cols = [col for col in exercise_features.columns if col.endswith('_encoded')] + ['duration_minutes', 'calories_per_minute']
        self.exercise_features = exercise_features[feature_cols]
        self.exercise_features_scaled = self.scaler.fit_transform(self.exercise_features)
        
        # Create user-exercise interaction matrix
        self.user_exercise_matrix = self.user_exercise_df.pivot_table(
            index='user_id', columns='exercise_id', values='rating', fill_value=0
        )
        
        return self.exercise_features, self.user_exercise_matrix
    
    def build_content_based_recommender(self):
        """Build content-based recommendation system"""
        exercise_similarity = cosine_similarity(self.exercise_features_scaled)
        
        def recommend_exercises(user_profile, n_recommendations=5):
            # Logic for content-based recommendations based on user profile
            if user_profile.get('fitness_goal') == 'Weight Loss':
                preferred_muscle_groups = ['Cardio', 'Full Body']
            elif user_profile.get('fitness_goal') == 'Muscle Gain':
                preferred_muscle_groups = ['Chest', 'Back', 'Legs', 'Arms']
            else:
                preferred_muscle_groups = ['Core', 'Flexibility']
            
            # Find exercises matching user preferences
            mask = self.exercises_df['muscle_group'].isin(preferred_muscle_groups)
            if not mask.any():
                mask = self.exercises_df.index < 5  # Default to first 5 exercises
                
            preferred_exercises = self.exercises_df[mask]
            
            if len(preferred_exercises) == 0:
                return self.exercises_df.head(n_recommendations)
                
            # Calculate similarity and recommend
            exercise_ids = preferred_exercises['exercise_id'].values - 1
            similarity_scores = exercise_similarity[exercise_ids].mean(axis=0)
            top_indices = similarity_scores.argsort()[-n_recommendations-len(exercise_ids):][::-1]
            
            recommended = self.exercises_df.iloc[top_indices]
            return recommended[~recommended['exercise_id'].isin(preferred_exercises['exercise_id'])].head(n_recommendations)
        
        return recommend_exercises
    
    def build_collaborative_filtering(self):
        """Build collaborative filtering model"""
        self.collaborative_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.collaborative_model.fit(self.user_exercise_matrix.values)
        
        def recommend_collaborative(user_id, n_recommendations=5):
            if user_id not in self.user_exercise_matrix.index:
                # Return popular exercises for new users
                popular = self.user_exercise_df.groupby('exercise_id')['rating'].mean().sort_values(ascending=False)
                top_exercise_ids = popular.head(n_recommendations).index
                return self.exercises_df[self.exercises_df['exercise_id'].isin(top_exercise_ids)]
            
            # Find similar users
            user_idx = self.user_exercise_matrix.index.get_loc(user_id)
            user_vector = self.user_exercise_matrix.values[user_idx].reshape(1, -1)
            distances, indices = self.collaborative_model.kneighbors(user_vector)
            
            # Get recommendations from similar users
            similar_users = [self.user_exercise_matrix.index[i] for i in indices[0][1:]]
            similar_users_data = self.user_exercise_df[
                (self.user_exercise_df['user_id'].isin(similar_users)) & 
                (self.user_exercise_df['rating'] >= 4)
            ]
            
            # Exclude exercises already rated by user
            user_exercises = self.user_exercise_df[self.user_exercise_df['user_id'] == user_id]['exercise_id'].values
            recommendations = similar_users_data[~similar_users_data['exercise_id'].isin(user_exercises)]
            
            top_exercise_ids = recommendations.groupby('exercise_id')['rating'].mean().sort_values(ascending=False).head(n_recommendations).index
            return self.exercises_df[self.exercises_df['exercise_id'].isin(top_exercise_ids)]
        
        return recommend_collaborative

class NutritionRecommendationSystem:
    """Nutrition analysis and meal planning system"""
    
    def __init__(self, foods_df, users_df):
        self.foods_df = foods_df
        self.users_df = users_df
    
    def calculate_daily_needs(self, user_profile):
        """Calculate daily nutritional requirements"""
        bmr = user_profile['bmr']
        activity_multipliers = {
            'Sedentary': 1.2, 'Lightly Active': 1.375, 
            'Moderately Active': 1.55, 'Very Active': 1.725
        }
        
        daily_calories = bmr * activity_multipliers.get(user_profile['activity_level'], 1.55)
        
        # Macro ratios based on fitness goals
        if user_profile['fitness_goal'] == 'Weight Loss':
            protein_ratio, carb_ratio, fat_ratio = 0.30, 0.40, 0.30
        elif user_profile['fitness_goal'] == 'Muscle Gain':
            protein_ratio, carb_ratio, fat_ratio = 0.25, 0.45, 0.30
        else:
            protein_ratio, carb_ratio, fat_ratio = 0.20, 0.50, 0.30
            
        return {
            'calories': daily_calories,
            'protein_g': (daily_calories * protein_ratio) / 4,
            'carbs_g': (daily_calories * carb_ratio) / 4,
            'fat_g': (daily_calories * fat_ratio) / 9
        }
    
    def recommend_foods(self, user_profile, daily_needs, n_recommendations=10):
        """Recommend foods based on nutritional needs and dietary preferences"""
        available_foods = self.foods_df.copy()
        
        # Filter by dietary preferences
        dietary_pref = user_profile.get('dietary_preferences', 'None')
        if dietary_pref == 'Vegetarian':
            available_foods = available_foods[available_foods['vegetarian'] == True]
        elif dietary_pref == 'Vegan':
            available_foods = available_foods[available_foods['vegan'] == True]
        elif dietary_pref == 'Gluten-Free':
            available_foods = available_foods[available_foods['gluten_free'] == True]
        
        # Score foods based on nutritional fit
        def nutrition_score(food_row):
            calorie_match = 1 - abs(food_row['calories_per_100g'] - daily_needs['calories']/10) / daily_needs['calories']
            protein_match = 1 - abs(food_row['protein_g'] - daily_needs['protein_g']/10) / daily_needs['protein_g']
            return 0.4 * max(0, calorie_match) + 0.4 * max(0, protein_match) + 0.2 * (food_row['fiber_g'] / 10)
        
        available_foods['nutrition_score'] = available_foods.apply(nutrition_score, axis=1)
        return available_foods.nlargest(n_recommendations, 'nutrition_score')

class PersonalizedFitnessPlannerApp:
    """Main application interface"""
    
    def __init__(self, fitness_planner):
        self.fitness_planner = fitness_planner
        self.content_recommender = fitness_planner.build_content_based_recommender()
        self.collaborative_recommender = fitness_planner.build_collaborative_filtering()
        self.nutrition_system = NutritionRecommendationSystem(fitness_planner.foods_df, fitness_planner.users_df)
        
    def get_recommendations(self, user_id_or_profile, rec_type='hybrid'):
        """Get personalized recommendations for user"""
        # Handle both user ID and profile input
        if isinstance(user_id_or_profile, int):
            user_id = user_id_or_profile
            user_profile = self.fitness_planner.users_df[
                self.fitness_planner.users_df['user_id'] == user_id
            ].iloc[0].to_dict()
        else:
            user_profile = user_id_or_profile
            user_id = user_profile.get('user_id', 9999)
        
        results = {}
        
        # Exercise recommendations
        if rec_type in ['content', 'hybrid']:
            results['content_exercises'] = self.content_recommender(user_profile, 5)
        if rec_type in ['collaborative', 'hybrid']:
            results['collab_exercises'] = self.collaborative_recommender(user_id, 5)
        
        # Nutrition recommendations
        daily_needs = self.nutrition_system.calculate_daily_needs(user_profile)
        results['daily_needs'] = daily_needs
        results['food_recommendations'] = self.nutrition_system.recommend_foods(user_profile, daily_needs, 8)
        
        return results
    
    def generate_weekly_plan(self, user_profile):
        """Generate complete weekly workout and meal plan"""
        exercise_recs = self.content_recommender(user_profile, 7)
        daily_needs = self.nutrition_system.calculate_daily_needs(user_profile)
        food_recs = self.nutrition_system.recommend_foods(user_profile, daily_needs, 21)  # 3 meals � 7 days
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_plan = {}
        
        for i, day in enumerate(days):
            if i < len(exercise_recs):
                workout = exercise_recs.iloc[i]
                daily_plan = {
                    'workout': f"{workout['exercise_name']} ({workout['muscle_group']}) - {workout['duration_minutes']} min",
                    'meals': {
                        'breakfast': food_recs.iloc[i*3]['food_name'] if i*3 < len(food_recs) else 'Oatmeal',
                        'lunch': food_recs.iloc[i*3+1]['food_name'] if i*3+1 < len(food_recs) else 'Salad',
                        'dinner': food_recs.iloc[i*3+2]['food_name'] if i*3+2 < len(food_recs) else 'Lean protein'
                    }
                }
            else:
                daily_plan = {
                    'workout': 'Rest day',
                    'meals': {'breakfast': 'Fruit', 'lunch': 'Vegetables', 'dinner': 'Light protein'}
                }
            
            weekly_plan[day] = daily_plan
        
        return weekly_plan

# Example usage and testing
if __name__ == "__main__":
    # Load datasets
    users_df = pd.read_csv('users_dataset.csv')
    exercises_df = pd.read_csv('exercises_dataset.csv')
    foods_df = pd.read_csv('foods_dataset.csv')
    user_exercise_df = pd.read_csv('user_exercise_interactions.csv')
    user_food_df = pd.read_csv('user_food_preferences.csv')
    
    # Initialize system
    planner = PersonalizedFitnessPlanner(users_df, exercises_df, foods_df, user_exercise_df, user_food_df)
    planner.preprocess_data()
    
    app = PersonalizedFitnessPlannerApp(planner)
    
    # Example recommendations
    sample_user = {
        'user_id': 1001,
        'age': 25,
        'gender': 'Female', 
        'fitness_goal': 'Weight Loss',
        'activity_level': 'Moderately Active',
        'dietary_preferences': 'Vegetarian',
        'bmr': 1400,
        'daily_calories': 2000
    }
    
    recommendations = app.get_recommendations(sample_user, 'hybrid')
    weekly_plan = app.generate_weekly_plan(sample_user)
    
    print("System ready for personalized fitness and nutrition recommendations!")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
