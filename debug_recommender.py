import os
import django
import pandas as pd
import numpy as np

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from ai_recommender.models import UserProfile, SkillAssessment
from ai_recommender.services import (
    enhanced_find_career_matches, 
    compute_match_percentage, 
    calculate_skills_match_bonus,
    get_combined_user_skills,
    CAREER_DF,
    VECTORIZER,
    CAREER_VECTORS
)

def debug_matching():
    print("--- DEBUGGING CAREER MATCHING ---")
    
    # 1. Get or Create a Test User
    user = UserProfile.objects.first()
    if not user:
        print("No user profile found!")
        return

    print(f"User: {user.user.username}")
    print(f"Skills: {user.skills}")
    print(f"Experience: {user.experience_years}")
    
    # 2. Check Combined Skills
    combined_skills = get_combined_user_skills(user)
    print(f"Combined Skills Objects: {[s.skill_name for s in combined_skills]}")
    
    # 3. Run Enhanced Match
    matches = enhanced_find_career_matches(user)
    
    print(f"\nFound {len(matches)} matches.")
    for i, m in enumerate(matches[:3]):
        print(f"\nMatch #{i+1}: {m['title']}")
        print(f"  Final Score: {m['match_score']}")
        print(f"  Skills Match Display: {m['skills_match']}")
        
        # 4. Deep Dive into Calculation for this match
        # Re-simulate the logic inside enhanced_find_career_matches for this specific career
        
        # Find index in CAREER_DF
        career_rows = CAREER_DF[CAREER_DF['career_name'] == m['title']]
        if career_rows.empty:
            print("  [ERROR] Career not found in DF")
            continue
            
        career_row = career_rows.iloc[0]
        
        # Calculate ML Score manually
        # Reconstruct profile text
        user_skills_text = ' '.join([f"{skill.skill_name} " * (int(skill.years_of_experience) + 1) for skill in combined_skills])
        profile_text = f"{user.education_level} {user.gender} {user.personality_type or ''} {user_skills_text} {getattr(user, 'interests', '')} {getattr(user, 'current_field', '')}"
        
        user_vector = VECTORIZER.transform([profile_text])
        # Find vector index? We don't know the exact index in CAREER_VECTORS easily without iterating
        # But we can compute similarity against ALL and find the max for this title?
        # Actually, let's just trust the ML score might be low, but check SKILLS score.
        
        # Check Skills Score
        class TempCareer:
            def __init__(self, title, req_skills):
                self.title = title
                self.required_skills = req_skills
        
        career_obj = TempCareer(m['title'], career_row.get('required_skills', ''))
        
        skills_score = calculate_skills_match_bonus(combined_skills, career_obj)
        print(f"  [DEBUG] Calculated Skills Score: {skills_score}")
        print(f"  [DEBUG] Required Skills Raw: {career_row.get('required_skills', '')}")
        
        # Check ML Score contribution
        # We can't easily get the exact ML score from here without re-running the whole vector search
        # But if Final Score is 0, and Skills Score is > 0, then something is wrong with weighting.

if __name__ == "__main__":
    debug_matching()
