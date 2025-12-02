import os
import django
import sys
import numpy as np

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from django.contrib.auth.models import User
from ai_recommender.models import UserProfile, SkillAssessment, Career
from ai_recommender.services import (
    enhanced_find_career_matches, 
    get_combined_user_skills, 
    VECTORIZER, 
    CAREER_VECTORS,
    CAREER_DF,
    clean_title_for_merge,
    smart_split_skills
)

def debug_recommendation_logic():
    print("DEBUG: Starting Recommendation Logic Analysis")
    print("-" * 50)

    # 1. Get or Create Test User
    username = "debug_user"
    try:
        user = User.objects.get(username=username)
        print(f"Found existing user {username}")
    except User.DoesNotExist:
        user = User.objects.create_user(username=username, password="password123")
        print(f"Created new user {username}")

    profile, _ = UserProfile.objects.get_or_create(user=user)
    
    # Set specific attributes for debugging
    profile.skills = "Python, Django, Machine Learning"
    profile.interests = "AI, Web Development"
    profile.current_field = "Computer Science"
    profile.experience_years = 2
    profile.personality_type = "INTJ"
    profile.save()
    
    # Clear DB skills to test combined logic
    SkillAssessment.objects.filter(user_profile=profile).delete()
    
    print(f"Profile Configured: Skills={profile.skills}, Interests={profile.interests}, Type={profile.personality_type}")

    # 2. Test Skill Extraction
    combined_skills = get_combined_user_skills(profile)
    print(f"\nCombined Skills Found ({len(combined_skills)}):")
    for s in combined_skills:
        print(f" - {s.skill_name} (Lvl: {s.skill_level}, Exp: {s.years_of_experience})")

    # 3. Test Vector Generation
    user_skills_text = ' '.join([f"{skill.skill_name} " * (int(skill.years_of_experience) + 1) for skill in combined_skills])
    profile_text = f"{profile.education_level} {profile.gender} {profile.personality_type or ''} {user_skills_text} {profile.interests} {profile.current_field}"
    print(f"\nProfile Text for ML:\n'{profile_text}'")
    
    if not profile_text.strip():
        print("FAIL: Profile text is empty!")
        return

    try:
        user_vector = VECTORIZER.transform([profile_text])
        print(f"Vector Shape: {user_vector.shape}")
        print(f"Non-zero elements in vector: {user_vector.nnz}")
        
        if user_vector.nnz == 0:
            print("FAIL: Vector is all zeros! Vocabulary mismatch?")
            # Print some vocabulary to check
            # print(f"Vectorizer Vocab Sample: {list(VECTORIZER.vocabulary_.keys())[:10]}")
    except Exception as e:
        print(f"FAIL: Vectorization error: {e}")
        return

    # 4. Test Cosine Similarity
    similarity_scores = (user_vector * CAREER_VECTORS.T).toarray().flatten()
    print(f"\nMax Similarity Score: {np.max(similarity_scores)}")
    print(f"Mean Similarity Score: {np.mean(similarity_scores)}")
    
    top_indices = np.argsort(similarity_scores)[::-1][:5]
    print("\nTop 5 ML Matches (Raw):")
    for idx in top_indices:
        if idx < len(CAREER_DF):
            print(f" - {CAREER_DF.iloc[idx]['career_name']}: {similarity_scores[idx]}")
        else:
            print(f" - Index {idx} out of bounds (DF len: {len(CAREER_DF)})")

    # 5. Test Full Matching Function
    print("\nRunning enhanced_find_career_matches...")
    matches = enhanced_find_career_matches(profile)
    
    print(f"\nFinal Matches Returned ({len(matches)}):")
    for m in matches:
        print(f" - {m['title']}: Score={m['match_score']}, SkillsMatch={m['skills_match']}")
        
        # Deep dive into the first match
        if m == matches[0]:
            print("\n   --- Deep Dive into Top Match ---")
            career_title = m['title']
            try:
                career_obj = Career.objects.get(title=career_title)
                req_skills = career_obj.required_skills
                print(f"   DB Required Skills: {req_skills}")
            except Career.DoesNotExist:
                print("   Career not in DB (using DF fallback)")
                req_skills = CAREER_DF[CAREER_DF['career_name'] == career_title].iloc[0]['required_skills']
                print(f"   DF Required Skills: {req_skills}")
            
            req_list = smart_split_skills(req_skills)
            req_set = set([s.strip().lower() for s in req_list])
            user_set = set([s.skill_name.lower() for s in combined_skills])
            overlap = user_set.intersection(req_set)
            print(f"   User Skills: {user_set}")
            print(f"   Required Set: {req_set}")
            print(f"   Overlap: {overlap}")

if __name__ == "__main__":
    with open('debug_output.txt', 'w') as f:
        sys.stdout = f
        debug_recommendation_logic()
        sys.stdout = sys.__stdout__
    print("Debug output written to debug_output.txt")
