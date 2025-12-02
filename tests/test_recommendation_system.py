import os
import django
import sys

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from django.contrib.auth.models import User
from ai_recommender.models import UserProfile, SkillAssessment, CareerRecommendation
from ai_recommender.services import generate_career_recommendations, predict_skill_gaps

def test_career_recommendation_system():
    print("Running Black-Box Test: Career Recommendation System (TC09)")
    print("-" * 50)

    # 1. Setup Test User
    username = "test_user_tc09"
    try:
        user = User.objects.get(username=username)
        print(f"Found existing user {username}, deleting...")
        user.delete()
    except User.DoesNotExist:
        pass

    user = User.objects.create_user(username=username, password="password123")
    profile = UserProfile.objects.create(
        user=user,
        age=25,
        gender='M',
        education_level='UG',
        experience_years=2,
        personality_type='INTJ' # Architect
    )
    print(f"Created test user: {username} (INTJ, 2 years exp)")

    # 2. Add Skills (Python Developer Profile)
    skills = [
        ('Python', 'expert', 3),
        ('Django', 'advanced', 2),
        ('SQL', 'intermediate', 2),
        ('Machine Learning', 'beginner', 1)
    ]
    
    for name, level, years in skills:
        SkillAssessment.objects.create(
            user_profile=profile,
            skill_name=name,
            skill_level=level,
            years_of_experience=years
        )
    
    # Update profile skills string
    profile.skills = ", ".join([s[0] for s in skills])
    profile.save()
    print(f"Added skills: {profile.skills}")

    # 3. Run Recommendation Engine
    print("\nGenerating recommendations...")
    generate_career_recommendations(profile)

    # 4. Verify Results
    recommendations = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')
    
    if not recommendations.exists():
        print("FAIL: No recommendations generated.")
        return

    print(f"\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec.recommended_career.title} - Match Score: {rec.match_score}%")
        
        # Verify Score Range
        if not (0 <= rec.match_score <= 100):
            print(f"  FAIL: Match score {rec.match_score} out of range (0-100)")
        
        # Verify Reasoning
        if not rec.reasoning:
            print(f"  FAIL: Missing reasoning")

    # 5. Verify Specific Match (Expect Python/Data roles)
    top_rec = recommendations.first()
    expected_keywords = ['Python', 'Developer', 'Engineer', 'Data', 'Backend']
    if any(k.lower() in top_rec.recommended_career.title.lower() for k in expected_keywords):
        print(f"\nPASS: Top recommendation '{top_rec.recommended_career.title}' is relevant.")
    else:
        print(f"\nWARNING: Top recommendation '{top_rec.recommended_career.title}' might not be relevant.")

    # 6. Verify Skill Gap Analysis
    print(f"\nChecking Skill Gap for: {top_rec.recommended_career.title}")
    gap_data = predict_skill_gaps(profile.skills, top_rec.recommended_career.title)
    
    print(f"Gap Score: {gap_data['gap_score']}")
    print(f"Missing Skills: {[s['name'] for s in gap_data['missing_skills']]}")
    
    if gap_data['gap_score'] >= 0:
        print("PASS: Skill gap analysis returned valid data.")
    else:
        print("FAIL: Skill gap analysis failed.")

    # Cleanup
    # user.delete()
    print("\nTest Complete.")

if __name__ == "__main__":
    test_career_recommendation_system()
