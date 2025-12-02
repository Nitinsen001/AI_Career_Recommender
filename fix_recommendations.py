import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from ai_recommender.models import UserProfile, CareerRecommendation
from ai_recommender.services import generate_career_recommendations

def fix_recommendations():
    print("--- FIXING CAREER RECOMMENDATIONS ---")
    
    # 1. Clear all existing recommendations
    count = CareerRecommendation.objects.count()
    print(f"Deleting {count} existing recommendations...")
    CareerRecommendation.objects.all().delete()
    
    # 2. Regenerate for all users
    profiles = UserProfile.objects.all()
    print(f"Regenerating for {profiles.count()} profiles...")
    
    for profile in profiles:
        print(f"Processing user: {profile.user.username}")
        try:
            generate_career_recommendations(profile)
            
            # Check results
            recs = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')
            print(f"  -> Generated {recs.count()} recommendations.")
            if recs.exists():
                top_rec = recs.first()
                print(f"  -> Top Match: {top_rec.recommended_career.title} ({top_rec.match_score}%)")
            else:
                print("  -> [WARNING] No recommendations generated!")
                
        except Exception as e:
            print(f"  -> [ERROR] Failed to generate: {e}")

    print("\n--- DONE ---")

if __name__ == "__main__":
    fix_recommendations()
