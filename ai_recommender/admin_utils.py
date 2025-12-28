import csv
import io
import random
from django.utils import timezone
from .models import Career, SuggestedCareer, UserProfile, CareerRecommendation

# --- CSV UPLOAD LOGIC ---
def process_career_csv(csv_file):
    """
    Parses CSV and updates/creates Career entries.
    Expected columns: title, description, skills, salary, growth
    """
    decoded_file = csv_file.read().decode('utf-8')
    io_string = io.StringIO(decoded_file)
    reader = csv.DictReader(io_string)
    
    count = 0
    errors = []
    
    for row in reader:
        try:
            # Basic validation
            if not row.get('title'): continue
            
            career, created = Career.objects.update_or_create(
                title=row['title'],
                defaults={
                    'description': row.get('description', 'Imported via CSV'),
                    'required_skills': row.get('skills', ''),
                    'average_salary': row.get('salary', 0),
                    'job_growth_rate': row.get('growth', 0.05),
                    # Fill other required fields with defaults if missing
                    'education_required': row.get('education', 'Bachelors'),
                    'work_environment': row.get('environment', 'Office'),
                }
            )
            count += 1
        except Exception as e:
            errors.append(f"Row {count+1}: {str(e)}")
            
    return count, errors

# --- API SYNC LOGIC (Dummy) ---
def sync_job_market_data():
    """
    Placeholder for JSearch/External API Sync.
    In a real app, this would fetch live data and update Career models.
    """
    # Mocking a successful sync
    updated_count = random.randint(3, 10)
    return {
        'status': 'success',
        'updated': updated_count,
        'message': f"Successfully synced {updated_count} careers with JSearch API."
    }

# --- AI LEARNING LOGIC (Mock/Heuristic) ---
def generate_ai_insights():
    """
    Scans UserProfiles with NO matches and suggests new Careers.
    This simulates 'Learning' from student data.
    """
    # 1. Find 'At Risk' students (No recommendations)
    at_risk_users = []
    for profile in UserProfile.objects.all():
        if not CareerRecommendation.objects.filter(user_profile=profile).exists():
            at_risk_users.append(profile)
            
    if not at_risk_users:
        return "No insights generated. All students have matching careers."
        
    # 2. Analyze their skills (Simple Counter)
    from collections import Counter
    all_skills = []
    for p in at_risk_users:
        if p.skills:
            all_skills.extend([s.strip() for s in p.skills.split(',')])
            
    skill_counts = Counter(all_skills)
    common_skills = skill_counts.most_common(3)
    
    # 3. Create a Suggestion based on top skills
    created_count = 0
    if common_skills:
        top_skill = common_skills[0][0]
        suggested_title = f"{top_skill} Specialist (AI Generated)"
        
        # Check if already exists
        if not SuggestedCareer.objects.filter(title=suggested_title).exists():
            SuggestedCareer.objects.create(
                title=suggested_title,
                skills_detected=f"{top_skill}, {common_skills[1][0] if len(common_skills)>1 else ''}",
                confidence_score=0.85,
                source=f"Found {len(at_risk_users)} students with these skills but no career match."
            )
            created_count += 1
            
    return f"Analyzed {len(at_risk_users)} profiles. Generated {created_count} new career suggestions."

def retrain_ai_models():
    """
    Placeholder for model retraining.
    """
    # Mocking retraining time
    import time
    time.sleep(1) 
    return "Models successfully retrained using latest student data."
