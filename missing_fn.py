def predict_skill_gaps(user_skills_str, target_career):
    """
    [HINGLISH]
    Use: Gap analysis between user skills and career requirements.
    """
    from .models import Career
    from .utils import smart_split_skills, clean_title_for_merge
    
    # 1. Get Required Skills for Career
    required_skills_raw = ""
    
    # Try DB first
    career_obj = Career.objects.filter(title=target_career).first()
    if career_obj and career_obj.required_skills:
        required_skills_raw = career_obj.required_skills
    
    # Fallback to CSV (using CAREER_DF)
    if not required_skills_raw and 'CAREER_DF' in globals() and not CAREER_DF.empty:
        clean_target = clean_title_for_merge(target_career)
        career_rows = CAREER_DF[CAREER_DF['clean_key'] == clean_target]
        if not career_rows.empty:
            required_skills_raw = career_rows.iloc[0].get('required_skills', '')
            
    if not required_skills_raw:
        return {
            'gap_score': 0, 'missing_skills': [], 'required_skills': [], 
            'current_skills': [], 'coverage_percentage': 0
        }
        
    # 2. Parse Required Skills
    required_skills_list = smart_split_skills(str(required_skills_raw))
    required_skills_set = set([s.strip().lower() for s in required_skills_list if s.strip()])
    
    # 3. Parse User Skills
    user_skills_set = set()
    if user_skills_str:
        user_skills_list = [s.strip() for s in user_skills_str.split(',') if s.strip()]
        user_skills_set = set([s.lower() for s in user_skills_list])
            
    # 4. Calculate Intersection and Gap
    matching_skills = required_skills_set.intersection(user_skills_set)
    missing_skills = required_skills_set - user_skills_set
    
    required_count = len(required_skills_set)
    matching_count = len(matching_skills)
    
    coverage = (matching_count / required_count * 100) if required_count > 0 else 0
    gap_score = 1.0 - (matching_count / required_count) if required_count > 0 else 0
    
    # Format for display
    return {
        'required_skills': [{'name': s.title()} for s in required_skills_set],
        'current_skills': [{'name': s.title()} for s in matching_skills],
        'missing_skills': [{'name': s.title()} for s in missing_skills],
        'gap_score': round(gap_score, 2),
        'required_skills_count': required_count,
        'current_skills_count': matching_count,
        'missing_skills_count': len(missing_skills),
        'coverage_percentage': round(coverage, 1)
    }
