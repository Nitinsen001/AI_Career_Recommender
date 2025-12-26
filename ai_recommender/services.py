import pandas as pd

import numpy as np

import joblib

import re

import requests



from django.core.cache import cache

from django.conf import settings

import PyPDF2

from docx import Document





from sklearn.metrics.pairwise import cosine_similarity



from .models import UserProfile, Career, PersonalityAssessment, SkillAssessment, CareerRecommendation

from .utils import (

    clean_skill_list, clean_title_for_merge, smart_split_skills, 

    format_salary, get_default_salary, get_default_salary_by_title,

    get_default_growth_by_title, calculate_demand_level, 

    generate_demand_level_by_title, safe_load_df, normalize_skill

)



# --- GLOBAL DATA LOADING ---

try:

    # --- HERE ARE THE GLOBAL DATA FRAMES (WITH COLUMN NORMALIZATION) ---

    CAREER_DF = safe_load_df('career_dataset.csv')

    if CAREER_DF is None:

        CAREER_DF = safe_load_df('career_dataset_final.csv')



    if CAREER_DF is None or CAREER_DF.empty:

        raise FileNotFoundError("Career data not found!")

    

    CAREER_DF['clean_key'] = CAREER_DF.get('career_name', pd.Series(['Unknown'] * len(CAREER_DF))).apply(clean_title_for_merge)

    

    CAREER_DF.drop_duplicates(subset=['clean_key'], keep='first', inplace=True)



    SKILLS_DF = safe_load_df('skills_dataset.csv')

    PERSONALITY_DF = safe_load_df('personality.csv')



    # ----------------------------------------

except Exception as e:

    print(f"FATAL ERROR: Could not load core datasets: {e}")

    CAREER_DF = pd.DataFrame(columns=['career_name', 'required_skills', 'description', 'career_id', 'education_required', 'average_salary', 'clean_key']) 

    SKILLS_DF = pd.DataFrame(columns=['skill_id', 'skill_name', 'category'])





# Load ML models (with error handling)

models_loaded = False

try:

    # Use the 'enhanced' models from the training script

    SKILL_EXTRACTOR_DATA = joblib.load('enhanced_models/enhanced_skill_extractor.pkl')

    CONFIDENCE_SCORER_DATA = joblib.load('enhanced_models/enhanced_confidence_scorer.pkl')

    

    # The skill extractor is now just a clean list of skills

    SKILL_LIST = SKILL_EXTRACTOR_DATA['all_skills_list']

    

    VECTORIZER = CONFIDENCE_SCORER_DATA['vectorizer']

    CAREER_VECTORS = CONFIDENCE_SCORER_DATA['career_vectors']

    

    models_loaded = True

except Exception as e:

    print(f"Warning: Could not load ENHANCED ML models: {e}")

    print("Some features will be limited. Please run 'python train_models.py'.")





# --- SKILL CATEGORIZATION ---



def get_skill_category_map():

    """

    [HINGLISH]

    Use: Ye function `group_skills_by_category` me use hota h.

    Why: Skills ko unki category (e.g., Python -> Programming Language) se map karne ke liye.

    Effect: Isse hum skills ko sahi group me dikha paate h.

    """

    global SKILLS_DF

    category_map = {}

    

    # Ensure SKILLS_DF has the required normalized columns

    if SKILLS_DF is not None and not SKILLS_DF.empty and 'skill_name' in SKILLS_DF.columns and 'category' in SKILLS_DF.columns:

        for _, row in SKILLS_DF.iterrows():

            skill = str(row['skill_name']).strip().lower()

            category = str(row['category']).strip()

            if skill and category:

                category_map[skill] = category

    return category_map



# Define Category Metadata for HTML styling

CATEGORY_METADATA = {

    'Programming Languages': {'icon': 'code', 'color_class': 'prog'},

    'Frameworks': {'icon': 'microchip', 'color_class': 'info'},

    'Databases': {'icon': 'database', 'color_class': 'db'},

    'Cloud Platforms': {'icon': 'cloud', 'color_class': 'cloud'},

    'DevOps': {'icon': 'cogs', 'color_class': 'tools'},

    'Soft Skills': {'icon': 'handshake', 'color_class': 'secondary'},

    # Add your specific categories here if needed

    'Other': {'icon': 'tag', 'color_class': 'secondary'},

}



def group_skills_by_category(profile_skills_list):

    """

    [HINGLISH]

    Use: Ye `views.py` me resume upload page par skills ko categorize karke dikhane ke liye use hota h.

    Why: Agar saare skills ek list me honge to user confuse ho jayega. Grouping se UI clean lagta h.

    Effect: User ko apne skills organized way me dikhte h (e.g., Languages alag, Tools alag).

    """

    skill_to_category_map = get_skill_category_map()

    categorized_skills = {}



    for skill_display_name in profile_skills_list:

        # Use lowercased version for lookup in the map

        skill_key = skill_display_name.lower()

        # Fallback to general category if exact match not found

        category_name = skill_to_category_map.get(skill_key, 'Other')

        

        # Get metadata for styling (for HTML use)

        category_info = CATEGORY_METADATA.get(category_name, CATEGORY_METADATA['Other'])



        if category_name not in categorized_skills:

            categorized_skills[category_name] = {

                'info': category_info,

                'skills': []

            }

        

        # Append the display name (Title Cased)

        categorized_skills[category_name]['skills'].append(skill_display_name)

    

    # Sort categories alphabetically

    return dict(sorted(categorized_skills.items()))





# --- RESUME ANALYSIS ---



def analyze_resume_file(resume_file):

    """

    [HINGLISH]

    Use: Ye `views.py` me jab user resume upload karta h tab use hota h.

    Why: Resume (PDF/Docx) se text padh kar usme se skills aur experience extract karne ke liye.

    Effect: User ko manually form nahi bharna padta, resume se data auto-fill ho jata h.

    """

    text = ''

    try:

        if resume_file.name.endswith('.pdf'):

            reader = PyPDF2.PdfReader(resume_file)

            text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])

        elif resume_file.name.endswith('.docx'):

            doc = Document(resume_file)

            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])

        else:

            return {'error': 'Unsupported file type.'}



        if not text.strip():

            return {'error': 'No text could be extracted from the file'}



        # Use the new robust extraction function

        skills_clean = extract_skills_from_text(text)



        experience_years = 0

        matches = re.findall(r'(\d+)\s*(?:year|yr|years|yrs)\s*(?:of)?\s*(?:exp|experience)', text, re.IGNORECASE)

        if matches:

            experience_years = max(int(m) for m in matches)



        return {

            'skills': skills_clean,

            'experience_years': experience_years,

            'analysis_method': 'enhanced_rule_match'

        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        return {'error': f'Analysis error: {str(e)}'}



# --- SKILL EXTRACTION LOGIC ---



def extract_skills_from_text(text):

    """

    Robust skill extraction using a combination of ML skill list (if available)

    and a comprehensive static fallback list.

    """

    if not text:

        return []

        

    # 1. Preprocess text

    text_lower = text.lower()

    # Replace common separators with spaces to ensure boundaries

    text_clean = re.sub(r'[,/()\[\]]', ' ', text_lower)

    # Keep only allowed characters for skills (alphanumeric, +, #, ., -)

    # We allow space inside skills, but we'll match against known phrases

    

    skills_found = set()

    

    # 2. Define Comprehensive Static Skill List (The "Black Box" Fix)

    # This ensures we catch skills even if the ML model misses them

    STATIC_SKILLS = {

        # Languages

        'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'php', 'ruby', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'dart', 'lua', 'perl', 'objective-c', 'assembly', 'vba',

        # Web Frontend

        'html', 'css', 'html5', 'css3', 'react', 'react.js', 'angular', 'vue', 'vue.js', 'svelte', 'jquery', 'bootstrap', 'tailwind', 'sass', 'less', 'webpack', 'babel',

        # Web Backend

        'django', 'flask', 'fastapi', 'node.js', 'express.js', 'spring', 'spring boot', 'laravel', 'rails', 'asp.net', 'flet', 'tkinter',

        # Mobile

        'flutter', 'react native', 'android', 'ios', 'xamarin', 'ionic',

        # Data Science & ML

        'machine learning', 'deep learning', 'data analysis', 'data science', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'opencv', 'nlp', 'computer vision', 'matplotlib', 'seaborn', 'plotly', 'jupyter',

        # Databases

        'sql', 'mysql', 'postgresql', 'sqlite', 'sqlite3', 'mongodb', 'redis', 'cassandra', 'oracle', 'sql server', 'sqlalchemy', 'firebase', 'dynamodb',

        # DevOps & Cloud

        'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'aws', 'amazon web services', 'azure', 'microsoft azure', 'google cloud', 'gcp', 'terraform', 'ansible', 'circleci', 'heroku', 'digitalocean',

        # Tools & Others
        'linux', 'bash', 'shell', 'jira', 'trello', 'slack', 'excel', 'power bi', 'tableau', 'figma', 'canva', 'photoshop', 'illustrator', 'premiere pro', 'automation', 'selenium', 'chatgpt', 'deepseek', 'llm', 'generative ai',
        
        # Office & Productivity
        'ms word', 'microsoft word', 'word', 'ms excel', 'microsoft excel', 'ms powerpoint', 'microsoft powerpoint', 'powerpoint',
        'ms office', 'microsoft office', 'office 365', 'outlook', 'access', 'sharepoint', 'onenote',
        'google docs', 'google sheets', 'google slides', 'spreadsheet', 'word processing'

    }

    

    # 3. Merge with ML list if available

    search_list = STATIC_SKILLS.copy()

    if models_loaded and 'SKILL_LIST' in globals():

        # Add ML skills but filter out very short ones that cause noise

        for s in SKILL_LIST:

            s_str = str(s).strip().lower()

            if len(s_str) > 2:

                search_list.add(s_str)

    

    # 4. Perform Extraction

    # We sort by length descending to match "Machine Learning" before "Learning"

    sorted_skills = sorted(list(search_list), key=len, reverse=True)

    

    # Create a single regex pattern for efficiency? 

    # No, for 1000+ skills, iterating might be safer to avoid regex complexity limits,

    # but a compiled regex of joined terms is usually faster.

    # Let's try direct substring search with boundary checks for accuracy.

    

    # Normalize text for search: add spaces around to help with boundaries

    # But we need to preserve "c++" etc.

    # Let's use a custom tokenizer approach or simple iteration with boundary check.

    

    # Simple iteration with boundary check

    for skill in sorted_skills:

        # Escape special chars like +, .

        pattern = r'(?:^|[\s,./(\[])' + re.escape(skill) + r'(?:$|[\s,./)\]])'

        if re.search(pattern, text_clean):

            skills_found.add(skill)

            

    # 5. Special Handling for C / C++ / C#

    # "C" is hard because it matches "C" in "Cloud", "Computer", etc.

    # We only match "C" if it's "C Language", "C Programming", or in a list like "C, C++, Java"

    if re.search(r'\bc\b', text_clean):

        # Check context: "C," or ", C" or "C " followed by language/programming

        if re.search(r'(?:^|[\s,])c(?:$|[\s,])', text_clean):

             # Heuristic: if "C++" is there, "C" might be there too. 

             # But often "C" just means "C". 

             # Let's be conservative: Only add "C" if "C++" is NOT the only match, 

             # or if it explicitly looks like a list.

             # For now, let's add it but rely on clean_skill_list to maybe filter?

             # Actually, clean_skill_list allows single letter 'c'.

             # Let's check for "C Programming" or comma separated

             if re.search(r'c\s+programming|language|developer', text_clean) or re.search(r',\s*c\s*,', text_clean):

                 skills_found.add('c')

    

    # 6. Clean and Normalize

    return clean_skill_list(list(skills_found))





# --- PERSONALITY LOGIC ---



def calculate_personality_scores(post_data):

    """

    [HINGLISH]

    Use: Ye `views.py` me personality test submit hone par use hota h.

    Why: User ke answers (1-5 scale) ko Big Five traits (Extraversion, etc.) me convert karne ke liye.

    Effect: Hame user ki personality ka score milta h jo career matching me help karta h.

    """

    scores = {

        'extraversion': 0,

        'agreeableness': 0,

        'conscientiousness': 0,

        'emotional_stability': 0,

        'openness': 0

    }

    

    # Correct mapping for the 10 questions to the Big Five traits

    trait_map = {

        'extraversion': [1, 6],

        'agreeableness': [2, 7],

        'conscientiousness': [3, 8],

        'emotional_stability': [4, 9],

        'openness': [5, 10],

    }

    

    # Calculate scores based on the actual question numbers

    for trait, q_nums in trait_map.items():

        for q_num in q_nums:

            # Safely get the score (default to 3 if missing or invalid)

            score = int(post_data.get(f'question_{q_num}', 3))

            scores[trait] += score

    

    # Normalize scores to 1-10 scale (since total is 10 max per trait)

    for trait in scores.keys():

        scores[trait] = max(1, min(10, scores[trait])) 



    return scores



def determine_mbti_type(scores):

    """

    [HINGLISH]

    Use: Ye `views.py` me personality result calculate karte waqt use hota h.

    Why: Big Five scores ko MBTI type (e.g., INTJ, ENFP) me convert karne ke liye.

    Effect: User ko ek familiar personality type milta h jo career matching me use hota h.

    """

    # Simplified mapping (not scientifically accurate but functional for this app)

    # E/I based on Extraversion (score >= 6 for E)

    e_i = 'E' if scores['extraversion'] >= 6 else 'I'

    # S/N based on Openness (score >= 6 for N)

    s_n = 'N' if scores['openness'] >= 6 else 'S'

    # T/F based on Conscientiousness/Agreeableness (High Conscientiousness/Low Agreeableness for T)

    t_f = 'T' if scores['conscientiousness'] >= 6 and scores['agreeableness'] < 6 else 'F'

    # J/P based on Conscientiousness (score >= 6 for J)

    j_p = 'J' if scores['conscientiousness'] >= 6 else 'P'

    

    return f"{e_i}{s_n}{t_f}{j_p}"



def get_key_strengths(personality_type, scores):

    """

    [HINGLISH]

    Use: Ye `views.py` me personality result page par dikhane ke liye use hota h.

    Why: User ko batane ke liye ki unki personality ke hisab se unki strengths kya hain.

    Effect: User ko confidence milta h aur wo apne career me in strengths ko use kar sakte h.

    """

    strengths_map = {

        'INTJ': ['Strategic Thinking', 'Analytical Mind', 'Independent Worker'],

        'ENTP': ['Innovative Thinking', 'Adaptability', 'Creative Problem Solving'],

        'INFP': ['Creativity', 'Empathy', 'Authenticity'],

        'ISTJ': ['Reliability', 'Attention to Detail', 'Practical Thinking']

    }

    return strengths_map.get(personality_type, ['Adaptability', 'Problem Solving', 'Learning Ability'])



def get_career_recommendations(personality_type, scores):

    """

    [HINGLISH]

    Use: Ye sirf `personality_result` page par quick suggestions ke liye use hota h.

    Why: Personality test ke turant baad kuch basic careers dikhane ke liye.

    Effect: User ko instant feedback milta h (Main matching dashboard par hoti h).

    """

    career_map = {

        'INTJ': [{'title': 'Data Scientist'}, {'title': 'Software Architect'}],

        'ENTP': [{'title': 'Entrepreneur'}, {'title': 'Product Manager'}],

        'INFP': [{'title': 'Graphic Designer'}, {'title': 'UX Designer'}],

        'ISTJ': [{'title': 'Financial Analyst'}, {'title': 'Systems Analyst'}]

    }

    return career_map.get(personality_type, [{'title': 'Business Analyst'}, {'title': 'Project Coordinator'}])





# --- CAREER RECOMMENDATION LOGIC ---



def generate_career_recommendations(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me dashboard load hone par call hota h.

    Why: Ye main function h jo ML models ko run karke best careers dhundhta h aur DB me save karta h.

    Effect: User ko dashboard par personalized career suggestions milte h.

    """

    

    # 1. Clear existing recommendations to avoid duplicates in DB

    CareerRecommendation.objects.filter(user_profile=profile).delete()



    # 2. Get matches using the enhanced ML function

    matches = enhanced_find_career_matches(profile)

    

    # FIXED: Fallback if no matches found (e.g. strict filtering returned nothing)

    if not matches:

        print("⚠️ No matches found with enhanced logic. Using fallback.")

        matches = enhanced_simple_match_fallback(profile, SkillAssessment.objects.filter(user_profile=profile))

    

    # 3. Save top 10 recommendations

    for match in matches[:10]:

        try:

            career_title = match['title']

            

            # --- DATA SYNC: Find full details from Global DataFrame (CAREER_DF) ---

            # This ensures we store Skills, Description, etc., not just the Title.

            career_data_row = {}

            if 'CAREER_DF' in globals() and not CAREER_DF.empty:

                # Find the row in CSV matching this title

                found_rows = CAREER_DF[CAREER_DF['career_name'] == career_title]

                if not found_rows.empty:

                    career_data_row = found_rows.iloc[0].to_dict()



            # Extract fields safely (handle missing data)

            description = match.get('description') or career_data_row.get('description', 'A highly recommended career path.')

            req_skills = career_data_row.get('required_skills', '')

            category = career_data_row.get('category', 'Technology') # Default if missing

            education = career_data_row.get('education_required', '')



            # --- DB UPDATE: Get or Create Career Object ---

            # FIXED: Provide defaults for mandatory fields to prevent IntegrityError on create

            # REMOVED: 'category' field as it does not exist in Career model

            defaults = {

                'description': description,

                'required_skills': req_skills,

                'education_required': education,

                'average_salary': float(career_data_row.get('average_salary', 50000.00) or 50000.00),

                'job_growth_rate': float(career_data_row.get('job_growth_rate', 0.05) or 0.05),

                'work_environment': 'Office/Remote', # Default

            }

            

            career_obj, created = Career.objects.get_or_create(

                title=career_title,

                defaults=defaults

            )



            if not created:

                # Update fields if object already exists

                career_obj.description = description

                if req_skills:

                    career_obj.required_skills = req_skills

                if education:

                    career_obj.education_required = education

                

                # Update stats if they are 0/default

                if not career_obj.average_salary:

                    career_obj.average_salary = defaults['average_salary']

                if not career_obj.job_growth_rate:

                    career_obj.job_growth_rate = defaults['job_growth_rate']

                

                career_obj.save()

            

            # --- CREATE RECOMMENDATION ENTRY ---

            CareerRecommendation.objects.create(

                user_profile=profile,

                recommended_career=career_obj,

                match_score=match['match_score'],

                reasoning=f"Match based on skills ({round(match.get('skills_match', 0), 1)}%), experience, and market fit."

            )

            

        except Exception as e:

            print(f"Error creating recommendation for {match.get('title', 'Unknown')}: {e}")

            continue



            print(f"Error creating recommendation for {match.get('title', 'Unknown')}: {e}")

            continue



class SkillWrapper:

    """Helper class to treat text-based skills same as DB objects"""

    def __init__(self, name, level='intermediate', years=0):

        self.skill_name = name

        self.skill_level = level

        self.years_of_experience = years



def get_combined_user_skills(profile):

    """

    Combines skills from SkillAssessment DB and profile.skills text.

    Returns a list of SkillWrapper-like objects.

    """

    # 1. Get DB skills

    db_skills = list(SkillAssessment.objects.filter(user_profile=profile))

    db_skill_names = {s.skill_name.lower() for s in db_skills}

    

    combined_skills = db_skills.copy()

    

    # 2. Get Text skills (from Resume/Profile)

    if profile.skills:

        # Use smart split or simple split

        text_skills = [s.strip() for s in profile.skills.split(',') if s.strip()]

        for s_name in text_skills:

            if s_name.lower() not in db_skill_names:

                # Add as wrapper object

                # Default to 'intermediate' and profile.experience_years

                combined_skills.append(SkillWrapper(

                    name=s_name, 

                    level='intermediate', 

                    years=profile.experience_years

                ))

                db_skill_names.add(s_name.lower())

                

    return combined_skills



def build_career_profile(career_row):

    """

    Constructs a text profile for a career from its data row.

    Used for vectorization and similarity matching.

    """

    # Combine relevant fields into a single string

    return f"{career_row.get('career_name', '')} {career_row.get('description', '')} {career_row.get('required_skills', '')} {career_row.get('education_required', '')}"



def compute_match_percentage(user_profile, career_obj, ml_score, user_skills):

    """

    Calculates the final match percentage based on ML score, skills, experience, and personality.

    Weights: ML (50%), Skills (30%), Experience (10%), Personality (10%)

    """

    # 1. ML Score (0-100)

    # ml_score is cosine similarity (0-1), so multiply by 100

    weighted_ml = min(100, ml_score * 100) * 0.50

    

    # 2. Skills Match (0-100)

    skills_score = calculate_skills_match_bonus(user_skills, career_obj)

    weighted_skills = skills_score * 0.30

    

    # 3. Experience Match (0-100)

    # Cap experience score at 10 years

    total_exp = sum(s.years_of_experience for s in user_skills)

    exp_score = min(100, (total_exp / 10) * 100)

    weighted_exp = exp_score * 0.10

    

    # 4. Personality Match (0-100)

    # Use the helper to get a multiplier, then convert to score

    career_dict = {'career_name': career_obj.title}

    p_multiplier = calculate_personality_bonus(user_profile.personality_type, career_dict)

    

    if p_multiplier >= 1.15:

        p_score = 100

    elif p_multiplier >= 1.10:

        p_score = 80

    elif p_multiplier > 1.0:

        p_score = 60

    else:

        p_score = 50 # Neutral/Default

        

    weighted_personality = p_score * 0.10

    

    final_score = weighted_ml + weighted_skills + weighted_exp + weighted_personality

    return min(99.9, final_score)



def enhanced_find_career_matches(user_profile):

    """

    Core recommendation engine using ML (Cosine Similarity) and Rule-based filtering.

    Restored to original working state.

    """

    profile = user_profile

    

    # 1. Get User Skills (Combined from DB and Profile Text)

    user_skills = get_combined_user_skills(profile)

    

    # Fallback if no skills

    if not user_skills:

        return enhanced_simple_match_fallback(profile, user_skills)



    # 2. Prepare Profile Text for ML

    # Weighted skills string: repeat skill name based on experience

    user_skills_text = ' '.join([f"{skill.skill_name} " * (int(skill.years_of_experience) + 1) for skill in user_skills])

    

    interests = getattr(profile, 'interests', '')

    current_field = getattr(profile, 'current_field', '')

    

    profile_text = f"{profile.education_level} {profile.gender} {profile.personality_type or ''} {user_skills_text} {interests} {current_field}"

    

    # 3. ML Model Matching

    if not models_loaded or not profile_text.strip():

        return enhanced_simple_match_fallback(profile, user_skills)

    

    try:

        user_vector = VECTORIZER.transform([profile_text])

        similarity_scores = cosine_similarity(user_vector, CAREER_VECTORS).flatten()

        

        # Sort by similarity score (descending)

        ranked_indices = np.argsort(similarity_scores)[::-1]

        

        matches = []

        seen_titles = set()

        

        # Process top matches

        for index in ranked_indices[:100]:

            if index >= len(CAREER_DF):

                continue

                

            career_row = CAREER_DF.iloc[index]

            career_title = career_row['career_name'].strip()

            

            # Duplicate Check

            clean_title = clean_title_for_merge(career_title)

            if clean_title in seen_titles:

                continue

            

            # Get Career Object (for required skills)

            try:

                career_obj = Career.objects.get(title=career_title)

            except Career.DoesNotExist:

                # Create temp object if not in DB

                class TempCareer:

                    def __init__(self, title, req_skills):

                        self.title = title

                        self.required_skills = req_skills

                career_obj = TempCareer(career_title, career_row.get('required_skills', ''))

            # Calculate Final Weighted Score

            ml_raw_score = similarity_scores[index]

            final_score = compute_match_percentage(profile, career_obj, ml_raw_score, user_skills)

            

            # Filter out very low scores (garbage matches)

            if final_score < 10:

                continue



            # Calculate skills match for display (0-100 scale)

            skills_match_display = calculate_skills_match_bonus(user_skills, career_obj)



            seen_titles.add(clean_title)

            

            matches.append({

                'career_id': career_row.get('career_id', index),

                'title': career_title,

                'match_score': round(final_score, 1),

                'description': career_row.get('description', 'No description available')[:150] + '...',

                'skills_match': skills_match_display,

            })

            

            if len(matches) >= 10:

                break

        

        # If no valid matches found via ML, use fallback

        if not matches:

             return enhanced_simple_match_fallback(profile, user_skills)

                

        return matches

        

    except Exception as e:

        print(f"Error in enhanced_find_career_matches: {e}")

        return enhanced_simple_match_fallback(profile, user_skills)



def calculate_skills_match_bonus(user_skills, career_obj):

    """

    Calculates skills match score (0-100).

    Restored to original set-based matching.

    """

    if not user_skills:

        return 0

        

    # Get required skills string

    req_skills_str = ""

    if hasattr(career_obj, 'required_skills'):

        req_skills_str = career_obj.required_skills

    elif isinstance(career_obj, dict):

        req_skills_str = career_obj.get('required_skills', '')

        

    if not req_skills_str:

        return 0

        

    # Parse required skills

    required_skills_list = smart_split_skills(req_skills_str)

    required_skills_set = set([s.strip().lower() for s in required_skills_list if s.strip()])

    

    if not required_skills_set:

        return 0

        

    # Parse user skills

    user_skill_names = {skill.skill_name.lower() for skill in user_skills}

    

    # Calculate intersection

    matching_skills = required_skills_set.intersection(user_skill_names)

    match_count = len(matching_skills)

    

    # Calculate percentage

    # Heuristic: 5+ skills or 50% match is good

    if match_count >= 5:

        base_score = 100

    elif match_count >= 3:

        base_score = 80

    elif match_count >= 1:

        base_score = 40

    else:

        base_score = 0

        

    # Add small bonus for skill level

    level_bonus = 0

    for skill in user_skills:

        if skill.skill_name.lower() in matching_skills:

            if getattr(skill, 'skill_level', 'intermediate').lower() == 'expert':

                level_bonus += 5

            elif getattr(skill, 'skill_level', 'intermediate').lower() == 'advanced':

                level_bonus += 3

                

    return min(100, base_score + level_bonus)

    

def calculate_personality_bonus(personality_type, career):

    """

    [HINGLISH]

    Use: Ye `enhanced_find_career_matches` me use hota h.

    Why: Agar user ki personality job role se match karti h (e.g., Introvert -> Coder), to bonus milta h.

    Effect: User ko wo jobs milti h jisme wo khush rahenge.

    """

    if not personality_type or personality_type == 'Not assessed':

        return 1.0

    

    career_title = career['career_name'].lower()

    

    # Define primary personality alignments for different career categories

    type_career_map = {

        # Analytical, Strategic, Technical (INTx, ENTx, ISTJ)

        'tech_analyt': ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'ISTJ'],

        # Creative, People-focused (INFx, ENFx, ISFP, ESFP)

        'creative_huma': ['INFJ', 'INFP', 'ENFJ', 'ENFP', 'ISFP', 'ESFP'],

        # Practical, Organized, Management (ESTJ, ISFJ, ISTP, ESTP)

        'admin_opera': ['ESTJ', 'ISFJ', 'ISTP', 'ESTP'],

    }

    

    # Define career category keywords

    career_keywords = {

        'tech_analyt': ['software', 'developer', 'engineer', 'data', 'ai', 'machine learning', 'cybersecurity', 'analyst', 'cloud'],

        'creative_huma': ['designer', 'writer', 'ux', 'ui', 'counselor', 'hr', 'marketing', 'product manager'],

        'admin_opera': ['manager', 'finance', 'logistics', 'supervisor', 'admin', 'operations', 'project coordinator'],

    }

    

    # Determine career category

    career_category = 'none'

    for cat, keywords in career_keywords.items():

        if any(keyword in career_title for keyword in keywords):

            career_category = cat

            break

            

    # Apply bonus based on alignment (Max multiplier 1.15)

    if (career_category == 'tech_analyt' and personality_type in type_career_map['tech_analyt']):

        return 1.15 

    elif (career_category == 'creative_huma' and personality_type in type_career_map['creative_huma']):

        return 1.15

    elif (career_category == 'admin_opera' and personality_type in type_career_map['admin_opera']):

        return 1.15

    

    # Secondary match (e.g., Creative in Tech)

    if (career_category == 'tech_analyt' and personality_type in type_career_map['creative_huma'] and 'designer' in career_title):

        return 1.10 # UX Designer (Creative in Tech)

        

    return 1.0



def enhanced_simple_match_fallback(profile, user_skills):

    """

    [HINGLISH]

    Use: Ye tab use hota h jab ML models fail ho jaye ya load na ho.

    Why: Taaki site crash na kare aur user ko kuch na kuch recommendations jarur dikhe.

    Effect: System robust rehta h, hamesha backup ready rehta h.

    """

    # Ensure user_skills is a list (it might be passed as QuerySet or list)

    if hasattr(user_skills, 'exists'): # QuerySet

         user_skills = list(user_skills)

         

    user_skill_names = {skill.skill_name.lower() for skill in user_skills}

    matches = []

    seen_titles = set()

    

    for _, career in CAREER_DF.iterrows():

        career_title = career['career_name']

        clean_title = clean_title_for_merge(career_title)

        

        if clean_title in seen_titles:

            continue

            

        seen_titles.add(clean_title)



        score = 0

        

        # Skills matching (60 points max)

        # Use the robust substring matching logic

        skills_score = calculate_skills_match_bonus(user_skills, {'required_skills': str(career.get('required_skills', ''))})

        score += (skills_score / 100) * 60 # Scale 0-100 to 0-60

        

        # Experience bonus (20 points max)

        

        # Experience bonus (20 points max)

        total_experience = sum(s.years_of_experience for s in user_skills)

        if total_experience >= 3:

            score += 20

        elif total_experience >= 1:

            score += 10

        

        # Education bonus (10 points max)

        if profile.education_level and 'bachelor' in str(profile.education_level).lower():

            score += 10

        

        # Personality bonus (10 points max)

        if profile.personality_type and profile.personality_type != 'Not assessed':

            # Use personality bonus logic to check for a match

            if calculate_personality_bonus(profile.personality_type, career) > 1.0:

                 score += 10

        

        final_score = max(10, min(98, round(score, 2)))

        

        # Only include if score is decent (lowered threshold for fallback)

        if final_score > 5:

            matches.append({

                'title': career['career_name'],

                'match_score': final_score,

                'description': str(career.get('description', ''))[:100] + '...',

                'skills_match': 0, # Cannot calculate accurately in fallback

            })

        

    # If still no matches, return top careers by default

    if not matches:

        # Return top 5 careers from DF as last resort

        for i, row in CAREER_DF.head(5).iterrows():

             matches.append({

                'title': row['career_name'],

                'match_score': 10.0, # Minimum score

                'description': str(row.get('description', ''))[:100] + '...',

                'skills_match': 0,

            })

            

    return sorted(matches, key=lambda x: x['match_score'], reverse=True)[:10]



def fetch_live_market_data(job_title):

    """

    Fetches real-time job market data using JSearch API (RapidAPI).

    Falls back to CSV data if API fails or key is missing.

    Results are cached for 24 hours to save quota.

    """

    clean_key = clean_title_for_merge(job_title)

    cache_key = f"market_data_{clean_key}"

    

    # 1. Check Cache

    cached_data = cache.get(cache_key)

    if cached_data:

        return cached_data



    # 2. Try API (if key exists)

    api_key = getattr(settings, 'RAPIDAPI_KEY', None)

    

    if api_key and api_key != 'YOUR_RAPIDAPI_KEY_HERE':

        try:

            url = "https://jsearch.p.rapidapi.com/search"

            querystring = {"query": f"{job_title} jobs in USA", "num_pages": "1"} # Defaulting to USA for broad data

            headers = {

                "X-RapidAPI-Key": api_key,

                "X-RapidAPI-Host": getattr(settings, 'RAPIDAPI_HOST', 'jsearch.p.rapidapi.com')

            }

            

            response = requests.get(url, headers=headers, timeout=5)

            

            if response.status_code == 200:

                data = response.json()

                jobs = data.get('data', [])

                

                if jobs:

                    # Calculate simple stats from the first page of results

                    salaries = []

                    for job in jobs:

                        # Extract salary if available (it"s often unstructured)

                        # This is a simplification; real parsing is complex

                        pass 

                        

                    # For now, JSearch mainly gives active job listings. 

                    # We can use the COUNT of jobs as a proxy for demand/growth

                    job_count = len(jobs)

                    

                    # Heuristic for growth/demand based on live job volume

                    growth_rate = 0.05

                    if job_count > 20: growth_rate = 0.15

                    elif job_count > 10: growth_rate = 0.08

                    

                    result = {

                        'job_growth_rate': growth_rate,

                        'demand_level': 'High' if job_count > 15 else 'Medium',

                        'source': 'Live API',

                        'job_count': job_count

                    }

                    

                    # Cache the result

                    cache.set(cache_key, result, timeout=86400) # 24 hours

                    return result

                    

        except Exception as e:

            print(f"API Fetch Error for {job_title}: {e}")



    # 3. Fallback removed (CSV deleted)

    return {

        'job_growth_rate': get_default_growth_by_title(job_title) / 100,

        'demand_level': generate_demand_level_by_title(job_title),

        'source': 'Heuristic'

    }



def enhance_recommendation_with_market_data(rec):

    """

    Attaches market data to a recommendation object using live data or fallback.

    """

    # Initialize default structure

    rec.market_data = {

        'job_growth_rate': 0.05,

        'demand_level': 'Medium',

        'salary_range': format_salary(rec.recommended_career.average_salary),

        'source': 'Default'

    }

    

    try:

        # Fetch data (Live -> CSV -> Heuristic)

        live_data = fetch_live_market_data(rec.recommended_career.title)

        

        # Update recommendation object temporary fields (not saving to DB to avoid thrashing)

        rec.market_data.update(live_data)

        

        # Format growth rate for display (e.g., 0.15 -> 15.0)

        rec.market_data['growth_percentage'] = round(rec.market_data['job_growth_rate'] * 100, 1)

        

        # If API returns salary (future implementation), update here. 

        # For now, keep DB salary but formatted

        

    except Exception as e:

        print(f"Error adhering market data: {e}")

        

    return rec

    from .models import Career

    

    # 1. Get Required Skills for Career

    required_skills_raw = ""

    

    # Try DB first

    career_obj = Career.objects.filter(title=target_career).first()

    if career_obj and career_obj.required_skills:

        required_skills_raw = career_obj.required_skills

    

    # Fallback to CSV

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

    if user_skills:

        # user_skills can be a string or list

        if isinstance(user_skills, str):

            user_skills_list = [s.strip() for s in user_skills.split(',') if s.strip()]

            user_skills_set = set([s.lower() for s in user_skills_list])

        elif isinstance(user_skills, list):

            # Assuming list of SkillWrapper or similar

            user_skills_set = set([s.skill_name.lower() for s in user_skills])

            

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







def update_user_profile_skills(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me tab use hota h jab user koi naya skill add/delete karta h.

    Why: `SkillAssessment` table aur `UserProfile` table ko sync rakhne ke liye.

    Effect: Profile me hamesha latest skills ki list rehti h.

    """

    try:

        user_skills = SkillAssessment.objects.filter(user_profile=profile)

        skills_list = [skill.skill_name for skill in user_skills]

        profile.skills = ', '.join(skills_list)

        profile.save()

    except Exception as e:

        print(f"❌ Error updating profile skills: {e}")



def generate_skill_recommendations_based_on_profile(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me skills page par suggestions dikhane ke liye use hota h.

    Why: User ko batane ke liye ki wo aur kya seekh sakte h.

    Effect: User engagement badhta h aur wo naye skills add karte h.

    """

    try:

        current_skills = {skill.skill_name.lower() for skill in SkillAssessment.objects.filter(user_profile=profile)}

        

        # Get trending skills from dataset (using SKILLS_DF if available)

        trending_skills = []

        if SKILLS_DF is not None and not SKILLS_DF.empty and 'skill_name' in SKILLS_DF.columns:

            trending_skills = SKILLS_DF['skill_name'].tolist()

        

        # Fallback if no data or SKILLS_DF is missing

        if not trending_skills:

            trending_skills = [

                'Python', 'JavaScript', 'Machine Learning', 'Data Analysis', 

                'Cloud Computing', 'React', 'SQL', 'Project Management',

                'Communication', 'Problem Solving', 'Team Leadership'

            ]

        

        # Get career-based recommendations

        career_based_skills = generate_career_based_skill_recommendations(profile)

        

        # Combine and remove duplicates

        all_recommendations = list(set(trending_skills + career_based_skills))

        

        # Filter out skills user already has

        recommended_skills = []

        for skill in all_recommendations:

            if skill.lower() not in current_skills:

                recommended_skills.append(skill)

            if len(recommended_skills) >= 8: # Limit to 8 recommendations

                break

        

        return recommended_skills

    except Exception as e:

        print(f"Error generating skill recommendations: {e}")

        return ['Python', 'Data Analysis', 'Communication', 'Problem Solving']



def generate_career_based_skill_recommendations(profile):

    """

    [HINGLISH]

    Use: Ye `generate_skill_recommendations_based_on_profile` me use hota h.

    Why: User ke top careers ke hisab se skills suggest karne ke liye.

    Effect: Recommendations relevant hoti h, random nahi.

    """

    try:

        career_based_skills = []

        

        # Get user's top career matches

        top_careers = CareerRecommendation.objects.filter(

            user_profile=profile

        ).order_by('-match_score')[:3]

        

        for career_rec in top_careers:

            career = career_rec.recommended_career

            # Use smart_split_skills here if career.required_skills stores data with complex formatting

            required_skills = smart_split_skills(career.required_skills) if career.required_skills else []

            career_based_skills.extend([skill.strip() for skill in required_skills if skill.strip()])

        

        # Add skills based on personality type

        personality_skills = get_skills_by_personality(profile.personality_type)

        career_based_skills.extend(personality_skills)

        

        return list(set(career_based_skills)) 

    except Exception as e:

        print(f"Error generating career-based skills: {e}")

        return []



def fetch_youtube_resources(skill_name, max_results=3):

    """

    Fetches videos for a skill using YouTube Data API.

    """

    cache_key = f"yt_res_{skill_name}_{max_results}"

    cached = cache.get(cache_key)

    if cached:

        return cached

        

    api_key = getattr(settings, 'YOUTUBE_API_KEY', None)

    if not api_key or api_key == 'YOUR_YOUTUBE_API_KEY_HERE':

        return []



    try:

        url = "https://www.googleapis.com/youtube/v3/search"

        params = {

            'part': 'snippet',

            'q': f"{skill_name} tutorial for beginners",

            'type': 'video',

            'maxResults': max_results,

            'key': api_key

        }

        

        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:

            data = response.json()

            items = data.get('items', [])

            

            resources = []

            for item in items:

                video_id = item['id']['videoId']

                snippet = item['snippet']

                resources.append({

                    'title': snippet['title'],

                    'channel': snippet['channelTitle'],

                    'url': f"https://www.youtube.com/watch?v={video_id}",

                    'thumbnail': snippet['thumbnails']['high']['url']

                })

            

            if resources:

                cache.set(cache_key, resources, timeout=86400 * 3) # Cache for 3 days

                return resources

                

    except Exception as e:

        print(f"YouTube API Error for {skill_name}: {e}")

        

    return []



def get_learning_resources(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me Learning Hub page par resources dikhane ke liye use hota h.

    Why: User ke target career aur missing skills ke hisab se curated content lane ke liye.

    Effect: User ko personalized roadmap aur videos milte h.

    """

    from .models import CareerRoadmap, LearningResource

    from .utils import normalize_skill

    

    # 1. Determine Target Career (Top Recommendation)

    target_career = "Software Engineer" # Default

    recommendation = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score').first()

    

    if recommendation:

        target_career = recommendation.recommended_career.title

        

    # 2. Get Roadmap (Dynamic from DB)

    # First, try to get from DB

    roadmap_steps = CareerRoadmap.objects.filter(career__title=target_career).order_by('step_number')

    

    if not roadmap_steps.exists():

        # If not in DB, try to generate/seed it

        generate_dynamic_roadmap(target_career)

        roadmap_steps = CareerRoadmap.objects.filter(career__title=target_career).order_by('step_number')

        

    # Convert to list of dicts for template

    roadmap = []

    if roadmap_steps.exists():

        for step in roadmap_steps:

            roadmap.append({

                'step': step.step_number,

                'title': step.title,

                'desc': step.description

            })

    else:

        # Fallback to empty if DB generation failed

        roadmap = []



    # 3. Identify Missing Skills (Gap Analysis)

    user_skills_objs = SkillAssessment.objects.filter(user_profile=profile)

    user_skills_list = [s.skill_name for s in user_skills_objs]

    if profile.skills:

        user_skills_list.extend([s.strip() for s in profile.skills.split(',') if s.strip()])

    user_skills_str = ", ".join(list(set(user_skills_list)))

    

    gap_data = predict_skill_gaps(user_skills_str, target_career)

    missing_skills = gap_data.get('missing_skills', [])

    

    # 4. Fetch Resources for Missing Skills (Dynamic from DB)

    video_resources = []

    course_resources = []

    

    # If no missing skills (perfect match), show advanced topics

    skills_to_learn = [s['name'] for s in missing_skills]

    if not skills_to_learn:

        skills_to_learn = ['System Design', 'Cloud Computing', 'Leadership']

        

    for skill in skills_to_learn:

        norm_skill = normalize_skill(skill)

        

        # A. Try DB First

        db_resources = LearningResource.objects.filter(skill_tag__iexact=norm_skill)

        

        # B. If no DB resources, Try API

        if not db_resources.exists():

            # Try Fetching from API

            api_videos = fetch_youtube_resources(skill) # Use original name for better search

            

            if api_videos:

                # Save to DB so we don"t hit API again for this skill (or we can just use cache)

                # Let's save to DB to persist "Seeded" state

                for vid in api_videos:

                    LearningResource.objects.get_or_create(

                        url=vid['url'],

                        defaults={

                            'title': vid['title'],

                            'resource_type': 'video',

                            'platform': vid['channel'],

                            'skill_tag': norm_skill,

                            'thumbnail_url': vid['thumbnail']

                        }

                    )

                # Refetch from DB

                db_resources = LearningResource.objects.filter(skill_tag__iexact=norm_skill)

            

        for res in db_resources:

            if res.resource_type == 'video':

                video_resources.append({

                    'skill': skill,

                    'title': res.title,

                    'channel': res.platform,

                    'url': res.url,

                    'thumbnail': res.thumbnail_url or f"https://img.youtube.com/vi/{res.url.split('v=')[-1]}/mqdefault.jpg"

                })

            elif res.resource_type == 'course':

                course_resources.append({

                    'skill': skill,

                    'title': res.title,

                    'platform': res.platform,

                    'url': res.url

                })



    return {

        'target_career': target_career,

        'roadmap': roadmap,

        'videos': video_resources[:6], 

        'courses': course_resources[:4] 

    }



def generate_dynamic_roadmap(career_title):

    """

    Generates a roadmap for a career if it doesn't exist in DB.

    Uses static templates or keyword-based logic.

    """

    from .models import Career, CareerRoadmap

    

    try:

        # Find the career object

        career_obj = Career.objects.filter(title=career_title).first()

        if not career_obj:

            # Should exist if recommended, but safety check

            return



        # 1. Check Static Templates (REMOVED)

        template = None

        

        # 2. If no template, use Keyword Logic

        if not template:

            title_lower = career_title.lower()

            if 'manager' in title_lower or 'lead' in title_lower:

                template = [

                    {'step': 1, 'title': 'Core Competencies', 'desc': 'Master the fundamentals of the domain.'},

                    {'step': 2, 'title': 'Project Management', 'desc': 'Learn Agile, Scrum, and resource management.'},

                    {'step': 3, 'title': 'Team Leadership', 'desc': 'Develop soft skills, conflict resolution, and mentoring.'},

                    {'step': 4, 'title': 'Strategic Planning', 'desc': 'Understand business goals and long-term strategy.'},

                    {'step': 5, 'title': 'Advanced Certification', 'desc': 'PMP, MBA, or specialized leadership certs.'}

                ]

            elif 'designer' in title_lower:

                 template = [

                    {'step': 1, 'title': 'Design Fundamentals', 'desc': 'Color theory, typography, and layout.'},

                    {'step': 2, 'title': 'Tools Mastery', 'desc': 'Figma, Adobe XD, Photoshop, Illustrator.'},

                    {'step': 3, 'title': 'UX Principles', 'desc': 'User research, wireframing, and prototyping.'},

                    {'step': 4, 'title': 'Portfolio Building', 'desc': 'Create real-world projects to showcase skills.'},

                    {'step': 5, 'title': 'Specialization', 'desc': 'Motion design, 3D, or interaction design.'}

                ]

            else:

                # Generic Technical/Professional Fallback

                template = [

                    {'step': 1, 'title': 'Foundations', 'desc': f'Learn the basic principles of {career_title}.'},

                    {'step': 2, 'title': 'Core Tools', 'desc': 'Master the essential software and tools used in the industry.'},

                    {'step': 3, 'title': 'Advanced Concepts', 'desc': 'Deep dive into complex topics and methodologies.'},

                    {'step': 4, 'title': 'Practical Application', 'desc': 'Build projects or gain internship experience.'},

                    {'step': 5, 'title': 'Professional Development', 'desc': 'Networking, resume building, and interview prep.'}

                ]



        # 3. Save to DB

        for step in template:

            CareerRoadmap.objects.create(

                career=career_obj,

                step_number=step['step'],

                title=step['title'],

                description=step['desc']

            )

            

    except Exception as e:

        print(f"Error generating roadmap for {career_title}: {e}")







def calculate_profile_completion(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me dashboard par progress bar dikhane ke liye use hota h.

    Why: User ko motivate karne ke liye ki wo apna profile pura bhare.

    Effect: Gamification add hota h aur data quality improve hoti h.

    """

    completion = 0

    fields_to_check = [

        ('age', 15), ('gender', 15), ('education_level', 15), 

        ('skills', 20), ('personality_type', 15), ('resume_file', 20)

    ]



    for field, percentage in fields_to_check:

        if getattr(profile, field, None) and getattr(profile, field, None) not in ['', 'Not assessed']:

            completion += percentage

    return min(completion, 100)



def generate_personalized_insights(profile):

    """

    [HINGLISH]

    Use: Ye `views.py` me dashboard par personalized messages dikhane ke liye use hota h.

    Why: Dashboard ko static ki jagah dynamic aur personal feel dene ke liye.

    Effect: User ko lagta h ki system unhe samajhta h.

    """

    insights = []



    if profile.experience_years and profile.experience_years >= 5:

        insights.append(f"With {profile.experience_years} years of experience, you're well-positioned for senior roles.")



    # Count skills by words/tokens, not by raw characters

    skills_count = 0

    if profile.skills:

        tokens = [s.strip() for s in re.split(r'[;,\n]', profile.skills) if s.strip()]

        skills_count = len(tokens)

        if skills_count >= 8:

            insights.append(f"You have a strong skill set with {skills_count} documented skills.")

        elif skills_count <= 3:

            insights.append("Consider adding more skills to enhance your career opportunities.")



    if profile.personality_type and profile.personality_type != 'Not assessed':

        insights.append(f"Your {profile.personality_type} personality suggests strengths in creative problem-solving.")



    if not insights:

        insights.append("Complete your profile to get personalized career insights.")



    return insights

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

