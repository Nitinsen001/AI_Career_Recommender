from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Sum, Count
from django.utils import timezone
from django.core.files.storage import default_storage

import pandas as pd
import numpy as np
import joblib
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps
import string 

# Resume Parsing Libraries
import PyPDF2
from docx import Document

from .models import UserProfile, Career, PersonalityAssessment, SkillAssessment, CareerRecommendation
from .forms import UserRegistrationForm, UserProfileForm, PersonalityAssessmentForm, SkillAssessmentForm, ResumeUploadForm

# --- 1. CONFIGURATION AND MODEL/DATA LOADING ---

def clean_skill_list(skills):
    """Utility to clean and normalize skill names for display/storage."""
    try:
        keep_single_letter = {'c', 'r'}
        acronyms = {'SQL', 'HTML', 'CSS', 'REST', 'AWS', 'GCP', 'NLP', 'ETL', 'API', 'CI/CD', 'C++', 'C#', 'R', 'AI', 'IOT'}
        cleaned = []
        seen = set()
        for s in skills:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if not s:
                continue
            
            s = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\-]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()

            sl = s.lower()
            if len(sl) < 2 and sl not in keep_single_letter:
                continue
            if len(sl.split()) > 4:
                continue
            if re.fullmatch(r'[\W_]+', sl):
                continue

            # Title case, then restore acronyms and special tokens
            nice = s.title()
            tokens = []
            for token in re.split(r'(\/|\+\+|#)', nice):
                up = token.upper().strip()
                if up in acronyms or token in ('/','++', '#'):
                    tokens.append(up)
                elif token.strip():
                    tokens.append(token.strip())
            nice = ' '.join(tokens).replace(' / ', '/').replace('Ci/Cd', 'CI/CD').replace('C++', 'C++').replace('C#', 'C#')
            
            key = nice.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(nice)
        return cleaned
    except Exception:
        return list(dict.fromkeys([str(s).strip() for s in skills if str(s).strip()]))

def clean_title_for_merge(title):
    """Clean job/career title: lowercase, remove quotes, remove special chars, trim spaces."""
    if pd.isna(title) or not isinstance(title, str):
        return ""
    text = title.lower().replace('"', '').replace("'", '').strip()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_load_df(filename):
    """Generic safe loader for CSVs with column normalization."""
    try:
        df = pd.read_csv(f'datasets/{filename}')
        # CRITICAL FIX: Normalize column names (e.g., 'Skill Name' -> 'skill_name')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

try:
    # --- HERE ARE THE GLOBAL DATA FRAMES (WITH COLUMN NORMALIZATION) ---
    CAREER_DF = safe_load_df('career_dataset.csv')
    if CAREER_DF is None:
        CAREER_DF = safe_load_df('career_dataset_final.csv')

    if CAREER_DF is None or CAREER_DF.empty:
        raise FileNotFoundError("Career data not found!")
    
    CAREER_DF['clean_key'] = CAREER_DF.get('career_name', pd.Series(['Unknown'] * len(CAREER_DF))).apply(clean_title_for_merge)

    MARKET_DF = safe_load_df('job_market.csv')
    if MARKET_DF is not None and not MARKET_DF.empty and 'job_title' in MARKET_DF.columns:
        MARKET_DF['clean_key'] = MARKET_DF['job_title'].apply(clean_title_for_merge)
    else:
        MARKET_DF = pd.DataFrame(columns=['job_title', 'job_growth_rate', 'location', 'avg_hiring_time_days', 'clean_key']) 

    SKILLS_DF = safe_load_df('skills_dataset.csv')
    PERSONALITY_DF = safe_load_df('personality.csv')

    # ----------------------------------------
except Exception as e:
    print(f"FATAL ERROR: Could not load core datasets: {e}")
    CAREER_DF = pd.DataFrame(columns=['career_name', 'required_skills', 'description', 'career_id', 'education_required', 'average_salary', 'clean_key']) 
    MARKET_DF = pd.DataFrame(columns=['job_title', 'job_growth_rate', 'location', 'avg_hiring_time_days', 'clean_key']) 
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
    print("All ENHANCED ML components loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load ENHANCED ML models: {e}")
    print("Some features will be limited. Please run 'python train_models.py'.")


# --- HELPER FUNCTIONS FOR SKILL CATEGORIZATION ---

def get_skill_category_map():
    """Builds a map from cleaned skill name (lowercase) to its Category."""
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
    """Groups user's extracted skills based on the CSV map."""
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


# --- 2. DECORATOR FOR FALLBACK ---

def fallback_if_models_fail(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not models_loaded:
            if func._name_ == 'predict_personality_type_ml':
                return {'type': 'INFP', 'confidence': 0.5, 'error': 'Models not loaded.'}
            if func._name_ == 'calculate_confidence_score':
                return 0.5
        return func(*args, **kwargs)
    return wrapper

# --- 3. CORE DJANGO VIEWS ---

def home(request):
    """Home/Landing page view"""
    return render(request, 'index.html')

def login_view(request):
    """User login view"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'login.html')

def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            
            profile = UserProfile.objects.create(
                user=user,
                age=form.cleaned_data['age'],
                gender=form.cleaned_data['gender'],
                education_level=form.cleaned_data['education_level'],
                experience_years=form.cleaned_data.get('experience_years', 0),
                skills=form.cleaned_data.get('skills', ''),
                personality_type=form.cleaned_data.get('personality_type', ''),
            )
            
            resume_file = form.cleaned_data.get('resume_file')
            if resume_file:
                analysis_result = analyze_resume_file(resume_file)
                profile.resume_file = resume_file
                profile.resume_filename = resume_file.name
                profile.resume_uploaded_at = timezone.now()
                if 'error' not in analysis_result:
                    profile.skills = ', '.join(analysis_result.get('skills', []))
                    profile.experience_years = analysis_result.get('experience_years', 0)
                    profile.resume_text = f"Skills: {profile.skills}; Exp: {profile.experience_years} years"
            
            profile.save()
            generate_career_recommendations(profile)
            messages.success(request, 'Registration successful! Please log in.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    """User logout view"""
    logout(request)
    return redirect('home')

@login_required
def dashboard(request):
    """User dashboard view"""
    profile = request.user.userprofile
    if not CareerRecommendation.objects.filter(user_profile=profile).exists():
        generate_career_recommendations(profile)

    top_recommendations = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')[:3]
    
    skill_gap_score = 0
    if top_recommendations:
        target_career_title = top_recommendations[0].recommended_career.title 
        user_skills_text = profile.skills
        skill_gaps_data = predict_skill_gaps(user_skills_text, target_career_title)
        skill_gap_score = skill_gaps_data.get('gap_score', 0) * 100
        
    context = {
        'profile_completion': calculate_profile_completion(profile),
        'top_recommendations': top_recommendations,
        'experience_years': profile.experience_years or 0,
        'skills_count': SkillAssessment.objects.filter(user_profile=profile).count(),
        'skill_gap_score': round(skill_gap_score, 1),
        'personality_type': profile.personality_type or 'Not assessed',
        'personalized_insights': generate_personalized_insights(profile),
    }
    return render(request, 'dashboard.html', context)

@login_required
def profile(request):
    """User profile view"""
    profile = request.user.userprofile
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)

    context = {
        'form': form,
        'profile': profile,
        'skills_count': SkillAssessment.objects.filter(user_profile=profile).count(),
        'recommendations_count': CareerRecommendation.objects.filter(user_profile=profile).count(),
        'profile_completion': calculate_profile_completion(profile)
    }
    return render(request, 'profile.html', context)

@login_required
def edit_profile(request):
    """Edit user profile view"""
    profile = request.user.userprofile
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)
    return render(request, 'edit_profile.html', {'form': form})

@login_required
def resume_upload(request):
    """Resume upload and analysis view - FIXED AND CATEGORIZED"""
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        messages.error(request, 'User profile not found.')
        return redirect('dashboard')

    if request.method == 'POST':
        # --- DELETE LOGIC ---
        if 'delete_resume' in request.POST:
            if profile.resume_file:
                if default_storage.exists(profile.resume_file.name):
                    default_storage.delete(profile.resume_file.name)
                
                profile.resume_file = None
                profile.resume_filename = ''
                profile.resume_uploaded_at = None
                profile.skills = ''
                profile.experience_years = 0
                profile.resume_text = ''
                profile.save()
                
                messages.success(request, 'Resume deleted successfully!')
            else:
                messages.info(request, 'No resume found to delete.')
            return redirect('resume')
        
        # --- UPLOAD/REPLACE LOGIC ---
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            
            if resume_file.size > 10 * 1024 * 1024:
                messages.error(request, 'File size too large. Maximum 10MB allowed.')
                return redirect('resume')
            
            valid_extensions = ('.pdf', '.docx')
            if not resume_file.name.lower().endswith(valid_extensions):
                messages.error(request, 'Invalid file type. Only PDF and DOCX files are supported.')
                return redirect('resume')

            try:
                # Analyze resume
                analysis_result = analyze_resume_file(resume_file)

                # Delete old file if exists
                if profile.resume_file and default_storage.exists(profile.resume_file.name):
                    default_storage.delete(profile.resume_file.name)

                # Save new file
                profile.resume_file.save(resume_file.name, resume_file, save=False)
                profile.resume_filename = resume_file.name
                profile.resume_uploaded_at = timezone.now()

                if 'error' not in analysis_result:
                    profile.skills = ', '.join(analysis_result.get('skills', []))
                    profile.experience_years = analysis_result.get('experience_years', 0)
                    profile.resume_text = f"Skills: {profile.skills}; Experience: {profile.experience_years} years"
                    messages.success(request, f'Resume uploaded successfully! Found {len(analysis_result.get("skills", []))} skills.')
                else:
                    messages.warning(request, f'Resume uploaded but analysis failed: {analysis_result["error"]}')
                
                profile.save()
                return redirect('resume')
                
            except Exception as e:
                error_msg = f'Error saving resume: {str(e)}'
                messages.error(request, error_msg)
                return redirect('resume')
        else:
            messages.error(request, 'No file selected. Please choose a file to upload.')
    
    # GET request logic
    profile = UserProfile.objects.get(id=profile.id)
    profile_skills = clean_skill_list([s.strip() for s in (profile.skills or '').split(',') if s.strip()]) if profile.skills else []
    
    # --- CRITICAL NEW LOGIC: SKILL CATEGORIZATION ---
    categorized_skills_data = group_skills_by_category(profile_skills)
    
    # Pass generic analysis result for the Analysis Result Card
    analysis_result = {
        'skills': profile_skills,
        'experience_years': profile.experience_years,
        'analysis_method': 'Model/Fallback'
    }

    # Context for the template
    return render(request, 'resume.html', {
        'form': ResumeUploadForm(), 
        'profile': profile,
        'profile_skills': profile_skills, # Simple list for stats/old sections
        'categorized_skills': categorized_skills_data, # NEW CATEGORIZED DATA for breakdown
        'analysis': analysis_result, # Data for the analysis card
    })

@login_required
def analyze_resume(request):
    """Analyze resume view (Used by the /resume/analyze/ URL path)"""
    return redirect('resume') # Redirect to the main upload page for simplicity

@login_required
def personality_assessment(request):
    """Personality assessment view (Used by /personality/ URL path)"""
    if request.method == 'POST':
        form = PersonalityAssessmentForm(request.POST)
        if form.is_valid():
            assessment = form.save(commit=False)
            assessment.user_profile = request.user.userprofile
            assessment.save()
            return redirect('personality_result')

        messages.error(request, 'Please correct the errors in the assessment form.')
        return render(request, 'personality_test.html', {'form': form, 'profile': request.user.userprofile})
    
    return redirect('take_personality_test')

def personality_result(request):
    """Handles logic for displaying personality test results (UI + logic improved)."""
    profile = request.user.userprofile

    latest_assessment = PersonalityAssessment.objects.filter(user_profile=profile).order_by('-assessment_date').first()
    if not latest_assessment:
        messages.error(request, "No assessment data found. Please take the test first.")
        return redirect('take_personality_test')

    # Compute Big Five scores from the 10 questions (1-5 Likert)
    # Map two questions per trait for better distribution
    trait_map = {
        'extraversion': [1, 6],
        'agreeableness': [2, 7],
        'conscientiousness': [3, 8],
        'emotional_stability': [4, 9],
        'openness': [5, 10],
    }
    raw = {
        1: latest_assessment.question_1,
        2: latest_assessment.question_2,
        3: latest_assessment.question_3,
        4: latest_assessment.question_4,
        5: latest_assessment.question_5,
        6: latest_assessment.question_6,
        7: latest_assessment.question_7,
        8: latest_assessment.question_8,
        9: latest_assessment.question_9,
        10: latest_assessment.question_10,
    }
    scores = {}
    for trait, qs in trait_map.items():
        total = sum(int(raw[q]) for q in qs)  # range 2-10
        # normalize to 1-10 scale already, but ensure bounds
        scores[trait] = max(1, min(10, total))

    # Determine personality type from scores
    personality_type = profile.personality_type or determine_mbti_type(scores)

    # Build enriched recommended careers using datasets
    recs_basic = get_career_recommendations(personality_type, scores)
    enriched_recs = []
    for r in recs_basic:
        title = r.get('title', 'Role')
        clean = clean_title_for_merge(title)
        # attach salary/growth from datasets if available
        md = MARKET_DF[MARKET_DF['clean_key'] == clean].head(1) if not MARKET_DF.empty and 'clean_key' in MARKET_DF.columns else pd.DataFrame()
        cd = CAREER_DF[CAREER_DF['clean_key'] == clean].head(1) if not CAREER_DF.empty and 'clean_key' in CAREER_DF.columns else pd.DataFrame()
        salary = None
        growth = None
        if not md.empty:
            salary = format_salary(md.iloc[0].get('average_salary', 0))
            gr = md.iloc[0].get('job_growth_rate', 0)
            growth = round(float(gr) * 100, 1) if pd.notna(gr) else get_default_growth_by_title(title)
        elif not cd.empty:
            salary = format_salary(cd.iloc[0].get('average_salary', 0))
            growth = get_default_growth_by_title(title)
        else:
            salary = get_default_salary_by_title(title)
            growth = get_default_growth_by_title(title)
        enriched_recs.append({
            'title': title,
            'description': r.get('description', ''),
            'salary': salary,
            'growth': growth,
            'category': r.get('category', ''),
        })

    context = {
        'personality_type': personality_type,
        'key_strengths': get_key_strengths(personality_type, scores),
        'recommended_careers': enriched_recs,
        'overall_match_score': 85,
        # expose scores for UI progress bars
        'extraversion_score': scores['extraversion'],
        'agreeableness_score': scores['agreeableness'],
        'conscientiousness_score': scores['conscientiousness'],
        'emotional_stability_score': scores['emotional_stability'],
        'openness_score': scores['openness'],
    }

    return render(request, 'personality_result.html', context)

@login_required
def take_personality_test(request):
    """Take personality test view (MBTI input or Quiz)"""
    profile = request.user.userprofile

    if request.method == 'POST':
        form = PersonalityAssessmentForm(request.POST)
        if form.is_valid():
            assessment = form.save(commit=False)
            assessment.user_profile = profile
            assessment.save()
            
            scores = calculate_personality_scores(request.POST)
            personality_type = determine_mbti_type(scores)
            
            profile.personality_type = personality_type
            profile.save()

            messages.success(request, f"Personality assessment complete! Type: {personality_type}")
            return redirect('personality_result')

        messages.error(request, 'Please correct the errors in the assessment form.')
        return render(request, 'personality_test.html', {'form': form, 'profile': profile})
    else:
        return render(request, 'personality_test.html', {'form': PersonalityAssessmentForm(), 'profile': profile})

# --- CAREER RECOMMENDATIONS VIEWS ---

@login_required
def career_recommendations(request):
    """Career recommendations view WITH MARKET DATA - FIXED VERSION"""
    profile = request.user.userprofile
    
    if not CareerRecommendation.objects.filter(user_profile=profile).exists():
        generate_career_recommendations(profile)

    recommendations = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')
    
    enhanced_recommendations = []
    high_demand_count = 0
    
    for rec in recommendations:
        enhanced_rec = enhance_recommendation_with_market_data(rec)
        enhanced_recommendations.append(enhanced_rec)
        
        market_data = getattr(enhanced_rec, 'market_data', {})
        demand_level = market_data.get('demand_level', '').lower() if market_data else ''
        
        if demand_level == 'high':
            high_demand_count += 1
    
    avg_match_score = recommendations.aggregate(Avg('match_score'))['match_score__avg'] or 0
    
    context = {
        'recommendations': enhanced_recommendations,
        'avg_match_score': round(avg_match_score, 1),
        'high_demand_roles': high_demand_count,
    }
    return render(request, 'career_recommendations.html', context)

@login_required
def career_detail(request, career_id):
    """Career detail view"""
    career = get_object_or_404(Career, pk=career_id)
    return render(request, 'career_detail.html', {'career': career})


# --- SMART SPLIT FUNCTION (Bracket safe) ---
def smart_split_skills(text):
    """
    Split skills by commas BUT DO NOT split commas inside brackets.
    Supports: (…) , {…} , […].
    """
    if not text:
        return []

    skills = []
    current = []
    depth_round = 0   # ()
    depth_square = 0  # []
    depth_curly = 0   # {}

    for char in text:
        if char == '(':
            depth_round += 1
        elif char == ')':
            depth_round -= 1
        elif char == '[':
            depth_square += 1
        elif char == ']':
            depth_square -= 1
        elif char == '{':
            depth_curly += 1
        elif char == '}':
            depth_curly -= 1

        # COMMA OUTSIDE bracket = SPLIT
        if char == ',' and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            skill = ''.join(current).strip()
            if skill:
                skills.append(skill)
            current = []
        else:
            current.append(char)

    # Add last item
    last = ''.join(current).strip()
    if last:
        skills.append(last)

    return skills

@login_required
def job_trends(request):
    """Job market trends view"""
    
    if len(CAREER_DF) == 0 or len(MARKET_DF) == 0:
        context = {
            'trends': [],
            'total_careers': 0,
            'avg_growth': 0,
            'avg_salary': 0,
            'unique_locations': 0,
            'error': 'Dataset not loaded. Please ensure data files exist in datasets/ folder.'
        }
        return render(request, 'trends.html', context)

    market_data = MARKET_DF.copy()
    career_data = CAREER_DF[['career_name', 'required_skills', 'average_salary', 'clean_key']].copy()

    trends_df = pd.merge(
        market_data, 
        career_data,
        on='clean_key', 
        how='left'
    )
    
    trends_df.drop_duplicates(subset=['clean_key'], keep='first', inplace=True)
    trends_df.dropna(subset=['job_title', 'average_salary'], inplace=True)
    
    median_salary = trends_df[trends_df['average_salary'] > 0]['average_salary'].median()
    trends_df['average_salary'].fillna(median_salary if not pd.isna(median_salary) else 0, inplace=True) 
    
    trends_df['required_skills'].fillna('Not Specified', inplace=True)
    trends_df['location'].fillna('Global/Remote', inplace=True)
    
    total_careers = len(trends_df)
    avg_growth = trends_df['job_growth_rate'].mean() * 100 if total_careers > 0 else 0
    valid_salaries = trends_df[trends_df['average_salary'] > 1000]['average_salary']
    avg_salary = valid_salaries.mean() if not valid_salaries.empty else 0
    
    all_locations = set()
    trends_df['location'].astype(str).apply(
        lambda x: all_locations.update(loc.strip() for loc in x.split(',') if loc.strip())
    )

    processed_trends = trends_df.to_dict('records')
    
    final_trends_list = []
    for trend in processed_trends:

        # === FIXED: BRACKET SAFE SKILL SPLIT ===
        raw_skills = str(trend.get('required_skills', ''))
        skills_list = smart_split_skills(raw_skills)

        # LOCATIONS normal split
        location_list = [l.strip() for l in str(trend.get('location', '')).split(',') if l.strip()]

        final_trends_list.append({
            'career_title': trend.get('job_title', 'Unknown'),
            'growth_rate': round(float(trend.get('job_growth_rate', 0)) * 100, 1),
            'average_salary': round(float(trend.get('average_salary', 0)), 0),

            # CLEAN LISTS
            'top_locations': location_list,
            'key_skills_in_demand': skills_list,

            'demand_level': 'High' if trend.get('job_growth_rate', 0) > 0.10 
                             else ('Medium' if trend.get('job_growth_rate', 0) > 0.05 else 'Low'),
            'avg_hiring_time_days': trend.get('avg_hiring_time_days', 'N/A'),
            'year': 2025
        })
    
    context = {
        'trends': sorted(final_trends_list, key=lambda x: x.get('growth_rate', 0), reverse=True)[:10],
        'total_careers': total_careers,
        'avg_growth': round(avg_growth, 1),
        'avg_salary': round(avg_salary / 1000, 0),
        'unique_locations': len(all_locations),
    }
    return render(request, 'trends.html', context)



@login_required
def skill_gap_analysis(request):
    """Skill gap analysis view - FIXED VERSION"""
    profile = request.user.userprofile
    
    # Get user skills from both SkillAssessment model and profile
    user_skills_from_assessment = SkillAssessment.objects.filter(user_profile=profile)
    user_skills_list = [skill.skill_name for skill in user_skills_from_assessment]
    
    # Also include skills from profile
    if profile.skills:
        profile_skills = [skill.strip() for skill in profile.skills.split(',') if skill.strip()]
        user_skills_list.extend(profile_skills)
    
    # Remove duplicates
    user_skills_list = list(set(user_skills_list))
    user_skills_text = ', '.join(user_skills_list)
    
    target_career_title = request.POST.get('target_career')
    skill_gaps = {}
    
    if target_career_title:
        skill_gaps = predict_skill_gaps(user_skills_text, target_career_title)
        
    return render(request, 'skill_gap_analysis.html', {
        'careers': CAREER_DF['career_name'].unique().tolist() if not CAREER_DF.empty else [],
        'user_skills': user_skills_from_assessment,
        'skill_gaps': skill_gaps,
        'target_career': target_career_title
    })

def chatbot(request):
    """Chatbot interface view"""
    return render(request, 'chatbot.html')

def about(request):
    """About page view"""
    return render(request, 'about.html')

def contact(request):
    """Contact page view"""
    return render(request, 'contact.html')

# --- 4. ML / HELPER FUNCTIONS ---

def simple_clean_text(text):
    """Common cleaning function for text inputs (used by ML models)"""
    if pd.isna(text) or not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|[0-9]+', ' ', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join(words).strip()

def calculate_profile_completion(profile):
    """Calculate profile completion percentage"""
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
    """Generate personalized insights for dashboard with correct skill counting by words."""
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

# --- RESUME ANALYSIS FUNCTIONS ---

def analyze_resume_file(resume_file):
    """Analyze uploaded resume file using trained models (SKILL_LIST)"""
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

        # Prepare text for skill matching (allows C++, C#, etc.)
        text_for_skills = text.lower()
        text_for_skills = re.sub(r'http\S+|www\S+|[0-9]+', ' ', text_for_skills, flags=re.MULTILINE)
        text_for_skills = re.sub(r'[^\w\s\+\.\#\-]', ' ', text_for_skills) # Keep C++, C#, .NET etc.

        skills_found = set()
        analysis_method = 'enhanced_rule_match'
        
        # Use the clean, extracted skill list from the joblib file
        if models_loaded and 'SKILL_LIST' in globals():
            for skill_name in SKILL_LIST:
                s = str(skill_name).strip().lower()
                # Check for exact word match using regex boundary \b
                pattern = r'\b' + re.escape(s) + r'\b'
                if re.search(pattern, text_for_skills):
                    skills_found.add(s)
        else:
            # Fallback hardcoded list (used if ML files are missing)
            analysis_method = 'keyword_fallback'
            for s in ['python','java','javascript','sql','html','css','react.js','node.js','django','flask','machine learning','data analysis']:
                if s in text_for_skills:
                    skills_found.add(s)

        # Clean and normalize skills (converts 'python' to 'Python', 'c++' to 'C++')
        skills_clean = clean_skill_list(list(skills_found))

        experience_years = 0
        matches = re.findall(r'(\d+)\s*(?:year|yr|years|yrs)\s*(?:of)?\s*(?:exp|experience)', text, re.IGNORECASE)
        if matches:
            experience_years = max(int(m) for m in matches)

        return {
            'skills': skills_clean,
            'experience_years': experience_years,
            'analysis_method': analysis_method
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': f'Analysis error: {str(e)}'}

# --- PERSONALITY ASSESSMENT FUNCTIONS ---

@fallback_if_models_fail
def predict_personality_type_ml(user_text):
    """Predict MBTI type using the trained Logistic Regression model"""
    cleaned_text = simple_clean_text(user_text)
    prediction_encoded = PERSONALITY_MODEL.predict([cleaned_text])[0]
    personality_type = PERSONALITY_LABEL_ENCODER.inverse_transform([prediction_encoded])[0]
    probabilities = PERSONALITY_MODEL.predict_proba([cleaned_text])[0]
    confidence = float(max(probabilities))

    return {
        'type': personality_type,
        'confidence': round(confidence, 2),
        'message': f"Based on your input, your type is {personality_type}."
    }

def calculate_personality_scores(post_data):
    """Calculate Big Five personality scores based on questionnaire responses"""
    scores = {
        'extraversion': 0,
        'agreeableness': 0,
        'conscientiousness': 0,
        'emotional_stability': 0,
        'openness': 0
    }
    
    positive_questions = {
        'extraversion': [1, 2], 'agreeableness': [9], 'conscientiousness': [5, 8], 
        'emotional_stability': [3], 'openness': [4, 10]
    }
    
    for trait, q_nums in positive_questions.items():
        for q_num in q_nums:
            score = int(post_data.get(f'question_{q_num}', 3))
            scores[trait] += score
    
    # Normalize scores to 1-10 scale
    for trait in scores.keys():
        scores[trait] = max(1, min(10, round(scores[trait] * 10 / 15)))

    return scores

def determine_mbti_type(scores):
    """Determine MBTI personality type based on Big Five scores"""
    e_i = 'E' if scores['extraversion'] >= 6 else 'I'
    s_n = 'N' if scores['openness'] >= 6 else 'S'
    t_f = 'T' if scores['conscientiousness'] >= 6 else 'F'
    j_p = 'J' if scores['conscientiousness'] >= 6 else 'P'
    
    return f"{e_i}{s_n}{t_f}{j_p}"

def get_key_strengths(personality_type, scores):
    """Get key strengths based on personality type and scores"""
    strengths_map = {
        'INTJ': ['Strategic Thinking', 'Analytical Mind', 'Independent Worker'],
        'ENTP': ['Innovative Thinking', 'Adaptability', 'Creative Problem Solving'],
        'INFP': ['Creativity', 'Empathy', 'Authenticity'],
        'ISTJ': ['Reliability', 'Attention to Detail', 'Practical Thinking']
    }
    return strengths_map.get(personality_type, ['Adaptability', 'Problem Solving', 'Learning Ability'])

def get_career_recommendations(personality_type, scores):
    """Get career recommendations based on personality type and scores"""
    career_map = {
        'INTJ': [{'title': 'Data Scientist'}, {'title': 'Software Architect'}],
        'ENTP': [{'title': 'Entrepreneur'}, {'title': 'Product Manager'}],
        'INFP': [{'title': 'Graphic Designer'}, {'title': 'UX Designer'}],
        'ISTJ': [{'title': 'Financial Analyst'}, {'title': 'Systems Analyst'}]
    }
    return career_map.get(personality_type, [{'title': 'Business Analyst'}, {'title': 'Project Coordinator'}])

# --- CAREER MATCHING FUNCTIONS ---

def find_career_matches(user_profile):
    """Core recommendation logic with 60% Skills + 40% Personality weighting"""
    
    user_skills_text = user_profile.get('skills', '')
    personality_type = user_profile.get('personality_type', '')
    
    if not models_loaded or not user_skills_text.strip() or len(CAREER_DF) == 0:
        return simple_match_fallback(user_profile)

    matches = []
    
    # Parse user skills
    user_skills_set = set([s.strip().lower() for s in user_skills_text.split(',') if s.strip()])
    
    for index, career in CAREER_DF.iterrows():
        # --- 1. SKILLS SCORE (60% weight) ---
        skills_score = calculate_detailed_skills_score(user_skills_set, career, user_profile)
        
        # --- 2. PERSONALITY SCORE (40% weight) ---
        personality_score = calculate_detailed_personality_score(personality_type, career)
        
        # --- 3. WEIGHTED FINAL SCORE ---
        weighted_score = (skills_score * 0.60) + (personality_score * 0.40)
        
        # --- 4. EXPERIENCE MULTIPLIER (up to 1.15x) ---
        experience_years = user_profile.get('experience_years', 0)
        experience_multiplier = min(1.15, 1.0 + (experience_years * 0.02))  # +2% per year, max 15%
        
        # --- 5. MARKET DEMAND BOOST (up to +8 points) ---
        market_boost = calculate_realistic_market_boost(career)
        
        # --- 6. CALCULATE FINAL SCORE ---
        final_score = (weighted_score * experience_multiplier) + market_boost
        final_score = max(45, min(98, final_score))  # Realistic range: 45-98%
        
        matches.append({
            'career_id': career.get('career_id', index),
            'title': career['career_name'], 
            'match_score': round(final_score, 1),
            'description': career.get('description', 'No description available')[:150] + '...',
            'skills_contribution': round(skills_score * 0.60, 1),
            'personality_contribution': round(personality_score * 0.40, 1),
        })
        
        if len(matches) >= 100: 
            break

    return sorted(matches, key=lambda x: x['match_score'], reverse=True)

def generate_career_recommendations(profile):
    """Generate and save career recommendations for the user - UPDATED"""
    # Convert profile to dict format for matching
    profile_data = {
        'skills': profile.skills or '',
        'experience_years': profile.experience_years or 0,
        'education_level': profile.education_level or '',
        'personality_type': profile.personality_type or ''
    }
    
    matches = find_career_matches(profile_data)

    # Clear existing recommendations
    CareerRecommendation.objects.filter(user_profile=profile).delete()
    
    # Save top 10 recommendations
    for match in matches[:10]:
        try:
            career_obj = Career.objects.get(title=match['title']) 
            CareerRecommendation.objects.create(
                user_profile=profile,
                recommended_career=career_obj,
                match_score=match['match_score'],
                reasoning=f"Match based on skills, experience, and market demand."
            )
        except Career.DoesNotExist:
            continue

def simple_match_fallback(user_profile):
    """Fallback logic if ML models are not loaded"""
    user_skills_set = set(user_profile.get('skills', '').lower().split(','))
    matches = []
    
    for _, career in CAREER_DF.iterrows():
        score = 0
        required_skills = set(str(career.get('required_skills', '')).lower().split(','))
        overlap = len(user_skills_set.intersection(required_skills))
        
        if required_skills:
            score = (overlap / len(required_skills)) * 80
            
        if user_profile.get('experience_years', 0) >= 3:
            score += 20
            
        matches.append({
            'title': career['career_name'],
            'match_score': round(score, 2),
            'description': career.get('description', '')[:100] + '...',
        })
        
    return sorted(matches, key=lambda x: x['match_score'], reverse=True)[:10]

def predict_skill_gaps(user_skills, target_career):
    """Predict skill gaps for a target career - FIXED VERSION"""
    if target_career not in CAREER_DF['career_name'].values: 
        return {
            'gap_score': 1.0, 
            'missing_skills': [],
            'required_skills': [],
            'current_skills': [],
            'required_skills_count': 0,
            'current_skills_count': 0,
            'missing_skills_count': 0,
            'coverage_percentage': 0
        }

    career_info = CAREER_DF[CAREER_DF['career_name'] == target_career].iloc[0]
    
    # Get required skills and clean them
    required_skills_raw = career_info.get('required_skills', '')
    if pd.isna(required_skills_raw):
        required_skills_raw = ''
    
    required_skills = set()
    if required_skills_raw:
        # Split by comma and clean each skill
        for skill in str(required_skills_raw).split(','):
            cleaned_skill = skill.strip()
            if cleaned_skill:
                required_skills.add(cleaned_skill.lower())

    # Get user skills and clean them
    user_skills_set = set()
    if user_skills:
        for skill in user_skills.split(','):
            cleaned_skill = skill.strip()
            if cleaned_skill:
                user_skills_set.add(cleaned_skill.lower())

    # Find matching skills (skills user has that are required)
    matching_skills = required_skills.intersection(user_skills_set)
    
    # Find missing skills (skills required but user doesn't have)
    missing_skills = required_skills - user_skills_set

    # Calculate metrics
    required_skills_count = len(required_skills)
    current_skills_count = len(matching_skills)
    missing_skills_count = len(missing_skills)
    
    gap_score = missing_skills_count / required_skills_count if required_skills_count > 0 else 0
    coverage_percentage = (current_skills_count / required_skills_count * 100) if required_skills_count > 0 else 0

    # Convert sets to lists of dictionaries for template
    required_skills_list = [{'name': skill.title()} for skill in required_skills]
    current_skills_list = [{'name': skill.title()} for skill in matching_skills]
    missing_skills_list = [{'name': skill.title()} for skill in missing_skills]

    return {
        'required_skills': required_skills_list,
        'current_skills': current_skills_list,
        'missing_skills': missing_skills_list,
        'gap_score': round(gap_score, 2),
        'required_skills_count': required_skills_count,
        'current_skills_count': current_skills_count,
        'missing_skills_count': missing_skills_count,
        'coverage_percentage': round(coverage_percentage, 1)
    }
# --- MARKET DATA ENHANCEMENT FUNCTIONS ---

def enhance_recommendation_with_market_data(recommendation):
    """Add market data to recommendation object - IMPROVED VERSION"""
    career_title = recommendation.recommended_career.title
    
    # Get market data
    market_data = get_market_data_for_career(career_title)
    
    # Add market data to recommendation object as dynamic attribute
    recommendation.market_data = market_data
    return recommendation

def get_market_data_for_career(career_title):
    """Extract market data for a specific career - IMPROVED VERSION"""
    clean_title = clean_title_for_merge(career_title)
    
    # Look in MARKET_DF first
    market_info = MARKET_DF[MARKET_DF['clean_key'] == clean_title]
    
    if not market_info.empty:
        market_row = market_info.iloc[0]
        growth_rate = market_row.get('job_growth_rate', 0)
        
        demand_level = calculate_demand_level(growth_rate)
        
        return {
            'salary': format_salary(market_row.get('average_salary', 0)),
            'growth_rate': round(float(growth_rate) * 100, 2) if growth_rate else 8.5,
            'demand_level': demand_level,
            'locations': market_row.get('location', 'Global'),
            'hiring_time': market_row.get('avg_hiring_time_days', '30-45 days')
        }
    
    # Fallback to CAREER_DF if market data not found
    career_info = CAREER_DF[CAREER_DF['clean_key'] == clean_title]
    if not career_info.empty:
        career_row = career_info.iloc[0]
        
        # Generate realistic demand level based on career type
        demand_level = generate_demand_level_by_title(career_title)
        
        return {
            'salary': format_salary(career_row.get('average_salary', 0)),
            'growth_rate': 12.5 if 'data' in career_title.lower() or 'ai' in career_title.lower() else 8.5,
            'demand_level': demand_level,
            'locations': 'Global',
            'hiring_time': '30-45 days'
        }
    
    # Final fallback - Smart defaults based on career title
    default_salary = get_default_salary_by_title(career_title)
    demand_level = generate_demand_level_by_title(career_title)
    
    return {
        'salary': default_salary,
        'growth_rate': get_default_growth_by_title(career_title),
        'demand_level': demand_level,
        'locations': 'Global',
        'hiring_time': '30-60 days'
    }

def generate_demand_level_by_title(title):
    """Generate realistic demand level based on career title"""
    title_lower = title.lower()
    
    # High demand careers
    high_demand_keywords = [
        'software', 'developer', 'engineer', 'data scientist', 'ai', 
        'machine learning', 'cybersecurity', 'cloud', 'devops'
    ]
    
    # Medium demand careers  
    medium_demand_keywords = [
        'analyst', 'consultant', 'manager', 'designer', 'marketing',
        'product', 'project', 'business'
    ]
    
    for keyword in high_demand_keywords:
        if keyword in title_lower:
            return 'High'
    
    for keyword in medium_demand_keywords:
        if keyword in title_lower:
            return 'Medium'
    
    return 'Low'

def get_default_growth_by_title(title):
    """Get default growth rate based on career title"""
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['data', 'ai', 'machine learning', 'software']):
        return 15.5
    elif any(word in title_lower for word in ['developer', 'engineer', 'cloud']):
        return 12.5
    elif any(word in title_lower for word in ['analyst', 'consultant']):
        return 9.5
    else:
        return 7.5

def calculate_demand_level(growth_rate):
    """Calculate demand level based on growth rate - IMPROVED"""
    if not growth_rate or pd.isna(growth_rate):
        return 'Medium'
    
    try:
        growth = float(growth_rate)
        if growth > 0.20: 
            return 'High'
        elif growth > 0.10:
            return 'Medium'
        else:
            return 'Low'
    except (ValueError, TypeError):
        return 'Medium'

def format_salary(salary_value):
    """Format salary to proper string format"""
    try:
        if pd.isna(salary_value) or salary_value == 0 or salary_value == '0':
            return get_default_salary()
        
        salary_num = float(salary_value)
        if salary_num >= 1000:
            return f"{salary_num:,.0f}"
        else:
            return f"{salary_num:.0f}"
    except (ValueError, TypeError):
        return get_default_salary()

def get_default_salary():
    """Get realistic default salary"""
    return "85,000"

def get_default_salary_by_title(title):
    """Get default salary based on career title"""
    title_lower = title.lower()
    
    salary_ranges = {
        'software': "120,000",
        'developer': "110,000", 
        'engineer': "115,000",
        'data': "105,000",
        'scientist': "120,000",
        'analyst': "75,000",
        'manager': "95,000",
        'designer': "85,000",
        'marketing': "70,000",
        'sales': "65,000",
        'financial': "80,000",
        'consultant': "90,000"
    }
    
    for key, salary in salary_ranges.items():
        if key in title_lower:
            return salary
    
    return "85,000"

# --- SKILLS ASSESSMENT FIXED VIEWS ---

@login_required
def skills_assessment(request):
    """Skills assessment view (Add and view) - FIXED VERSION"""
    profile = request.user.userprofile
    user_skills = SkillAssessment.objects.filter(user_profile=profile).order_by('-created_at')

    if request.method == 'POST':
        form = SkillAssessmentForm(request.POST)
        if form.is_valid():
            skill = form.save(commit=False)
            skill.user_profile = profile
            
            # Check for duplicate skills
            existing_skill = SkillAssessment.objects.filter(
                user_profile=profile, 
                skill_name__iexact=skill.skill_name
            ).first()
            
            if existing_skill:
                messages.warning(request, f'Skill "{skill.skill_name}" already exists in your profile.')
                return redirect('skills')
            
            skill.save()
            
            # Update user profile skills string
            update_user_profile_skills(profile)
            
            messages.success(request, f'Skill "{skill.skill_name}" added successfully!')
            return redirect('skills')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = SkillAssessmentForm()

    # Calculate skills statistics
    total_skills = user_skills.count()
    
    # Calculate skill levels distribution
    try:
        skill_levels = user_skills.values('skill_level').annotate(count=Count('skill_level'))
    except Exception:
        skill_levels = []
    
    # Calculate average experience
    try:
        avg_experience = user_skills.aggregate(avg=Avg('years_of_experience'))['avg'] or 0
    except Exception:
        avg_experience = 0
    
    # Calculate total experience
    try:
        total_experience = user_skills.aggregate(total=Sum('years_of_experience'))['total'] or 0
    except Exception:
        total_experience = 0
    
    # Find most common skill level
    try:
        top_category_data = user_skills.values('skill_level').annotate(count=Count('skill_level')).order_by('-count').first()
        top_category = top_category_data['skill_level'].title() if top_category_data else "Beginner"
    except Exception:
        top_category = "Beginner"
    
    # Get skill recommendations based on current skills
    recommended_skills = generate_skill_recommendations_based_on_profile(profile)

    context = {
        'form': form,
        'user_skills': user_skills,
        'total_skills': total_skills,
        'skill_levels': skill_levels,
        'avg_experience': round(avg_experience, 1),
        'total_experience': round(total_experience, 1),
        'top_category': top_category,
        'recommended_skills': recommended_skills,
    }
    return render(request, 'skills.html', context)

@login_required
def add_skill(request):
    """Add new skill view - FIXED"""
    if request.method == 'POST':
        form = SkillAssessmentForm(request.POST)
        if form.is_valid():
            skill = form.save(commit=False)
            skill.user_profile = request.user.userprofile
            
            # Check for duplicates
            if SkillAssessment.objects.filter(
                user_profile=skill.user_profile, 
                skill_name__iexact=skill.skill_name
            ).exists():
                messages.warning(request, f'Skill "{skill.skill_name}" already exists!')
                return redirect('skills')
            
            skill.save()
            
            # Update profile skills
            update_user_profile_skills(skill.user_profile)
            
            messages.success(request, f'Skill "{skill.skill_name}" added successfully!')
            return redirect('skills')
        else:
            # If form is invalid, redirect back to skills page with errors
            messages.error(request, 'Please correct the errors in the form.')
            return redirect('skills')
    return redirect('skills')

@login_required
def delete_skill(request, skill_id):
    """Delete skill view"""
    try:
        skill = get_object_or_404(SkillAssessment, id=skill_id, user_profile=request.user.userprofile)
        skill_name = skill.skill_name
        skill.delete()
        
        # Update profile skills
        update_user_profile_skills(request.user.userprofile)
        
        messages.success(request, f'Skill "{skill_name}" deleted successfully!')
    except Exception:
        messages.error(request, 'Error deleting skill.')
    
    return redirect('skills')

@login_required 
def edit_skill(request, skill_id):
    """Edit skill view"""
    skill = get_object_or_404(SkillAssessment, id=skill_id, user_profile=request.user.userprofile)
    
    if request.method == 'POST':
        form = SkillAssessmentForm(request.POST, instance=skill)
        if form.is_valid():
            form.save()
            
            # Update profile skills
            update_user_profile_skills(request.user.userprofile)
            
            messages.success(request, f'Skill "{skill.skill_name}" updated successfully!')
            return redirect('skills')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = SkillAssessmentForm(instance=skill)
    
    return render(request, 'edit_skill.html', {'form': form, 'skill': skill})

# --- SKILLS HELPER FUNCTIONS ---

def update_user_profile_skills(profile):
    """Update UserProfile skills field with current skills"""
    try:
        user_skills = SkillAssessment.objects.filter(user_profile=profile)
        skills_list = [skill.skill_name for skill in user_skills]
        profile.skills = ', '.join(skills_list)
        profile.save()
    except Exception as e:
        print(f"❌ Error updating profile skills: {e}")

def generate_skill_recommendations_based_on_profile(profile):
    """Generate intelligent skill recommendations based on user profile and current skills"""
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
    """Generate skill recommendations based on user's career interests and personality"""
    try:
        career_based_skills = []
        
        # Get user's top career matches
        top_careers = CareerRecommendation.objects.filter(
            user_profile=profile
        ).order_by('-match_score')[:3]
        
        for career_rec in top_careers:
            career = career_rec.recommended_career
            required_skills = career.required_skills.split(',') if career.required_skills else []
            career_based_skills.extend([skill.strip() for skill in required_skills if skill.strip()])
        
        # Add skills based on personality type
        personality_skills = get_skills_by_personality(profile.personality_type)
        career_based_skills.extend(personality_skills)
        
        return list(set(career_based_skills)) 
    except Exception as e:
        print(f"Error generating career-based skills: {e}")
        return []

def get_skills_by_personality(personality_type):
    """Get recommended skills based on MBTI personality type"""
    if not personality_type:
        return ['Communication', 'Problem Solving', 'Teamwork']
        
    personality_skill_map = {
        'INTJ': ['Strategic Planning', 'Data Analysis', 'Research', 'Systems Thinking', 'Problem Solving'],
        'INTP': ['Research', 'Theoretical Analysis', 'Innovation', 'Technical Writing', 'Programming'],
        'ENTJ': ['Leadership', 'Project Management', 'Strategic Planning', 'Decision Making', 'Public Speaking'],
        'ENTP': ['Innovation', 'Brainstorming', 'Sales', 'Marketing', 'Creative Problem Solving'],
        'INFJ': ['Counseling', 'Writing', 'Research', 'Teaching', 'Conflict Resolution'],
        'INFP': ['Creative Writing', 'Design', 'Counseling', 'Research', 'Languages'],
        'ENFJ': ['Teaching', 'Mentoring', 'Public Speaking', 'Team Building', 'Communication'],
        'ENFP': ['Creative Thinking', 'Networking', 'Marketing', 'Writing', 'Public Relations'],
        'ISTJ': ['Organization', 'Accounting', 'Data Management', 'Quality Control', 'Administration'],
        'ISFJ': ['Healthcare', 'Teaching', 'Social Work', 'Customer Service', 'Organization'],
        'ESTJ': ['Management', 'Administration', 'Operations', 'Logistics', 'Supervision'],
        'ESFJ': ['Healthcare', 'Teaching', 'Customer Service', 'Event Planning', 'Team Coordination'],
        'ISTP': ['Technical Skills', 'Troubleshooting', 'Mechanical Skills', 'Analysis', 'Hands-on Work'],
        'ISFP': ['Design', 'Art', 'Healthcare', 'Customer Service', 'Creative Arts'],
        'ESTP': ['Sales', 'Marketing', 'Entrepreneurship', 'Networking', 'Action Planning'],
        'ESFP': ['Entertainment', 'Hospitality', 'Sales', 'Customer Service', 'Event Planning'],
    }
    
    return personality_skill_map.get(personality_type, [
        'Communication', 'Problem Solving', 'Teamwork', 'Adaptability', 'Time Management'
    ])

# --- ENHANCED CAREER MATCHING FUNCTIONS ---

def enhanced_find_career_matches(user_profile):
    """Enhanced career matching with skills integration"""
    profile = user_profile
    
    # Get user skills
    user_skills = SkillAssessment.objects.filter(user_profile=profile)
    user_skills_text = ', '.join([skill.skill_name for skill in user_skills])
    
    # Calculate total experience
    total_experience = user_skills.aggregate(total=Sum('years_of_experience'))['total'] or 0
    
    # Prepare profile text for ML model
    profile_text = f"{profile.education_level} {user_skills_text} {profile.personality_type or ''}"
    
    if not models_loaded or not user_skills_text.strip():
        return enhanced_simple_match_fallback(profile, user_skills)
    
    try:
        user_vector = VECTORIZER.transform([profile_text])
        similarity_scores = cosine_similarity(user_vector, CAREER_VECTORS).flatten()
        
        ranked_indices = np.argsort(similarity_scores)[::-1]
        matches = []
        
        for index in ranked_indices[:50]: # Limit to top 50 for performance
            career = CAREER_DF.iloc[index]
            base_score = similarity_scores[index] * 100
            
            # SKILLS MATCHING BONUS
            skills_match_bonus = calculate_skills_match_bonus(user_skills, career)
            base_score += skills_match_bonus
            
            # EXPERIENCE BONUS
            experience_bonus = calculate_experience_bonus(total_experience)
            base_score *= experience_bonus
            
            # MARKET BOOST
            market_boost = calculate_market_boost(career)
            base_score += market_boost
            
            # PERSONALITY BONUS
            personality_bonus = calculate_personality_bonus(profile.personality_type, career)
            base_score *= personality_bonus
            
            final_score = max(0, min(100, base_score))
            
            matches.append({
                'career_id': career.get('career_id', index),
                'title': career['career_name'], 
                'match_score': round(final_score, 2),
                'description': career.get('description', 'No description available')[:150] + '...',
                'skills_match': skills_match_bonus,
            })
            
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)
        
    except Exception as e:
        print(f"Error in enhanced_find_career_matches: {e}")
        return enhanced_simple_match_fallback(profile, user_skills)

def calculate_skills_match_bonus(user_skills, career):
    """Calculate bonus based on skills match"""
    if not user_skills:
        return 0
    
    career_required_skills = set()
    if career.get('required_skills'):
        career_required_skills = set([s.strip().lower() for s in career['required_skills'].split(',')])
    
    user_skill_names = {skill.skill_name.lower() for skill in user_skills}
    
    # Calculate overlap
    overlap = len(user_skill_names.intersection(career_required_skills))
    
    if career_required_skills:
        match_percentage = (overlap / len(career_required_skills)) * 30 # Max 30 points for skills
        return match_percentage
    
    return 0

def calculate_experience_bonus(total_experience):
    """Calculate experience bonus multiplier"""
    if total_experience >= 10:
        return 1.25
    elif total_experience >= 5:
        return 1.15
    elif total_experience >= 3:
        return 1.10
    elif total_experience >= 1:
        return 1.05
    else:
        return 1.0

def calculate_personality_bonus(personality_type, career):
    """Calculate personality compatibility bonus"""
    if not personality_type:
        return 1.0
    
    # Simple personality-career compatibility (can be enhanced)
    tech_careers = ['software', 'developer', 'engineer', 'data', 'ai', 'machine learning']
    creative_careers = ['designer', 'writer', 'artist', 'creative', 'ux', 'ui']
    business_careers = ['manager', 'analyst', 'consultant', 'marketing', 'sales']
    
    career_title = career['career_name'].lower()
    
    # INTJ, INTP, ENTJ, ENTP - Good with tech
    if personality_type in ['INTJ', 'INTP', 'ENTJ', 'ENTP']:
        if any(keyword in career_title for keyword in tech_careers):
            return 1.10
    
    # INFJ, INFP, ENFJ, ENFP - Good with creative
    if personality_type in ['INFJ', 'INFP', 'ENFJ', 'ENFP']:
        if any(keyword in career_title for keyword in creative_careers):
            return 1.10
    
    # ISTJ, ISFJ, ESTJ, ESFJ - Good with business
    if personality_type in ['ISTJ', 'ISFJ', 'ESTJ', 'ESFJ']:
        if any(keyword in career_title for keyword in business_careers):
            return 1.10
    
    return 1.0

def calculate_market_boost(career):
    """Calculate market demand boost"""
    career_merge_key = career['clean_key']
    market_info = MARKET_DF[MARKET_DF['clean_key'] == career_merge_key]
    
    if not market_info.empty:
        growth_rate = market_info['job_growth_rate'].iloc[0]
        if not pd.isna(growth_rate):
            return float(growth_rate) * 50 # Max 50 points for market
    
    return 0

def enhanced_simple_match_fallback(profile, user_skills):
    """Enhanced fallback matching with skills"""
    user_skill_names = {skill.skill_name.lower() for skill in user_skills}
    matches = []
    
    for _, career in CAREER_DF.iterrows():
        score = 0
        
        # Skills matching (60 points)
        if career.get('required_skills'):
            required_skills = set([s.strip().lower() for s in career['required_skills'].split(',')])
            overlap = len(user_skill_names.intersection(required_skills))
            if required_skills:
                score += (overlap / len(required_skills)) * 60
        
        # Experience bonus (20 points)
        total_experience = user_skills.aggregate(total=Sum('years_of_experience'))['total'] or 0
        if total_experience >= 3:
            score += 20
        elif total_experience >= 1:
            score += 10
        
        # Education bonus (10 points)
        if profile.education_level and profile.education_level != 'Not Specified':
            score += 10
        
        # Personality bonus (10 points)
        if profile.personality_type and profile.personality_type != 'Not assessed':
            score += 10
            
        matches.append({
            'title': career['career_name'],
            'match_score': round(score, 2),
            'description': career.get('description', '')[:100] + '...',
        })
        
    return sorted(matches, key=lambda x: x['match_score'], reverse=True)[:10]