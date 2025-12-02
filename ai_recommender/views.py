from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Avg, Sum, Count
from django.utils import timezone
from django.core.files.storage import default_storage

import pandas as pd

from .models import UserProfile, Career, PersonalityAssessment, SkillAssessment, CareerRecommendation
from .forms import UserRegistrationForm, UserProfileForm, UserUpdateForm, PersonalityAssessmentForm, SkillAssessmentForm, ResumeUploadForm

# Import from new modules
from .utils import (
    clean_skill_list, smart_split_skills, format_salary
)
from .services import (
    analyze_resume_file, generate_career_recommendations, predict_skill_gaps,
    calculate_profile_completion, generate_personalized_insights,
    calculate_personality_scores, determine_mbti_type, get_key_strengths,
    get_career_recommendations, enhance_recommendation_with_market_data,
    update_user_profile_skills, generate_skill_recommendations_based_on_profile,
    group_skills_by_category,
    CAREER_DF, MARKET_DF, SKILLS_DF
)

# --- CORE DJANGO VIEWS ---

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
                    # FIXED: Keep manual experience if it's greater than extracted experience
                    extracted_exp = analysis_result.get('experience_years', 0)
                    manual_exp = form.cleaned_data.get('experience_years', 0)
                    profile.experience_years = max(manual_exp, extracted_exp)
                    
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
    
    skill_match_score = 0
    if top_recommendations:
        target_career_title = top_recommendations[0].recommended_career.title 
        user_skills_text = profile.skills
        skill_gaps_data = predict_skill_gaps(user_skills_text, target_career_title)
        # FIXED: Use coverage_percentage for "Match Score" (Higher is better)
        skill_match_score = skill_gaps_data.get('coverage_percentage', 0)
        
    context = {
        'profile_completion': calculate_profile_completion(profile),
        'top_recommendations': top_recommendations,
        'experience_years': profile.experience_years or 0,
        'skills_count': SkillAssessment.objects.filter(user_profile=profile).count(),
        'skill_match_score': round(skill_match_score, 1),
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
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile = profile_form.save(commit=False)
            
            # FIXED: Update resume_text to reflect manual changes so UI stays consistent
            current_skills = profile_form.cleaned_data.get('skills', '')
            current_exp = profile_form.cleaned_data.get('experience_years', 0)
            
            # Preserve existing resume info if available, just update the stats part
            base_text = "Manual Update"
            if profile.resume_filename:
                base_text = f"Resume: {profile.resume_filename}"
                
            profile.resume_text = f"{base_text} | Skills: {current_skills}; Exp: {current_exp} years"
            profile.save()
            
            # FIXED: Regenerate recommendations based on new profile data
            generate_career_recommendations(profile)
            
            messages.success(request, 'Profile updated successfully! Recommendations refreshed.')
            return redirect('dashboard')
        else:
            print("User Form Errors:", user_form.errors)
            print("Profile Form Errors:", profile_form.errors)
            messages.error(request, 'Please correct the errors below.')
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = UserProfileForm(instance=profile)
    
    context = {
        'user_form': user_form,
        'profile_form': profile_form
    }
    return render(request, 'edit_profile.html', context)

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
def personality_assessment(request):
    """Personality assessment view (Used by /personality/ URL path)"""
    if request.method == 'POST':
        form = PersonalityAssessmentForm(request.POST)
        if form.is_valid():
            assessment = form.save(commit=False)
            profile = request.user.userprofile
            assessment.user_profile = profile
            assessment.save()
            
            # --- FIXED: Calculate and Save Personality Type ---
            scores = calculate_personality_scores(request.POST)
            personality_type = determine_mbti_type(scores)
            
            profile.personality_type = personality_type
            profile.save()
            
            # --- FIXED: Refresh Recommendations based on new personality ---
            generate_career_recommendations(profile)
            
            messages.success(request, f"Personality type determined: {personality_type}. Recommendations updated!")
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

    # Compute Big Five scores using shared logic
    # Construct a dict similar to request.POST
    post_data = {
        f'question_{i}': getattr(latest_assessment, f'question_{i}')
        for i in range(1, 11)
    }
    scores = calculate_personality_scores(post_data)

    # Determine personality type from scores
    personality_type = profile.personality_type
    
    # Fallback: If profile doesn't have type but we have assessment, save it!
    if not personality_type:
        personality_type = determine_mbti_type(scores)
        profile.personality_type = personality_type
        profile.save()
        
        # Also refresh recommendations since we have new data
        generate_career_recommendations(profile)

    # Build enriched recommended careers using datasets
    recs_basic = get_career_recommendations(personality_type, scores)
    enriched_recs = []
    for r in recs_basic:
        title = r.get('title', 'Role')
        # We need clean_title_for_merge but it's not imported directly, wait it is in utils but not imported?
        # Ah, I imported it from .utils
        from .utils import clean_title_for_merge, format_salary, get_default_growth_by_title, get_default_salary_by_title
        
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

            # --- FIXED: Refresh Recommendations ---
            generate_career_recommendations(profile)

            messages.success(request, f"Personality assessment complete! Type: {personality_type}")
            return redirect('personality_test_success')

        messages.error(request, 'Please correct the errors in the assessment form.')
        return render(request, 'personality_test.html', {'form': form, 'profile': profile})
    else:
        return render(request, 'personality_test.html', {'form': PersonalityAssessmentForm(), 'profile': profile})

@login_required
def personality_test_success(request):
    """Success page after submitting personality test"""
    return render(request, 'personality_success.html')

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
    """Chatbot interface view - Dynamic Version"""
    profile = request.user.userprofile
    
    # 1. Get Career Recommendations
    if not CareerRecommendation.objects.filter(user_profile=profile).exists():
        generate_career_recommendations(profile)
    
    recommendations = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')[:3]
    top_careers = []
    for rec in recommendations:
        top_careers.append({
            'title': rec.recommended_career.title,
            'match': round(rec.match_score ),
            'salary': format_salary(rec.recommended_career.average_salary)
        })

    # 2. Get Missing Skills for Top Career
    missing_skills = []
    if recommendations:
        top_rec = recommendations[0]
        target_career_title = top_rec.recommended_career.title
        user_skills_text = profile.skills
        skill_gaps_data = predict_skill_gaps(user_skills_text, target_career_title)
        missing_skills = skill_gaps_data.get('missing_skills', [])[:3]

    # 3. Get Market Trends (Top 3 by growth)
    trending_careers = []
    if not MARKET_DF.empty:
        trends = MARKET_DF.sort_values(by='job_growth_rate', ascending=False).head(3)
        for _, row in trends.iterrows():
            trending_careers.append({
                'title': row.get('job_title', 'Unknown'),
                'growth': round(row.get('job_growth_rate', 0) * 100, 1)
            })

    # 4. Personality
    personality_type = profile.personality_type or "Unknown"

    chatbot_data = {
        'user_name': request.user.username,
        'top_careers': top_careers,
        'missing_skills': missing_skills,
        'trending_careers': trending_careers,
        'personality_type': personality_type
    }

    import json
    return render(request, 'chatbot.html', {'chatbot_data': json.dumps(chatbot_data)})

def about(request):
    """About page view - Dynamic Version"""
    total_careers = len(CAREER_DF) if not CAREER_DF.empty else 1000
    
    unique_industries = 0
    if not CAREER_DF.empty and 'category' in CAREER_DF.columns:
         unique_industries = CAREER_DF['category'].nunique()
    else:
        unique_industries = 50

    context = {
        'total_careers': total_careers,
        'unique_industries': unique_industries
    }
    return render(request, 'about.html', context)

def contact(request):
    """Contact page view"""
    return render(request, 'contact.html')

# --- SKILLS ASSESSMENT FIXED VIEWS ---

@login_required
def skills_assessment(request):
    """
    [HINGLISH]
    Use: Ab ye 'Learning Resources Hub' ban gaya h.
    Why: User ne request kiya ki skills add karne ki jagah resources dikhao.
    Effect: User ko roadmap, videos, aur courses milte h.
    """
    profile = request.user.userprofile
    
    # Fetch learning resources using the new service
    from .services import get_learning_resources
    context = get_learning_resources(profile)
    
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