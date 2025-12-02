from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    EDUCATION_CHOICES = [
        ('HS', 'High School'),
        ('UG', 'Undergraduate'),
        ('PG', 'Postgraduate'),
        ('PHD', 'PhD'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField(default=25)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='O')
    education_level = models.CharField(max_length=3, choices=EDUCATION_CHOICES, default='UG')
    current_field = models.CharField(max_length=100, blank=True)
    interests = models.TextField(blank=True)
    skills = models.TextField(blank=True)
    experience_years = models.IntegerField(default=0)
    personality_type = models.CharField(max_length=4, blank=True)  # MBTI type
    resume_file = models.FileField(
        upload_to='resumes/',
        blank=True,
        null=True,
        help_text='Upload your resume in PDF or DOCX format'
    )
    resume_text = models.TextField(blank=True, help_text='Extracted text content from resume file')
    resume_filename = models.CharField(max_length=255, blank=True)
    resume_uploaded_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

class Career(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    required_skills = models.TextField()
    education_required = models.CharField(max_length=100)
    average_salary = models.DecimalField(max_digits=10, decimal_places=2)
    job_growth_rate = models.DecimalField(max_digits=5, decimal_places=2)
    work_environment = models.TextField()
    related_fields = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class PersonalityAssessment(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    question_1 = models.IntegerField()  # Scale 1-5
    question_2 = models.IntegerField()
    question_3 = models.IntegerField()
    question_4 = models.IntegerField()
    question_5 = models.IntegerField()
    question_6 = models.IntegerField()
    question_7 = models.IntegerField()
    question_8 = models.IntegerField()
    question_9 = models.IntegerField()
    question_10 = models.IntegerField()
    assessment_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Assessment for {self.user_profile.user.username}"

# ai_recommender/models.py
class SkillAssessment(models.Model):
    SKILL_LEVELS = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'), 
        ('advanced', 'Advanced'),
        ('expert', 'Expert'),
    ]
    
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    skill_name = models.CharField(max_length=100)
    skill_level = models.CharField(max_length=20, choices=SKILL_LEVELS, default='beginner')
    years_of_experience = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.skill_name} ({self.skill_level}) - {self.user_profile.user.username}"

class CareerRecommendation(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    recommended_career = models.ForeignKey(Career, on_delete=models.CASCADE)
    match_score = models.DecimalField(max_digits=5, decimal_places=2)
    reasoning = models.TextField()
    recommended_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.recommended_career.title} for {self.user_profile.user.username}"

class JobMarketTrend(models.Model):
    career = models.ForeignKey(Career, on_delete=models.CASCADE)
    trend_year = models.IntegerField()
    demand_level = models.CharField(max_length=50)  # High, Medium, Low
    average_salary_trend = models.DecimalField(max_digits=10, decimal_places=2)
    top_locations = models.TextField()
    key_skills_in_demand = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.career.title} - {self.trend_year}"

class CareerRoadmap(models.Model):
    career = models.ForeignKey(Career, on_delete=models.CASCADE, related_name='roadmaps')
    step_number = models.IntegerField()
    title = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['step_number']
        unique_together = ['career', 'step_number']

    def __str__(self):
        return f"{self.career.title} - Step {self.step_number}: {self.title}"

class LearningResource(models.Model):
    RESOURCE_TYPES = [
        ('video', 'Video'),
        ('course', 'Course'),
        ('article', 'Article'),
        ('book', 'Book'),
    ]
    
    title = models.CharField(max_length=255)
    url = models.URLField()
    resource_type = models.CharField(max_length=20, choices=RESOURCE_TYPES)
    platform = models.CharField(max_length=100) # e.g., YouTube, Udemy, Coursera
    skill_tag = models.CharField(max_length=100) # e.g., Python, Machine Learning
    thumbnail_url = models.URLField(blank=True, null=True)
    is_free = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.platform})"
