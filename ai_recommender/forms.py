from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile, Career, PersonalityAssessment, SkillAssessment, CareerRecommendation

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    # Profile fields
    age = forms.IntegerField(
        min_value=16,
        max_value=70,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter your age'})
    )
    gender = forms.ChoiceField(
        choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    education_level = forms.ChoiceField(
        choices=[('HS', 'High School'), ('UG', 'Undergraduate'), ('PG', 'Postgraduate'), ('PHD', 'PhD')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    current_field = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Computer Science, Marketing'})
    )
    interests = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Tell us about your career interests'})
    )
    skills = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'List your skills (comma-separated)'})
    )
    experience_years = forms.IntegerField(
        min_value=0,
        max_value=50,
        initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Years of experience'})
    )
    personality_type = forms.CharField(
        max_length=4,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., INTJ, ENFP (optional)'})
    )
    resume_file = forms.FileField(
        required=False,
        label='Resume (PDF/DOCX)',
        help_text='Upload your resume in PDF or DOCX format (optional)',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.pdf,.docx'})
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2",
                  "age", "gender", "education_level", "current_field", "interests",
                  "skills", "experience_years", "personality_type", "resume_file")

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=100, required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'gender', 'education_level', 'current_field', 'interests', 'skills', 'experience_years', 'personality_type', 'resume_file']
        widgets = {
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': 16, 'max': 70}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'education_level': forms.Select(attrs={'class': 'form-control'}),
            'current_field': forms.TextInput(attrs={'class': 'form-control'}),
            'interests': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'skills': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'experience_years': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 50}),
            'personality_type': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., INTJ, ENFP'}),
            'resume_file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.pdf,.docx'}),
        }

class PersonalityAssessmentForm(forms.ModelForm):
    class Meta:
        model = PersonalityAssessment
        fields = ['question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                 'question_6', 'question_7', 'question_8', 'question_9', 'question_10']
        widgets = {
            'question_1': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_2': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_3': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_4': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_5': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_6': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_7': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_8': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_9': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
            'question_10': forms.Select(attrs={'class': 'form-control'}, choices=[(i, i) for i in range(1, 6)]),
        }

class SkillAssessmentForm(forms.ModelForm):
    class Meta:
        model = SkillAssessment
        fields = ['skill_name', 'skill_level', 'years_of_experience']
        widgets = {
            'skill_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Python, JavaScript, Machine Learning'
            }),
            'skill_level': forms.Select(attrs={'class': 'form-control'}),
            'years_of_experience': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Years of experience',
                'min': '0',
                'step': '0.5'
            })
        }
    
    def clean_skill_name(self):
        skill_name = self.cleaned_data.get('skill_name')
        if not skill_name:
            raise forms.ValidationError("Skill name is required.")
        return skill_name.strip().title()
    
class CareerForm(forms.ModelForm):
    class Meta:
        model = Career
        fields = ['title', 'description', 'required_skills', 'education_required',
                 'average_salary', 'job_growth_rate', 'work_environment', 'related_fields']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'required_skills': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'education_required': forms.TextInput(attrs={'class': 'form-control'}),
            'average_salary': forms.NumberInput(attrs={'class': 'form-control'}),
            'job_growth_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'work_environment': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'related_fields': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
        }

class ResumeUploadForm(forms.Form):
    resume_file = forms.FileField(
        required=False,
        label='Upload Resume',
        help_text='Upload your resume in PDF or DOCX format',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.pdf,.docx'})
    )

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    subject = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'class': 'form-control'}))
    message = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5}))