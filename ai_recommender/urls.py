from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    path('resume/', views.resume_upload, name='resume'),

    
    
    path('skills/', views.skills_assessment, name='skills'),
    path('skills/add/', views.add_skill, name='add_skill'),
    path('skills/delete/<int:skill_id>/', views.delete_skill, name='delete_skill'),
    path('skills/edit/<int:skill_id>/', views.edit_skill, name='edit_skill'),
    path('skill-gap/', views.skill_gap_analysis, name='skill_gap_analysis'),
    
    path('personality/', views.personality_assessment, name='personality'),
    path('personality/assess/', views.take_personality_test, name='take_personality_test'),
    path('personality/result/', views.personality_result, name='personality_result'),
    path('personality/success/', views.personality_test_success, name='personality_test_success'),
    
    path('career-recommendations/', views.career_recommendations, name='career_recommendations'),
    path('career/<int:career_id>/', views.career_detail, name='career_detail'),
    
    
    path('trends/', views.job_trends, name='trends'),
    path('chatbot/', views.chatbot, name='chatbot'),
    
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]