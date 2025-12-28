from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, Career, AIConfig, SkillAssessment

class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'age', 'gender', 'education_level', 
                  'current_field', 'experience_years', 'personality_type', 'created_at']

class CareerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Career
        fields = '__all__'

class AIConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIConfig
        fields = '__all__'

class DashboardStatsSerializer(serializers.Serializer):
    total_users = serializers.IntegerField()
    total_careers = serializers.IntegerField()
    total_recommendations = serializers.IntegerField()
