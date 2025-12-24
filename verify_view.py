
import os
import django
import sys
import traceback

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Career_Recommender.settings')
django.setup()

from django.test import RequestFactory
from ai_recommender.views import job_trends
from django.contrib.auth.models import User

def verify():
    print("--- Verifying Job Trends View ---")
    try:
        factory = RequestFactory()
        request = factory.get('/trends/')
        
        # Mock User
        user = User.objects.first()
        if not user:
            print("⚠️ No user found. Creating temp user.")
            user = User.objects.create_user('testuser', 'test@example.com', 'password')
            
        request.user = user
        
        print("Calling job_trends...")
        response = job_trends(request)
        print(f"View Response Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: job_trends view ran successfully.")
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            
    except Exception as e:
        print("❌ EXCEPTION OCCURRED:")
        traceback.print_exc()

if __name__ == "__main__":
    verify()
