from rest_framework import viewsets, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from django.contrib.auth import authenticate, login, logout
from django.views.generic import TemplateView
from django.shortcuts import redirect
from django.contrib.auth.mixins import UserPassesTestMixin
from django.db.models import Count

from .models import UserProfile, Career, AIConfig, CareerRecommendation, SuggestedCareer
from .serializers import UserProfileSerializer, CareerSerializer, AIConfigSerializer

# --- PERMISSIONS ---
class IsAdminUser(permissions.BasePermission):
    """
    Allows access only to admin users.
    """
    def has_permission(self, request, view):
        return request.user and request.user.is_superuser

class AdminRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser
    
    def handle_no_permission(self):
        return redirect('admin_login')

# --- HTML TEMPLATE VIEWS ---

class AdminLoginView(TemplateView):
    template_name = 'custom_admin/login.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated and request.user.is_superuser:
            return redirect('admin_dashboard')
        return super().get(request, *args, **kwargs)

class DashboardView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Dashboard'
        context['total_users'] = UserProfile.objects.count()
        context['total_careers'] = Career.objects.count()
        context['total_recs'] = CareerRecommendation.objects.count()
        context['popular_careers'] = CareerRecommendation.objects.values('recommended_career__title')\
                                .annotate(count=Count('recommended_career'))\
                                .order_by('-count')[:5]
        
        # New AI Insights
        context['pending_insights'] = SuggestedCareer.objects.filter(status='PENDING').count()
        context['at_risk_count'] = UserProfile.objects.annotate(rec_count=Count('careerrecommendation')).filter(rec_count=0).count()
        return context

class UserManagementView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/users.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'User Management'
        
        # Search Logic
        query = self.request.GET.get('q')
        queryset = UserProfile.objects.select_related('user').all().order_by('-created_at')
        
        if query:
            from django.db.models import Q
            queryset = queryset.filter(
                Q(user__username__icontains=query) | 
                Q(user__email__icontains=query) |
                Q(skills__icontains=query)
            )
            context['search_query'] = query
            
        context['users'] = queryset
        return context

class CareerManagementView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/careers.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Career Management'
        context['careers'] = Career.objects.all().order_by('title')
        return context

class AIConfigView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/ai_config.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'AI Configuration'
        context['configs'] = AIConfig.objects.all().order_by('key')
        return context


# --- API VIEWS (DRF) ---

class UserDetailView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/user_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user_id = self.kwargs.get('pk')
        profile = UserProfile.objects.select_related('user').get(id=user_id)
        
        context['page_title'] = f"Student 360: {profile.user.username}"
        context['profile'] = profile
        context['skills_list'] = profile.get_skills_list() if hasattr(profile, 'get_skills_list') else profile.skills.split(',')
        context['recommendations'] = CareerRecommendation.objects.filter(user_profile=profile).order_by('-match_score')
        context['assessments'] = {
            'skills': profile.skillassessment_set.all(),
            'personality': profile.personalityassessment_set.all()
        }
        return context

class AISuggestionsView(AdminRequiredMixin, TemplateView):
    template_name = 'custom_admin/ai_suggestions.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'AI Career Suggestions'
        context['suggestions'] = SuggestedCareer.objects.filter(status='PENDING').order_by('-confidence_score')
        return context
    
    def post(self, request, *args, **kwargs):
        action = request.POST.get('action')
        suggestion_id = request.POST.get('suggestion_id')
        
        try:
            suggestion = SuggestedCareer.objects.get(id=suggestion_id)
            if action == 'approve':
                # Convert to Career
                Career.objects.get_or_create(
                    title=suggestion.title,
                    defaults={
                        'description': f"AI Generated from {suggestion.source}",
                        'required_skills': suggestion.skills_detected,
                        'average_salary': 0,
                        'job_growth_rate': 0.05
                    }
                )
                suggestion.status = 'APPROVED'
            elif action == 'reject':
                suggestion.status = 'REJECTED'
            
            suggestion.save()
            return redirect('admin_ai_suggestions')
        except Exception as e:
            # simple error handling
            return redirect('admin_ai_suggestions')

class CareerActionAPI(APIView):
    """
    API for CSV Upload and Sync
    """
    permission_classes = [permissions.IsAdminUser]

    def post(self, request, action_type):
        from .admin_utils import process_career_csv, sync_job_market_data
        
        if action_type == 'upload_csv':
            try:
                file = request.FILES.get('csv_file')
                if not file: return Response({'error': 'No file provided'}, status=400)
                count, errors = process_career_csv(file)
                return Response({'message': f'Uploaded {count} careers.', 'errors': errors})
            except Exception as e:
                return Response({'error': str(e)}, status=500)
        
        elif action_type == 'sync_api':
            res = sync_job_market_data()
            return Response(res)
            
        return Response({'error': 'Invalid action'}, status=400)

class AdminAuthAPI(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        
        if user is not None:
            if user.is_superuser:
                login(request, user)
                return Response({'message': 'Login successful', 'redirect': '/admin/dashboard/'})
            else:
                return Response({'error': 'Unauthorized access'}, status=status.HTTP_403_FORBIDDEN)
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

class AdminLogoutAPI(APIView):
    def post(self, request):
        logout(request)
        return Response({'message': 'Logged out'})

class DashboardStatsAPI(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        stats = {
            'total_users': UserProfile.objects.count(),
            'total_careers': Career.objects.count(),
            'total_recommendations': CareerRecommendation.objects.count(),
            'popular_careers': CareerRecommendation.objects.values('recommended_career__title')
                                .annotate(count=Count('recommended_career'))
                                .order_by('-count')[:5]
        }
        return Response(stats)

class UserViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAdminUser]

class CareerViewSet(viewsets.ModelViewSet):
    queryset = Career.objects.all()
    serializer_class = CareerSerializer
    permission_classes = [permissions.IsAdminUser]

class AIConfigViewSet(viewsets.ModelViewSet):
    queryset = AIConfig.objects.all()
    serializer_class = AIConfigSerializer
    permission_classes = [permissions.IsAdminUser]
