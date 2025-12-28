from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import admin_views

router = DefaultRouter()
router.register(r'users', admin_views.UserViewSet)
router.register(r'careers', admin_views.CareerViewSet)
router.register(r'ai-config', admin_views.AIConfigViewSet)

urlpatterns = [
    # --- HTML UI Routes ---
    path('login/', admin_views.AdminLoginView.as_view(), name='admin_login'),
    path('dashboard/', admin_views.DashboardView.as_view(), name='admin_dashboard'),
    path('users/', admin_views.UserManagementView.as_view(), name='admin_users'),
    path('users/<int:pk>/', admin_views.UserDetailView.as_view(), name='admin_user_detail'),
    path('careers/', admin_views.CareerManagementView.as_view(), name='admin_careers'),
    path('ai-suggestions/', admin_views.AISuggestionsView.as_view(), name='admin_ai_suggestions'),
    path('ai-settings/', admin_views.AIConfigView.as_view(), name='admin_ai_settings'),

    # --- Actions ---
    path('api/actions/<str:action_type>/', admin_views.CareerActionAPI.as_view(), name='api_career_actions'),

    # --- API Routes ---
    path('api/auth/login/', admin_views.AdminAuthAPI.as_view(), name='api_admin_login'),
    path('api/auth/logout/', admin_views.AdminLogoutAPI.as_view(), name='api_admin_logout'),
    path('api/stats/', admin_views.DashboardStatsAPI.as_view(), name='api_admin_stats'),
    path('api/', include(router.urls)),
]
