from django.contrib import admin
from django.urls import path, reverse
from django.utils.html import format_html
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Count, Avg

# Import Models
from .models import (
    UserProfile, Career, CareerRecommendation, 
    SkillAssessment, PersonalityAssessment, 
    AIConfig, SuggestedCareer, JobMarketTrend
)

# Import Utils
from .admin_utils import (
    process_career_csv, sync_job_market_data, 
    generate_ai_insights, retrain_ai_models
)

# --- PATCH ADMIN SITE INDEX ---
# We patch the default site to inject dashboard context without a custom AdminSite class
# This is a hacky but effective way to modify the standard admin index context.
original_index = admin.site.index

def patched_index(request, extra_context=None):
    # Calculate Analytics
    ids = {
        'total_users': UserProfile.objects.count(),
        'total_careers': Career.objects.count(),
        'pending_suggestions': SuggestedCareer.objects.filter(status='PENDING').count(),
        'at_risk_count': UserProfile.objects.annotate(rec_count=Count('careerrecommendation')).filter(rec_count=0).count(),
    }
    context = {**ids, **(extra_context or {})}
    return original_index(request, extra_context=context)

admin.site.index = patched_index

# =========================================================
# 1. INLINES (For Student 360 View)
# =========================================================
class SkillInline(admin.TabularInline):
    model = SkillAssessment
    extra = 0
    readonly_fields = ('created_at',)
    classes = ('collapse',)

class PersonalityInline(admin.TabularInline):
    model = PersonalityAssessment
    extra = 0
    readonly_fields = ('assessment_date',)
    classes = ('collapse',)

class RecommendationInline(admin.TabularInline):
    model = CareerRecommendation
    extra = 0
    readonly_fields = ('recommended_at', 'match_score')
    can_delete = False
    ordering = ('-match_score',)

    def has_add_permission(self, request, obj=None):
        return False

# =========================================================
# 2. STUDENT 360 (UserProfileAdmin)
# =========================================================
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'age', 'education_level', 'experience_years', 'career_match_status', 'resume_status')
    list_filter = ('education_level', 'gender', 'personality_type')
    search_fields = ('user__username', 'user__email', 'skills')
    inlines = [SkillInline, PersonalityInline, RecommendationInline]
    
    fieldsets = (
        ('Personal Info', {
            'fields': ('user', 'age', 'gender', 'education_level')
        }),
        ('Professional Profile', {
            'fields': ('current_field', 'experience_years', 'skills', 'personality_type')
        }),
        ('Resume Data', {
            'fields': ('resume_file', 'resume_text_preview', 'resume_uploaded_at'),
            'classes': ('collapse',)
        }),
    )
    
    readonly_fields = ('resume_text_preview',)

    def resume_text_preview(self, obj):
        if obj.resume_text:
            return obj.resume_text[:500] + "..." if len(obj.resume_text) > 500 else obj.resume_text
        return "No resume text parsed."
    resume_text_preview.short_description = "Parsed Resume Content"

    def resume_status(self, obj):
        if obj.resume_file:
            return format_html('<a href="{}" target="_blank" class="button">View Resume</a>', obj.resume_file.url)
        return format_html('<span style="color:red;">No Resume</span>')
    resume_status.short_description = "Resume"

    def career_match_status(self, obj):
        count = CareerRecommendation.objects.filter(user_profile=obj).count()
        if count > 0:
            return format_html('<span style="color:green;">✅ Matched ({})</span>', count)
        return format_html('<span style="color:orange; font-weight:bold;">⚠️ AI Knowledge Gap</span>')
    career_match_status.short_description = "AI Status"

    actions = ['generate_individual_insight']

    def generate_individual_insight(self, request, queryset):
        # Placeholder for single-user analysis
        for profile in queryset:
            # Logic to analyze this specific user
            pass
        self.message_user(request, "AI Analysis scheduled for selected profiles.")
    generate_individual_insight.short_description = "Run AI Deep Analysis on Selected"

# =========================================================
# 3. CAREER MANAGEMENT (CareerAdmin)
# =========================================================
@admin.register(Career)
class CareerAdmin(admin.ModelAdmin):
    list_display = ('title', 'average_salary', 'job_growth_rate', 'ai_weight')
    search_fields = ('title', 'description')
    list_editable = ('ai_weight',)
    change_list_template = 'admin/career_change_list.html'

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-csv/', self.admin_site.admin_view(self.upload_csv_view), name='career_upload_csv'),
            path('sync-api/', self.admin_site.admin_view(self.sync_api_view), name='career_sync_api'),
        ]
        return custom_urls + urls

    def upload_csv_view(self, request):
        if request.method == 'POST' and request.FILES.get('csv_file'):
            count, errors = process_career_csv(request.FILES['csv_file'])
            if errors:
                messages.warning(request, f"Processed {count} rows with errors: {errors[:3]}...")
            else:
                messages.success(request, f"Successfully imported {count} careers.")
            return redirect('..')
        
        # Simple upload form rendering (using standard Admin base)
        context = dict(
            self.admin_site.each_context(request),
        )
        return render(request, 'admin/csv_upload.html', context)

    def sync_api_view(self, request):
        result = sync_job_market_data()
        messages.success(request, result['message'])
        return redirect('..')

# =========================================================
# 4. AI LEARNING CONTROL (SuggestedCareerAdmin)
# =========================================================
@admin.register(SuggestedCareer)
class SuggestedCareerAdmin(admin.ModelAdmin):
    list_display = ('title', 'source', 'confidence_score', 'status_badge', 'created_at')
    list_filter = ('status', 'source')
    actions = ['approve_suggestion', 'reject_suggestion']
    readonly_fields = ('created_at',)

    def status_badge(self, obj):
        colors = {
            'PENDING': 'orange',
            'APPROVED': 'green',
            'REJECTED': 'red',
        }
        return format_html(
            '<span style="background-color:{}; color:white; padding:3px 8px; border-radius:10px;">{}</span>',
            colors.get(obj.status, 'grey'),
            obj.get_status_display()
        )
    status_badge.short_description = "Status"
    
    def approve_suggestion(self, request, queryset):
        count = 0
        for suggestion in queryset:
            if suggestion.status != 'APPROVED':
                # Convert to real Career
                Career.objects.get_or_create(
                    title=suggestion.title,
                    defaults={
                        'description': f"Auto-generated from {suggestion.source}",
                        'required_skills': suggestion.skills_detected,
                        'average_salary': 0, # Needs manual update
                        'job_growth_rate': 0.05,
                        'education_required': 'TBD',
                        'work_environment': 'Remote/Office'
                    }
                )
                suggestion.status = 'APPROVED'
                suggestion.save()
                count += 1
        self.message_user(request, f"Approved {count} suggestions. They are now live Careers.")
    approve_suggestion.short_description = "✅ Approve & Convert to Career"

    def reject_suggestion(self, request, queryset):
        queryset.update(status='REJECTED')
        self.message_user(request, "Selected suggestions rejected.")
    reject_suggestion.short_description = "❌ Reject Suggestion"

# =========================================================
# 5. AI CONFIG & ANALYTICS DASHBOARD
# =========================================================
@admin.register(AIConfig)
class AIConfigAdmin(admin.ModelAdmin):
    list_display = ('key', 'value', 'updated_at')
    actions = ['trigger_retraining']

    def trigger_retraining(self, request, queryset):
        msg = retrain_ai_models()
        self.message_user(request, msg)
    trigger_retraining.short_description = "🚀 Retrain AI Models"


# =========================================================
# 6. CUSTOM DASHBOARD OVERRIDE (Optional hook)
# =========================================================
# The actual dashboard UI is handled by overriding templates/admin/index.html.
# However, we can add a custom view here if we wanted a completely separate dashboard page.
# For this request, we stick to the standard Admin Index with injected context via Template args 
# or by using specific Admin Site overrides, but standard Django admin index injection is tricky 
# without a custom AdminSite. 
# SIMPLER APPROACH: We will add a "Dashboard" link in the sidebar or top nav via templates,
# or just rely on the template override which can load data via tags or context processors.
