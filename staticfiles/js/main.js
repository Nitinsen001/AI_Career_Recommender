// AI Career Recommender JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // File upload validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Validate file size (5MB max)
                if (file.size > 5 * 1024 * 1024) {
                    showAlert('File size must be less than 5MB', 'danger');
                    this.value = '';
                    return;
                }

                // Validate file type for resumes
                if (this.accept && this.accept.includes('.pdf,.docx')) {
                    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
                    if (!allowedTypes.includes(file.type)) {
                        showAlert('Please upload only PDF or DOCX files', 'danger');
                        this.value = '';
                        return;
                    }
                }
            }
        });
    });
});

// Utility Functions
function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-temporary');
    existingAlerts.forEach(alert => alert.remove());

    // Add new alert
    const container = document.querySelector('.container') || document.body;
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert-temporary';
    alertDiv.innerHTML = alertHtml;

    if (document.querySelector('.container')) {
        document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.container').firstChild);
    } else {
        document.body.insertBefore(alertDiv, document.body.firstChild);
    }
}

function showLoading(button) {
    const originalText = button.textContent;
    button.disabled = true;
    button.innerHTML = '<span class="loading-spinner me-2"></span>Processing...';

    return function() {
        button.disabled = false;
        button.textContent = originalText;
    };
}

function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });

    return isValid;
}

// Chart.js integration for skills visualization
function createSkillsChart(ctx, labels, data) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#667eea',
                    '#764ba2',
                    '#28a745',
                    '#17a2b8',
                    '#ffc107',
                    '#dc3545'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Personality test functionality
function calculatePersonalityType(answers) {
    // Simple MBTI-like calculation
    let e_i = 0; // Extraversion vs Introversion
    let s_n = 0; // Sensing vs Intuition
    let t_f = 0; // Thinking vs Feeling
    let j_p = 0; // Judging vs Perceiving

    // This is a simplified calculation - in reality, you'd use a proper MBTI assessment
    answers.forEach((answer, index) => {
        switch(index % 4) {
            case 0: e_i += answer; break;
            case 1: s_n += answer; break;
            case 2: t_f += answer; break;
            case 3: j_p += answer; break;
        }
    });

    const type = [
        e_i > 2.5 ? 'E' : 'I',
        s_n > 2.5 ? 'S' : 'N',
        t_f > 2.5 ? 'T' : 'F',
        j_p > 2.5 ? 'J' : 'P'
    ].join('');

    return type;
}

// Career recommendation algorithm (simplified)
function calculateCareerMatch(userProfile, careerList) {
    return careerList.map(career => {
        let score = 0;

        // Education match (30%)
        if (userProfile.education_level === career.education_required) {
            score += 30;
        } else if (userProfile.education_level === 'PG' && career.education_required === 'UG') {
            score += 20;
        }

        // Skills match (40%)
        const userSkills = userProfile.skills ? userProfile.skills.toLowerCase() : '';
        const requiredSkills = career.required_skills.toLowerCase();
        const skillMatches = requiredSkills.split(',').filter(skill =>
            userSkills.includes(skill.trim())
        ).length;
        const totalSkills = requiredSkills.split(',').length;
        score += (skillMatches / totalSkills) * 40;

        // Interest match (30%)
        const userInterests = userProfile.interests ? userProfile.interests.toLowerCase() : '';
        const careerFields = career.related_fields ? career.related_fields.toLowerCase() : '';
        const interestMatches = careerFields.split(',').filter(field =>
            userInterests.includes(field.trim())
        ).length;
        const totalFields = careerFields.split(',').length || 1;
        score += (interestMatches / totalFields) * 30;

        return {
            career: career,
            score: Math.round(score)
        };
    }).sort((a, b) => b.score - a.score);
}

// Export functions for use in other scripts (no AJAX)
window.CareerRecommender = {
    showAlert,
    showLoading,
    validateForm,
    createSkillsChart,
    calculatePersonalityType,
    calculateCareerMatch
};