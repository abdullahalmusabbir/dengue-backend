from django.contrib import admin
from .models import DenguePredictionRecord


@admin.register(DenguePredictionRecord)
class DenguePredictionRecordAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'gender', 'age', 'prediction',
        'probability_percent', 'risk_group', 'dchss_score', 'created_at',
    )
    list_filter = ('prediction', 'risk_group', 'gender')
    search_fields = ('id',)
    readonly_fields = [f.name for f in DenguePredictionRecord._meta.fields]
    ordering = ('-created_at',)

    def has_add_permission(self, request):
        # Prediction শুধু API থেকেই তৈরি হবে, admin panel থেকে manually না
        return False