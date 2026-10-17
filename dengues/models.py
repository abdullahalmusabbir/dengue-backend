from django.db import models


class DenguePredictionRecord(models.Model):
    """
    প্রতিটা prediction request এর history রাখার জন্য model.
    Login লাগবে না - যে কেউ predict করলে এখানে save হবে (admin panel এ দেখা যাবে)।
    """

    GENDER_CHOICES = (
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    )
    PREDICTION_CHOICES = (
        ('Positive', 'Positive'),
        ('Negative', 'Negative'),
    )
    RISK_CHOICES = (
        ('Low', 'Low'),
        ('Moderate', 'Moderate'),
        ('High', 'High'),
    )

    # ---- Input fields (CBC) ----
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.FloatField()
    hemoglobin = models.FloatField()
    neutrophils = models.FloatField()
    lymphocytes = models.FloatField()
    monocytes = models.FloatField()
    eosinophils = models.FloatField()
    rbc = models.FloatField()
    hct = models.FloatField()
    mcv = models.FloatField()
    mch = models.FloatField()
    mchc = models.FloatField()
    rdw_cv = models.FloatField()
    platelet = models.FloatField()
    mpv = models.FloatField()
    pdw = models.FloatField()
    pct = models.FloatField()
    wbc = models.FloatField()

    # ---- Output fields ----
    probability_percent = models.FloatField()
    prediction = models.CharField(max_length=10, choices=PREDICTION_CHOICES)
    risk_group = models.CharField(max_length=10, choices=RISK_CHOICES)
    dchss_score = models.FloatField()
    threshold_used_percent = models.FloatField()
    top_feature_explanations = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Dengue Prediction Record"
        verbose_name_plural = "Dengue Prediction Records"

    def __str__(self):
        return f"#{self.id} | {self.prediction} ({self.probability_percent}%) | {self.created_at:%Y-%m-%d %H:%M}"