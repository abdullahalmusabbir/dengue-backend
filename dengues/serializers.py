from rest_framework import serializers
from .models import DenguePredictionRecord


class DengueInputSerializer(serializers.Serializer):
    """
    Frontend থেকে আসা raw CBC input validate করার জন্য।
    Login/user field নাই - শুধু medical values।
    """
    # serializers.py line 10
    gender = serializers.ChoiceField(choices=['male', 'female', 'other'])
    age = serializers.FloatField(min_value=0, max_value=120)

    hemoglobin = serializers.FloatField(min_value=0)
    neutrophils = serializers.FloatField(min_value=0, max_value=100)
    lymphocytes = serializers.FloatField(min_value=0, max_value=100)
    monocytes = serializers.FloatField(min_value=0, max_value=100)
    eosinophils = serializers.FloatField(min_value=0, max_value=100)
    rbc = serializers.FloatField(min_value=0)
    hct = serializers.FloatField(min_value=0, max_value=100)
    mcv = serializers.FloatField(min_value=0)
    mch = serializers.FloatField(min_value=0)
    mchc = serializers.FloatField(min_value=0)
    rdw_cv = serializers.FloatField(min_value=0)
    platelet = serializers.FloatField(min_value=0)
    mpv = serializers.FloatField(min_value=0)
    pdw = serializers.FloatField(min_value=0)
    pct = serializers.FloatField(min_value=0)
    wbc = serializers.FloatField(min_value=0)


class DenguePredictionRecordSerializer(serializers.ModelSerializer):
    """History listing এর জন্য (GET /history/)"""

    class Meta:
        model = DenguePredictionRecord
        fields = '__all__'
        read_only_fields = ['id', 'created_at']