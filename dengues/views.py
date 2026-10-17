from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .serializers import DengueInputSerializer, DenguePredictionRecordSerializer
from .models import DenguePredictionRecord
from .services.deocker_services import predict_dengue, ModelNotLoadedError


class DenguePredictionView(APIView):
    """
    POST /api/dengues/predict/
    ⭐ Login/Authentication লাগবে না - সম্পূর্ণ public endpoint।
    Input: CBC values (JSON body) -> Output: probability %, risk group, explanation
    """
    permission_classes = [AllowAny]
    authentication_classes = []  # কোনো authentication class দরকার নাই

    def post(self, request):
        serializer = DengueInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        raw_input = serializer.validated_data

        try:
            result = predict_dengue(raw_input)
        except ModelNotLoadedError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            return Response(
                {'error': f'Prediction failed: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # History সংরক্ষণ (optional কিন্তু admin panel এ দেখার জন্য রাখা হলো)
        DenguePredictionRecord.objects.create(
            gender=raw_input['gender'],
            age=raw_input['age'],
            hemoglobin=raw_input['hemoglobin'],
            neutrophils=raw_input['neutrophils'],
            lymphocytes=raw_input['lymphocytes'],
            monocytes=raw_input['monocytes'],
            eosinophils=raw_input['eosinophils'],
            rbc=raw_input['rbc'],
            hct=raw_input['hct'],
            mcv=raw_input['mcv'],
            mch=raw_input['mch'],
            mchc=raw_input['mchc'],
            rdw_cv=raw_input['rdw_cv'],
            platelet=raw_input['platelet'],
            mpv=raw_input['mpv'],
            pdw=raw_input['pdw'],
            pct=raw_input['pct'],
            wbc=raw_input['wbc'],
            probability_percent=result['probability_percent'],
            prediction=result['prediction'],
            risk_group=result['risk_group'],
            dchss_score=result['dchss_score'],
            threshold_used_percent=result['threshold_used_percent'],
            top_feature_explanations=result['top_feature_explanations'],
        )

        return Response(result, status=status.HTTP_200_OK)


class DenguePredictionHistoryView(APIView):
    """
    GET /api/dengues/history/
    সাম্প্রতিক ৫০টা prediction history দেখানোর জন্য (public, login ছাড়াই)।
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        records = DenguePredictionRecord.objects.all()[:50]
        serializer = DenguePredictionRecordSerializer(records, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)