from django.apps import AppConfig


class DenguesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dengues'

    def ready(self):
        from . import admin