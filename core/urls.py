# myproject/urls.py

from django.contrib import admin
from django.urls import path, include
from IMED import urls
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application



urlpatterns = [
    path('admin/', admin.site.urls),
    path('IMED/', include('IMED.urls')),  # Include your app's HTTP URL patterns
]
