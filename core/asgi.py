import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from IMED import consumers # Replace 'myapp' with your app name

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Define WebSocket URL patterns
# myapp/urls.py

from django.urls import re_path


websocket_urlpatterns = [
    re_path(r'ws/detector/$',consumers.Detector_eyes.as_asgi()),
    re_path(r'ws/simple/$',consumers.SimpleEchoConsumer.as_asgi()),
    re_path(r'ws/distance/$',consumers.Distance.as_asgi()),
]


# Create the ASGI application
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(websocket_urlpatterns),
})
