from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('result', views.result, name="result"),
    path('About', views.about, name="about"),
    path('team', views.ourteams, name="ourteams"),
    path('form', views.ourservices, name="ourservices"),
    path('diseases', views.diseases, name="diseases"),
    path('upload', views.uploadImage, name="uploadImage"),
    path('d_result', views.d_result, name="diseases_result"),

    # path('signup', views.handleSignup, name="handleSignup"),
    # path('login', views.handleLogin, name="handleLogin"),
    # path('logout', views.handleLogout, name="handleLogout"),




    path("register", views.register_request, name="register"),
    path("login", views.login_request, name="login"),
    path("logout", views.logout_request, name="logout"),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
