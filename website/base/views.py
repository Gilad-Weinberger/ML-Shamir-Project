from django.shortcuts import render

from .forms import ImageUploadForm


def Home(request):
    """Serve the upload page shell; predict/upload handled via API + client JS."""
    return render(request, "base/home.html", {"form": ImageUploadForm()})
