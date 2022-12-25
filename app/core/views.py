from django.shortcuts import render
from django.http import JsonResponse

from app.core.modal import NameGenerator

def index(request):
    return render(
        request,
        "index.html",
        {
            "title": "Django example",
        },
    )


def names(request):
    names = NameGenerator().get_names()
    return JsonResponse({"names": names})
