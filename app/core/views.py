import os
from django.shortcuts import render
from django.http import JsonResponse

from app.core.modal import NameGenerator

def index(request):
    return render(
        request,
        "index.html",
        {
            "title": "Django example alkdjf",
        },
    )


def names(request):
    nums = request.GET.get('num_results', 10)
    names_val = NameGenerator().get_names(nums)
    return JsonResponse({"names_url": names_val})
