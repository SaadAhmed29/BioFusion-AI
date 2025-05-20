from django.shortcuts import render, HttpResponse

def index(request):
    # context is set of variable we send to our template
    context = { 
        'variable': 'this is sent'
    }
    return render(request, 'index.html', context)
    #return HttpResponse('This is homepage')
