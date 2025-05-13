from django.conf import settings

def target_column_processor(request):
    # Retrieve the target_column from the session
    target_column = request.session.get('target_column', None)
    return {
        'current_column': target_column
    }
