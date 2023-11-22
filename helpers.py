import re

def custom_sort(filename):
    # This regex will match the pattern {int}key-{int}n
    match = re.search(r'(\d+)key-(\d+)n', filename)
    if match:
        # Extract the two integers and return them as a tuple
        return (int(match.group(1)), int(match.group(2)))
    else:
        # Return a default value for files that don't match the pattern
        return (0, 0)
