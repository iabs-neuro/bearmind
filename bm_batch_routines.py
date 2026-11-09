"""Session names from file names; bm_examinator takes extract_name_with_pattern from here."""
from naming import extract_session_id


def find_identifiers(text):
    """Extract session identifier from text using flexible pattern."""
    identifier = extract_session_id(text)
    if not identifier:
        print("ID not found in text:", text)
    return identifier

def extract_name_with_pattern(text):
    """Extract session name with pattern - alias for extract_session_id."""
    result = extract_session_id(text)
    if not result:
        print(f'Pattern not found in text: {text}')
    return result
