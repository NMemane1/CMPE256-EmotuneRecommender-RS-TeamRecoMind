from typing import List, Dict


# Hardcoded song recommendations based on emotions
EMOTION_SONG_MAP = {
    "Joy": [
        {"title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "year": 2013},
        {"title": "Good Vibrations", "artist": "The Beach Boys", "genre": "Rock", "year": 1966},
        {"title": "Walking on Sunshine", "artist": "Katrina and the Waves", "genre": "Pop", "year": 1983},
        {"title": "Don't Stop Me Now", "artist": "Queen", "genre": "Rock", "year": 1978},
        {"title": "I Got You (I Feel Good)", "artist": "James Brown", "genre": "Soul", "year": 1965}
    ],
    "Sadness": [
        {"title": "Someone Like You", "artist": "Adele", "genre": "Pop", "year": 2011},
        {"title": "The Night We Met", "artist": "Lord Huron", "genre": "Indie", "year": 2015},
        {"title": "Fix You", "artist": "Coldplay", "genre": "Alternative", "year": 2005},
        {"title": "Hurt", "artist": "Johnny Cash", "genre": "Country", "year": 2002},
        {"title": "Mad World", "artist": "Gary Jules", "genre": "Alternative", "year": 2001}
    ],
    "Anger": [
        {"title": "Break Stuff", "artist": "Limp Bizkit", "genre": "Nu Metal", "year": 1999},
        {"title": "Killing in the Name", "artist": "Rage Against the Machine", "genre": "Rock", "year": 1992},
        {"title": "Bodies", "artist": "Drowning Pool", "genre": "Metal", "year": 2001},
        {"title": "You're Gonna Go Far, Kid", "artist": "The Offspring", "genre": "Punk Rock", "year": 2008},
        {"title": "Sabotage", "artist": "Beastie Boys", "genre": "Hip Hop", "year": 1994}
    ],
    "Calmness": [
        {"title": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "year": 2011},
        {"title": "Breathe Me", "artist": "Sia", "genre": "Alternative", "year": 2004},
        {"title": "The Scientist", "artist": "Coldplay", "genre": "Alternative", "year": 2002},
        {"title": "Strawberry Swing", "artist": "Coldplay", "genre": "Alternative", "year": 2008},
        {"title": "Pure Shores", "artist": "All Saints", "genre": "Pop", "year": 2000}
    ],
    "Excitement": [
        {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "Funk", "year": 2014},
        {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "genre": "Pop", "year": 2016},
        {"title": "Shut Up and Dance", "artist": "Walk the Moon", "genre": "Pop Rock", "year": 2014},
        {"title": "September", "artist": "Earth, Wind & Fire", "genre": "Funk", "year": 1978},
        {"title": "Mr. Blue Sky", "artist": "Electric Light Orchestra", "genre": "Rock", "year": 1977}
    ],
    "Anxiety": [
        {"title": "Breathe", "artist": "Telepopmusik", "genre": "Electronic", "year": 2001},
        {"title": "Let It Be", "artist": "The Beatles", "genre": "Rock", "year": 1970},
        {"title": "Don't Worry, Be Happy", "artist": "Bobby McFerrin", "genre": "Reggae", "year": 1988},
        {"title": "Three Little Birds", "artist": "Bob Marley", "genre": "Reggae", "year": 1977},
        {"title": "Here Comes the Sun", "artist": "The Beatles", "genre": "Rock", "year": 1969}
    ],
    "Romantic": [
        {"title": "Thinking Out Loud", "artist": "Ed Sheeran", "genre": "Pop", "year": 2014},
        {"title": "Perfect", "artist": "Ed Sheeran", "genre": "Pop", "year": 2017},
        {"title": "All of Me", "artist": "John Legend", "genre": "R&B", "year": 2013},
        {"title": "Make You Feel My Love", "artist": "Adele", "genre": "Pop", "year": 2008},
        {"title": "Wonderful Tonight", "artist": "Eric Clapton", "genre": "Rock", "year": 1977}
    ],
    "Concentration": [
        {"title": "Time", "artist": "Hans Zimmer", "genre": "Soundtrack", "year": 2010},
        {"title": "Clair de Lune", "artist": "Claude Debussy", "genre": "Classical", "year": 1905},
        {"title": "Porcelain", "artist": "Moby", "genre": "Electronic", "year": 1999},
        {"title": "Intro", "artist": "The xx", "genre": "Indie", "year": 2009},
        {"title": "Teardrop", "artist": "Massive Attack", "genre": "Trip Hop", "year": 1998}
    ],
    "Amusement": [
        {"title": "Crazy Little Thing Called Love", "artist": "Queen", "genre": "Rock", "year": 1979},
        {"title": "Dancing Queen", "artist": "ABBA", "genre": "Pop", "year": 1976},
        {"title": "I Wanna Dance with Somebody", "artist": "Whitney Houston", "genre": "Pop", "year": 1987},
        {"title": "Lovely Day", "artist": "Bill Withers", "genre": "Soul", "year": 1977},
        {"title": "Good Times", "artist": "Chic", "genre": "Disco", "year": 1979}
    ],
    "Surprise": [
        {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "year": 1975},
        {"title": "Paranoid Android", "artist": "Radiohead", "genre": "Alternative", "year": 1997},
        {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Grunge", "year": 1991},
        {"title": "Thunderstruck", "artist": "AC/DC", "genre": "Rock", "year": 1990},
        {"title": "Seven Nation Army", "artist": "The White Stripes", "genre": "Rock", "year": 2003}
    ]
}


def get_recommendations(top_emotion: str, all_emotions: List[Dict] = None, limit: int = 5) -> List[Dict]:
    """
    Get song recommendations based on detected emotions.
    
    This function will be replaced with actual recommender system integration.
    For now, it returns hardcoded songs based on the top emotion.
    
    Args:
        top_emotion: The primary detected emotion name
        all_emotions: Full list of detected emotions with scores (for future use)
        limit: Maximum number of recommendations to return
        
    Returns:
        List of song recommendation dictionaries
    """
    # Normalize emotion name (capitalize first letter)
    emotion = top_emotion.capitalize()
    
    # Get songs for the emotion, or default to Joy if not found
    songs = EMOTION_SONG_MAP.get(emotion, EMOTION_SONG_MAP["Joy"])
    
    # Return limited results
    return songs[:limit]


def get_all_emotions() -> List[str]:
    """
    Get list of all supported emotions
    
    Returns:
        List of emotion names
    """
    return list(EMOTION_SONG_MAP.keys())
