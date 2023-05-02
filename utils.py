
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
LENGTH = 600
ANS_LEN = 5


def clean_background(bg):
    bg = bg.split('For more information')[0]
    bg = bg.split('; for more information')[0]
    bg = bg.split('see:  www')[0]
    bg = bg.split('(www')[0]
    bg = bg.split('( http')[0]
    bg = bg.split('(http')[0].rstrip()+ ' '
    return bg