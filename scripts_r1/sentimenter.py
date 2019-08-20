"""
This file defines the Sentimenter class for multi-package sentiment analysis

"""

# Helper libraries
import logging
import os
import pandas

# Sentiment analysis libraries
from pattern.nl import sentiment
from polyglot.text import Text
from polyglot.downloader import downloader
from resources.sentistrength import senti_client


logging.basicConfig(level="WARNING")

class Sentimenter():

    logger = logging.getLogger(__name__)

    def __init__(verbose=True, debug=False):

        self.verbose = True

        if debug:
            self.logger.setLevel('DEBUG')

        # ensure polyglot supports NL
        downloader.download('sentiment2.nl')

    def get_sentiment(text):
        sentiment = {
            "pattern"  : sentiment(text)[0],
            "polyglot" : Text(text, hint_language_code='nl').polarity

        }
