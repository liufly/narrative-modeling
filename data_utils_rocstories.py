from __future__ import absolute_import

import os, sys
import re
import numpy as np
import xml.etree.ElementTree
from collections import defaultdict
import nltk
import operator
import json
import random

def vectorize_data(stories, max_story_size, max_ending_size, word_processor):
    ret_stories = word_processor.transform(
        [s[1] for s in stories]
    )
    # [None, max_story_size]
    assert ret_stories.shape[1] == max_story_size

    ret_endings = word_processor.transform(
        [s[2] for s in stories]
    )
    ret_endings = ret_endings[:, :max_ending_size]
    # [None, max_ending_size]

    ret_labels = [s[3] for s in stories]
    ret_labels = np.array(ret_labels, dtype=np.int32)
    ret_labels = ret_labels.reshape(-1, 1)

    assert len(ret_stories) == len(stories)
    assert len(ret_stories) == len(ret_endings)
    assert len(ret_stories) == len(ret_labels)

    return ret_stories, ret_endings, ret_labels

def vectorize_event_sentiment_topic(event, sentiment, topic, max_story_size):
    assert len(event) == len(sentiment)
    assert len(event) == len(topic)
    n = len(event)
    ret_event = np.zeros((n, max_story_size))
    ret_sentiment = np.zeros((n, max_story_size))
    ret_topic = np.zeros((n, max_story_size))
    for i, e in enumerate(event):
        if len(e) > max_story_size:
            e = e[:max_story_size]
        ret_event[i, :len(e)] = e
    for i, s in enumerate(sentiment):
        if len(s) > max_story_size:
            s = s[:max_story_size]
        ret_sentiment[i, :len(s)] = s
    for i, t in enumerate(topic):
        if len(t) > max_story_size:
            t = t[:max_story_size]
        ret_topic[i, :len(t)] = t
    return ret_event, ret_sentiment, ret_topic

def get_all_vocab(data):
    vocab_set = set()
    for story_id, story, ending, answer in data:
        vocab_set |= set(story)
        vocab_set |= set(ending)
    return vocab_set
