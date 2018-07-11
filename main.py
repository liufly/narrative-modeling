from __future__ import absolute_import
from __future__ import print_function

from data_utils_rocstories import *
from vocab_processor import *
from sklearn import metrics
from entnet_rocstories import EntNet_ROCStories
from itertools import chain
from six.moves import range, reduce
from sklearn.model_selection import train_test_split
from collections import defaultdict

import tensorflow as tf
import numpy as np

import os
import sys
import random
import logging

import pprint
pp = pprint.PrettyPrinter()

tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
tf.flags.DEFINE_string("task", "ROCStories", "ROCStories")
tf.flags.DEFINE_integer("random_state", 83, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/rocstories/", "Directory containing ROC story data")
tf.flags.DEFINE_string("opt", "ftrl", "Optimizer [ftrl]")
tf.flags.DEFINE_string("embedding_file_path", None, "Embedding file path [None]")
tf.flags.DEFINE_boolean("update_embeddings", False, "Update embeddings [False]")
tf.flags.DEFINE_boolean("case_folding", False, "Case folding [False]")
tf.flags.DEFINE_integer("n_cpus", 6, "N CPUs [1]")
tf.flags.DEFINE_integer("n_keys", 4, "Number of keys [8]")
tf.flags.DEFINE_float("input_keep_prob", 0.5, "input keep prob [0.5]")
tf.flags.DEFINE_float("output_keep_prob", 1.0, "output keep prob [1.0]")
tf.flags.DEFINE_float("state_keep_prob", 1.0, "state keep prob [1.0]")
tf.flags.DEFINE_float("entnet_input_keep_prob", 0.8, "entnet input keep prob [0.8]")
tf.flags.DEFINE_float("entnet_output_keep_prob", 1.0, "entnet output keep prob [1.0]")
tf.flags.DEFINE_float("entnet_state_keep_prob", 1.0, "entnet state keep prob [1.0]")
tf.flags.DEFINE_float("final_layer_keep_prob", 0.8, "final layer keep prob [0.8]")
tf.flags.DEFINE_boolean("debug", False, "Debug [False]")
tf.flags.DEFINE_float("l2_final_layer", 0.001, "Lambda L2 final layer [0.001]")
tf.flags.DEFINE_string("save_model", None, "Path to save the model [None]")
tf.flags.DEFINE_string("load_model", None, "Path to load the model [None]")
tf.flags.DEFINE_string("opinion_lexicon_dir", "data/opinion-lexicon/", "Opinion lexicon")
tf.flags.DEFINE_float("tagging_weight", 0.5, "Weight on the tagging task [0.5]")

FLAGS = tf.flags.FLAGS

def convert_labels(labels):
    ret = []
    assert len(labels) % 2 == 0
    for i in range(0, len(labels), 2):
        first_ending = labels[i][0]
        second_ending = labels[i + 1][0]
        if first_ending >= second_ending:
            ret.append(0)
        else:
            ret.append(1)
    assert len(ret) == len(labels) / 2
    return ret

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(" ".join(sys.argv))
    logger.info("Started Task: %s" % FLAGS.task)
    
    logger.info(pp.pformat(FLAGS.__flags))

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=FLAGS.n_cpus,
        inter_op_parallelism_threads=FLAGS.n_cpus,
    )

    with tf.Session(config=session_conf) as sess:

        np.random.seed(FLAGS.random_state)

        train_stories = json.load(open("data/train.json"))
        val_stories = json.load(open("data/val.json"))
        test_stories = json.load(open("data/test.json"))

        all_vocab = get_all_vocab(train_stories + val_stories + test_stories)

        train_event_feats = [d[1] for d in json.load(open("data/train_event.json"))]
        val_event_feats = [d[1] for d in json.load(open("data/val_event.json"))]
        test_event_feats = [d[1] for d in json.load(open("data/test_event.json"))]

        train_sentiment_feats = [d[1] for d in json.load(open("data/train_sentiment.json"))]
        val_sentiment_feats = [d[1] for d in json.load(open("data/val_sentiment.json"))]
        test_sentiment_feats = [d[1] for d in json.load(open("data/test_sentiment.json"))]

        train_topic_feats = [d[1] for d in json.load(open("data/train_topic.json"))]
        val_topic_feats = [d[1] for d in json.load(open("data/val_topic.json"))]
        test_topic_feats = [d[1] for d in json.load(open("data/test_topic.json"))]

        assert len(train_stories) == len(train_event_feats)
        assert len(train_stories) == len(train_topic_feats)
        assert len(train_stories) == len(train_sentiment_feats)
        assert len(val_stories) == len(val_event_feats)
        assert len(val_stories) == len(val_topic_feats)
        assert len(val_stories) == len(val_sentiment_feats)
        assert len(test_stories) == len(test_event_feats)
        assert len(test_stories) == len(test_topic_feats)
        assert len(test_stories) == len(test_sentiment_feats)
        
        assert np.array_equal([len(s[1]) for s in train_stories], [len(f) for f in train_event_feats])
        assert np.array_equal([len(s[1]) for s in train_stories], [len(f) for f in train_topic_feats])
        assert np.array_equal([len(s[1]) for s in train_stories], [len(f) for f in train_sentiment_feats])
        assert np.array_equal([len(s[1]) for s in val_stories], [len(f) for f in val_event_feats])
        assert np.array_equal([len(s[1]) for s in val_stories], [len(f) for f in val_topic_feats])
        assert np.array_equal([len(s[1]) for s in val_stories], [len(f) for f in val_sentiment_feats])
        assert np.array_equal([len(s[1]) for s in test_stories], [len(f) for f in test_event_feats])
        assert np.array_equal([len(s[1]) for s in test_stories], [len(f) for f in test_topic_feats])
        assert np.array_equal([len(s[1]) for s in test_stories], [len(f) for f in test_sentiment_feats])

        data = train_stories + val_stories + test_stories

        max_story_size = max(map(lambda x: len(x[1]), data))
        logger.info('Max story size: %d' % max_story_size)
        max_ending_size = max(map(lambda x: len(x), [s[2] for s in data]))
        logger.info('Max ending size: %d' % max_ending_size)

        train_event_feats, train_sentiment_feats, train_topic_feats = vectorize_event_sentiment_topic(
            train_event_feats, train_sentiment_feats, train_topic_feats, max_story_size
        )
        val_event_feats, val_sentiment_feats, val_topic_feats = vectorize_event_sentiment_topic(
            val_event_feats, val_sentiment_feats, val_topic_feats, max_story_size
        )
        test_event_feats, test_sentiment_feats, test_topic_feats = vectorize_event_sentiment_topic(
            test_event_feats, test_sentiment_feats, test_topic_feats, max_story_size
        )

        train_stories_feat = np.concatenate(
            [
                np.expand_dims(train_event_feats, axis=-1),
                np.expand_dims(train_sentiment_feats, axis=-1),
                np.expand_dims(train_topic_feats, axis=-1),
            ],
            axis=2
        )
        # [None, stori_size, 3]
        val_stories_feat = np.concatenate(
            [
                np.expand_dims(val_event_feats, axis=-1),
                np.expand_dims(val_sentiment_feats, axis=-1),
                np.expand_dims(val_topic_feats, axis=-1),
            ],
            axis=2
        )
        # [None, stori_size, 3]
        test_stories_feat = np.concatenate(
            [
                np.expand_dims(test_event_feats, axis=-1),
                np.expand_dims(test_sentiment_feats, axis=-1),
                np.expand_dims(test_topic_feats, axis=-1),
            ],
            axis=2
        )
        # [None, stori_size, 3]

        embedding_size = FLAGS.embedding_size

        word_vocab = EmbeddingVocabulary(
            in_file=FLAGS.embedding_file_path,
            vocab_list=all_vocab,
        )
        word_vocab_processor = EmbeddingVocabularyProcessor(
            max_document_length=max_story_size,
            vocabulary=word_vocab,
        )
        embedding_mat = word_vocab.embeddings
        embedding_size = word_vocab.embeddings.shape[1]

        train_stories, train_endings, train_labels = vectorize_data(train_stories, max_story_size, max_ending_size, word_vocab_processor)
        val_stories, val_endings, val_labels = vectorize_data(val_stories, max_story_size, max_ending_size, word_vocab_processor)
        test_stories, test_endings, test_labels = vectorize_data(test_stories, max_story_size, max_ending_size, word_vocab_processor)

        vocab_size = len(word_vocab)

        logger.info("Training stories shape " + str(train_stories.shape))
        logger.info("Training endings shape " + str(train_endings.shape))
        logger.info("Training labels shape " + str(train_labels.shape))
        logger.info("Validation stories shape " + str(val_stories.shape))
        logger.info("Validation endings shape " + str(val_endings.shape))
        logger.info("Validation labels shape " + str(val_labels.shape))
        logger.info("Test stories shape " + str(test_stories.shape))
        logger.info("Test endings shape " + str(test_endings.shape))
        logger.info("Test labels shape " + str(test_labels.shape))

        # params
        n_train = train_stories.shape[0]
        n_val = val_stories.shape[0]
        n_test = test_stories.shape[0]
        
        tf.set_random_seed(FLAGS.random_state)
        batch_size = FLAGS.batch_size
        
        global_step = None
        optimizer = None

        if FLAGS.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
        elif FLAGS.opt == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate)

        batches = zip(range(0, max(1, n_train-batch_size), batch_size), range(batch_size, max(batch_size + 1, n_train), batch_size))
        batches = [(start, end) for start, end in batches]
        
        last_train_acc, last_val_acc = None, None

        model = EntNet_ROCStories(
            batch_size, 
            vocab_size, 
            max_story_size, 
            max_ending_size,
            embedding_size, 
            session=sess,
            embedding_mat=None if (FLAGS.embedding_file_path is None and FLAGS.embedding_pickle_path is None) else word_vocab.embeddings,
            update_embeddings=FLAGS.update_embeddings,
            n_keys=FLAGS.n_keys,
            l2_final_layer=FLAGS.l2_final_layer,
            tagging_weight=FLAGS.tagging_weight,
            max_grad_norm=FLAGS.max_grad_norm, 
            optimizer=optimizer,
            global_step=global_step,
        )

        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            total_training_instances = 0
            for start, end in batches:
                stories = train_stories[start:end]
                endings = train_endings[start:end]
                answers = train_labels[start:end]
                cost_t = model.fit(
                    stories, endings, answers, 
                    FLAGS.input_keep_prob,
                    FLAGS.output_keep_prob, 
                    FLAGS.state_keep_prob,
                    FLAGS.entnet_input_keep_prob,
                    FLAGS.entnet_output_keep_prob,
                    FLAGS.entnet_state_keep_prob,
                    FLAGS.final_layer_keep_prob,
                    train_stories_feat[start:end],
                )
                total_cost += cost_t
                total_training_instances += len(train_stories[start:end])

            if t % FLAGS.evaluation_interval == 0:
                train_preds = model.predict(
                    train_stories, train_endings, 
                    batch_size=batch_size,
                )
                
                train_acc = metrics.accuracy_score(
                    convert_labels(train_labels),
                    convert_labels(np.array(train_preds)),
                )

                val_preds = model.predict(
                    val_stories, val_endings,
                    batch_size=batch_size,
                )
    
                val_acc = metrics.accuracy_score(
                    convert_labels(val_labels), 
                    convert_labels(np.array(val_preds))
                )

                test_preds = model.predict(
                    test_stories, test_endings,
                    batch_size=batch_size
                )
                test_acc = metrics.accuracy_score(
                    convert_labels(test_labels), 
                    convert_labels(np.array(test_preds))
                )

                assert total_training_instances != 0

                logger.info('-----------------------')
                logger.info('Epoch %d' % t)
                logger.info('Avg Cost: %f' % (total_cost / total_training_instances))
                logger.info('Training Accuracy: %f' % train_acc)
                logger.info('Validation Accuracy: %f' % val_acc)
                logger.info('Test Accuracy: %f' % test_acc)
                logger.info('-----------------------')
