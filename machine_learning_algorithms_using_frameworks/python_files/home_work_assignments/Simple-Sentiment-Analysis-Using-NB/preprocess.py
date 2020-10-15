import nltk.classify
from nltk.corpus import movie_reviews
from nltk.corpus import opinion_lexicon
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")

def create_word_features_pos(words):
    useful_words = [word for word in words if word not in stop_words]
    my_list = [({word: True}, 'positive') for word in useful_words]
    return my_list


def create_word_features_neg(words):
    useful_words = [word for word in words if word not in stop_words]
    my_list = [({word: True}, 'negative') for word in useful_words]
    return my_list


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    
    pos_txt = get_tokenized_file(u"positive-words.txt")
    neg_txt = get_tokenized_file(u"negative-words.txt")
    
    my_dict = dict([(word, True) for word in pos_txt if word in useful_words])
    my_dict1 = dict([(word, False) for word in neg_txt if word in useful_words])
    
    my_dict.update(my_dict1)
    
    return my_dict

def get_tokenized_file(file):
    return word_tokenize(open(file, 'r').read())

def get_data():
	print("Collecting Negative Words")
	neg_txt = get_tokenized_file(u"negative-words.txt")
	neg_features = create_word_features_neg(neg_txt)

	print("Collecting Positive Words")
	pos_txt = get_tokenized_file(u"positive-words.txt")
	pos_features = create_word_features_pos(pos_txt)
	return pos_features + neg_features

def process(data):
    return [word.lower() for word in word_tokenize(data)]