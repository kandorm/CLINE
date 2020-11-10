from nltk.corpus import wordnet as wn
#from nltk.stem import WordNetLemmatizer
from lemminflect import getInflection

#wnl = WordNetLemmatizer()

REPLACE_TAG = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # [NNP, NNPS]
REPLACE_POS = ['NOUN', 'VERB', 'ADJ', 'ADV']
POS_TO_TAGS = {'NOUN': ['NN', 'NNS'], 
               'ADJ': ['JJ', 'JJR', 'JJS'],
               'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
               'ADV': ['RB', 'RBR', 'RBS']}


def get_synonym(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_synset = set()
    if pos not in REPLACE_POS:
        return list(word_synset)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        words = synset.lemma_names()
        for word in words:
            #word = wnl.lemmatize(word, pos=eval("wn."+pos))
            if word.lower() != text.lower() and word.lower() != lemma.lower():
                # inflt = getInflection(word, tag=tag)
                # word = inflt[0] if len(inflt) else word
                word = word.replace('_', ' ')
                word_synset.add(word)

    return list(word_synset)


def get_hypernyms(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_hypernyms = set()
    if pos not in REPLACE_POS:
        return list(word_hypernyms)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        for hyperset in synset.hypernyms():
            words = hyperset.lemma_names()
            for word in words:
                #word = wnl.lemmatize(word, pos=eval("wn."+pos))
                if word.lower() != text.lower() and word.lower() != lemma.lower():
                    # inflt = getInflection(word, tag=tag)
                    # word = inflt[0] if len(inflt) else word
                    word = word.replace('_', ' ')
                    word_hypernyms.add(word)

    return list(word_hypernyms)


def get_antonym(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_antonym = set()
    if pos not in REPLACE_POS:
        return list(word_antonym)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        for synlemma in synset.lemmas():
            for antonym in synlemma.antonyms():
                word = antonym.name()
                #word = wnl.lemmatize(word, pos=eval("wn."+pos))
                if word.lower() != text.lower() and word.lower() != lemma.lower():
                    # inflt = getInflection(word, tag=tag)
                    # word = inflt[0] if len(inflt) else word
                    word = word.replace('_', ' ')
                    word_antonym.add(word)

    return list(word_antonym)


def get_lemminflect(token):
    text = token.text
    lemma = token.lemma_
    tag = token.tag_
    pos = token.pos_
    word_lemminflect = set()
    if pos not in REPLACE_POS:
        return list(word_lemminflect)

    tags = POS_TO_TAGS[pos]
    for tg in tags:
        if tg == tag: continue
        inflects = getInflection(lemma, tag=tg)
        for word in inflects:
            if word.lower() != text.lower():
                word_lemminflect.add(word)

    return list(word_lemminflect)
