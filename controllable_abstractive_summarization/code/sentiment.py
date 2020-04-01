from nltk.corpus import wordnet
import os
import csv
import argparse
import ast

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer, SentiText

B_INCR = 0.293
B_DECR = -0.293


BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, 
     "completely": B_INCR, "considerable": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormous": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptional": B_INCR, "exceptionally": B_INCR, 
     "extreme": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR, 
     "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, 
     "incredible": B_INCR, "incredibly": B_INCR, "intensely": B_INCR, 
     "major": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "total": B_INCR, "totally": B_INCR, "tremendous": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utter": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginal": B_DECR, "marginally": B_DECR,
     "occasional": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarce": B_DECR, "scarcely": B_DECR, "slight": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

def plot_hist(hist, label, diff=None):

    sbplt = 141
    plt.figure(figsize=(16,8)) 
    for key in hist.keys():

        if diff is not None:
            plt_hist = [st - sm for st, sm in zip(hist[key], diff[key])]
        else:
            plt_hist = hist[key]
        
        if key == 'pos':
            color = 'lightgreen'
        elif key == 'neg':
            color = 'salmon'
        elif key =='neu':
            color = 'gold'
        else: 
            color = 'blue'

        plt.subplot(sbplt)
        mean = sum(plt_hist)/len(plt_hist)
        min_ylim, max_ylim = plt.ylim()
        plt.hist(x=plt_hist, bins=10, edgecolor='grey', color=color, alpha=0.7)
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.text(mean*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean))
        plt.xlabel('sentiment score')
        plt.ylabel('count')
        plt.title(key)

        sbplt += 1

    plt.savefig(label + '_hist.png')
    plt.clf()

def get_word_valence(word, sid):
    sentitext = SentiText(word)
    sentiments = []    
    words_and_emoticons = sentitext.words_and_emoticons
    valence = 0
    try:
        sentiments = sid.sentiment_valence(valence, sentitext, words_and_emoticons[0], 0, sentiments)
        return sentiments[0]
    except IndexError:
        return 0

def get_words_valence_scores(text, sid):
    sentitext = SentiText(text)
    sentiments = []
    words_and_emoticons = sentitext.words_and_emoticons

    for i, item in enumerate(words_and_emoticons):
        valence = 0
        if item.lower() in BOOSTER_DICT:
            sentiments.append(valence)
            continue
        if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
            words_and_emoticons[i + 1].lower() == "of"):
            sentiments.append(valence)
            continue

        sentiments = sid.sentiment_valence(valence, sentitext, item, i, sentiments)
    sentiments = sid._but_check(words_and_emoticons, sentiments)
    sentiment_scores = sid.score_valence(sentiments, text)

    sentind = np.nonzero(sentiments)[0]
    sentvalues = np.asarray(sentiments)[sentind]
    sentwords = np.asarray(words_and_emoticons)[sentind]
    sentwindows = np.asarray([words_and_emoticons[max(0, i-2):i+3] for i in sentind])
    # dic = {word: (value, window) for word, value, window in zip(sentwords, sentvalues, sentwindows)}
    return sentiment_scores, sentvalues, sentwords, sentwindows

def add_synonyms(top_words, limit = 4):
    all_synonyms = []
    for word in top_words:
        syns = []
        synset = wordnet.synsets(word)
        for synsubset in synset:
            for i, syn in enumerate(synsubset.lemmas()):
                if syn.name() != word:
                    if i > (limit-1):
                        break
                    syns.append(syn.name())
            if len(syns) > 0:
                break

        all_synonyms.append(syns)
    return all_synonyms

def print_word_frequency_results(words, scores, freqs, sid, k, title):
    scores = np.asarray(scores)
    pos_ind = np.where(scores > 0)
    neg_ind = np.where(scores < 0)
    top_words_pos = words[pos_ind][0:k]
    top_scores_pos = scores[pos_ind][0:k]
    top_freqs_pos = freqs[pos_ind][0:k]
    pos_synonyms = add_synonyms(top_words_pos)

    top_words_neg = words[neg_ind][0:k]
    top_scores_neg = scores[neg_ind][0:k]
    top_freqs_neg = freqs[neg_ind][0:k]
    neg_synonyms = add_synonyms(top_words_neg)

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    # pd.set_option('display.max_colwidth', -1)  # or 199
    pos_words_val, pos_syns_val, pos_avg_syns_val = analyze_synonyms(top_words_pos, pos_synonyms, sid)
    neg_words_val, neg_syns_val, neg_avg_syns_val = analyze_synonyms(top_words_neg, neg_synonyms, sid)

    df_pos = pd.DataFrame(data={'words':top_words_pos, 'sum_scores':top_scores_pos, 'ind_scores': pos_words_val,
                                'frequencies':top_freqs_pos, 'synonyms':pos_synonyms, 'syn_scores': pos_avg_syns_val})
    print(df_pos)

    df_neg = pd.DataFrame(data={'words':top_words_neg, 'sum_scores':top_scores_neg, 'ind_scores': neg_words_val, 
                                'frequencies':top_freqs_neg, 'synonyms':neg_synonyms, 'syn_scores': neg_avg_syns_val})
    print(df_neg)
    with pd.ExcelWriter('analysis' + title + '.xlsx') as writer:  
        df_pos.to_excel(writer, sheet_name='positive words')
        df_neg.to_excel(writer, sheet_name='negative words')


    return {'pos':df_pos, 'neg':df_neg}

def analyze_synonyms(words, synonyms, sid):
    words_val = []
    syns_val = []
    avg_syns_val = []
    for word, synset in zip(words, synonyms):
        words_val.append(get_word_valence(word, sid))
        syns_valence = []
        if len(synset) > 0:
            for syn in synset:
                syns_valence.append(get_word_valence(syn, sid))
            avg_syns_val.append(sum(syns_valence)/len(syns_valence))
        else:
            avg_syns_val.append(0)
        syns_val.append(syns_valence)
    return words_val, syns_val, avg_syns_val
    







def analyze_sentiment(full):
    data_path = os.path.join(os.getcwd(), 'data/')
    processed_data = {'id': [], 'stories':[], 'length_tokens': [], 
                        'length_sentences': [], 'source': [], 'entities': [], 'summary': []}
    if full:
        csv_path = os.path.join(data_path, 'cnn_dailymail.csv')
    else:
        csv_path = os.path.join(data_path, 'cnn_dailymail_test_purposes.csv')

    
    sentiments_summary = {'neu':[], 'pos':[], 'neg':[], 'compound':[]}
    sentiments_story = {'neu':[], 'pos':[], 'neg':[], 'compound':[]}
    valence_words_summary = []
    valence_words_story = []
    
    sid = SentimentIntensityAnalyzer()
    counter = 0

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file, fieldnames=processed_data.keys())

        
        for no, line in enumerate(reader):
            tmp_words_summary = {'words': [], 'scores': [], 'windows': []}
            tmp_words_story = {'words': [], 'scores': [], 'windows': []}

            story = line['stories'].replace('@@ ', '')
            summary = line['summary'].replace('@@ ', '')

            # entities = ast.literal_eval(line['entities'])
            # for entity in entities.keys():
            #     story = story.replace(' ' + entity + ' ', ' ' + entities[entity] + ' ')
            #     summary = summary.replace(' ' + entity + ' ', ' ' + entities[entity] + ' ')
    
            sentiment_scores, sentvalues, sentwords, sentwindows = get_words_valence_scores(story, sid)
            for key in sentiment_scores.keys():
                sentiments_story[key].append(sentiment_scores[key])
            valence_words_story.append({'words': sentwords, 'scores': sentvalues, 'windows': sentwindows})
            
            sentiment_scores, sentvalues, sentwords, sentwindows = get_words_valence_scores(summary, sid)
            for key in sentiment_scores.keys():
                sentiments_summary[key].append(sentiment_scores[key])
            valence_words_summary.append({'words': sentwords, 'scores': sentvalues, 'windows': sentwindows})
    all_words_story = np.concatenate([valence_words_story[i]['words'] for i in range(len(valence_words_story))])
    words_story, freqs_story = np.unique(all_words_story,return_counts=True)
    words_story = words_story[np.argsort(freqs_story)[::-1]]
    freqs_story[::-1].sort()

    all_words_summary = np.concatenate([valence_words_summary[i]['words'] for i in range(len(valence_words_summary))])
    words_summary, freqs_summary = np.unique(all_words_summary,return_counts=True)
    words_summary = words_summary[np.argsort(freqs_summary)[::-1]]
    freqs_summary[::-1].sort()

    scores_story = []
    all_scores_story = np.concatenate([valence_words_story[i]['scores'] for i in range(len(valence_words_story))])
    for word in words_story:
        scores_story.append(np.sum(all_scores_story[np.where(all_words_story == word)]))
    scores_summary = []
    all_scores_summary = np.concatenate([valence_words_summary[i]['scores'] for i in range(len(valence_words_summary))])
    for word in words_summary:
        scores_summary.append(np.sum(all_scores_summary[np.where(all_words_summary == word)]))
    df_story = print_word_frequency_results(words_story, scores_story, freqs_story, sid, 100, 'stories')
    df_summary = print_word_frequency_results(words_summary, scores_summary, freqs_summary, sid, 100, 'summaries')
    print(f'Summary positive, word average: {df_summary["pos"].ind_scores.mean()}')
    print(f'Summary positive, synonym-word average: {(df_summary["pos"].syn_scores - df_story["pos"].ind_scores).mean()}')
    print(f'Summary negative, word average: {df_summary["neg"].ind_scores.mean()}')
    print(f'Summary negative, synonym-word average: {(df_summary["neg"].syn_scores - df_story["neg"].ind_scores).mean()}')

    # plot_hist(sentiments_summary, 'summary')

    # plot_hist(sentiments_story, 'story')

    # plot_hist(sentiments_summary, 'diff', sentiments_story)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--full', action='store_true',
                        help='Analyze full dataset')
    
    args = parser.parse_args()
    analyze_sentiment(args.full)
