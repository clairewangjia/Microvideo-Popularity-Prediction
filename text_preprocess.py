# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import re
import nltk
# nltk.download()
# from nltk.corpus import stopwords
import simplejson as json
import pickle
import numpy as np
import emoji


def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>',re.S)
    return html_prog.sub('', str)

def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)

def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)

def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)

def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)

def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)

def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def pre_process(str):
    # do not change the preprocessing order only if you know what you're doing 
    # str = str.lower()
    str = rm_url(str)        
    str = rm_at_user(str)        
    str = rm_repeat_chars(str) 
    str = rm_hashtag_symbol(str)       
    str = rm_time(str) 
    str = emoji.demojize(str, delimiters=(" e_", " ")) #.replace("_"," ") # replace emoji with its name/description
    return str
                            



if __name__ == "__main__":
    data_dir = './data'  ##Setting your own file path here.

    x_filename = 'video_text.txt'


    ##load and process samples
    print('start loading and process samples...')
    tweets = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding='utf-8') as f:
        for i, line in enumerate(f):
            postprocess_tweet = pre_process(line.strip())
            tweets.append(postprocess_tweet)
        print(len(tweets))


    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'video_processed.txt'), 'w', encoding='utf-8')
    for tweet in tweets:
        tweet = tweet.replace('\n', ' ').replace('\r', '')
        fout.write('%s\n' %tweet)
    fout.close()

    print("Preprocessing is completed")