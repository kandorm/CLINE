# -*- coding:utf-8 -*-
import os
import sys
import spacy
import re
from multiprocessing import Process
from argparse import ArgumentParser

nlp = spacy.load('en_core_web_sm')
#boundary = re.compile('[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]')
def custom_seg(doc):
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text in ['"', "'", "‘", "’", "“", "”"] and index!=(length - 1):
            doc[index+1].sent_start = False
    return doc
nlp.add_pipe(custom_seg, before='parser')

def get_articles(path):
    file = open(path, "r", encoding='utf-8')
    articles = [eval(x)['text'] for x in file.readlines()]
    file.close()
    return articles

def get_sentences(article):
    doc = nlp(article)
    sents = list(doc.sents)
    return sents

class MyProcess(Process):
    def __init__(self, files, dirname, outname):
        super(MyProcess, self).__init__()
        self.files = files
        self.dirname = dirname
        self.outname = outname

    def run(self):
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        outfile = open(os.path.join(self.dirname, str(self.outname)), 'w', encoding="utf-8")
        for idx, path in enumerate(self.files):
            #print(idx)
            articles = get_articles(path)
            for arti in articles:
                arti = str(arti)
                arti = arti.strip()
                arti = re.sub('[\s]+', ' ', arti)
                arti = arti.strip()
                if not arti: continue
                outfile.write('{}\n'.format(arti))
                # sents = get_sentences(arti)
                # for sen in sents:
                #     sen = str(sen)
                #     sen = sen.strip()
                #     sen = re.sub('[\n]+', ' ', sen)
                #     sen = sen.strip()
                #     if not sen: continue
                #     #if len(sen) < 2: continue
                #     #print(sen.encode('ascii'))
                #     outfile.write('{}\n'.format(sen))
                # outfile.write('\n')
        outfile.close()

def bisector_list(tabulation, num):
    seg = len(tabulation)//num
    ans = []
    for i in range(num):
        start = i*seg
        end = (i+1)*seg if i!=num-1 else len(tabulation)
        ans.append(tabulation[start:end])
    return ans

def walk(path):
    out = []
    for root, dirs, files in os.walk(path):
        for name in files:
            out.append(os.path.join(root, name))
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, nargs='?', required=True, help="")
    parser.add_argument("--output_path", type=str, nargs='?', required=True, help="")
    parser.add_argument("--processnum", type=int, default=6, nargs='?', help="")
    args = parser.parse_args()

    dir_path = args.input_path
    out_path = args.output_path
    process_num = args.processnum

    files = walk(dir_path)
    n_files = bisector_list(files, process_num)

    processes = []
    for i in range(process_num):
        p = MyProcess(n_files[i], out_path, i)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
