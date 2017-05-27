import codecs
import nltk
import string
import numpy as np
import os
import re
class DataLoader():
    def __init__(self,args):
        self.args=args
        self.vocab_size=max(map(lambda x:ord(x),string.printable))
        self.PAD = 0
        self.PAD_CHAR =chr(0)
        self.reg = re.compile(r'[\.,\?!;]')

    def replace_punc(self,txt):

        return self.reg.sub('',txt)
    def load_txt_data(self):
        with codecs.open(self.args.data_path,encoding='utf8') as f:
            txt = f.read()
            sentances = txt.split('\n')#nltk.sent_tokenize(txt)
            sentances = list(map(lambda x:str(x.encode('ascii',errors='ignore')),sentances))
            self.max_len = max(map(len,sentances))
            self.max_len =500
            sentances = map(lambda x: x.ljust(self.max_len,self.PAD_CHAR)[:self.max_len],sentances)
            org_lower = list(map(lambda x: (x,self.replace_punc(x.lower()).ljust(self.max_len,self.PAD_CHAR)[:self.max_len]),sentances))
        return org_lower

    @staticmethod
    def is_pure_ascii(sent):
        set(sent).issubset(string.printable)


    def txt_to_array(self,org_lower):
        temp_org,temp_low =[], []
        for (o,l) in org_lower:
            temp_org.append(np.array(list(map(ord,o))))
            temp_low.append(np.array(list(map(ord, l))))
        org = np.stack(temp_org)
        low = np.stack(temp_low)
        return np.stack([org,low],axis=1) # [num_sent,2,max_len]

    def load_data(self):
        if os.path.exists(self.args.saved_data_path):
            self.dataset = np.load(self.args.saved_data_path)
        else:
            org_low = self.load_txt_data()
            self.dataset = self.txt_to_array(org_low)
            np.save(self.args.saved_data_path,self.dataset)


    def get_batch(self):
        start=0
        end = start+self.args.batch_size
        np.random.shuffle(self.dataset)
        while start < len(self.dataset):
            batch = self.dataset[start:end]
            original = batch[:,0,:]
            lowered = batch[:, 1, :]
            yield original,lowered
            start+=self.args.batch_size
            end  += self.args.batch_size

    def ar_to_str(self,ar):
        return ''.join(map(chr, ar))






