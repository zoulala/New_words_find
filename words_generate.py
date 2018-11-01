# -*- coding=utf8 -*-

"""
根据大规模语料，自动生成词库，可以用于：挖掘流行词、分词、聊天习惯和兴趣 等
# 技术：统计词频、互信息、信息熵 http://blog.csdn.net/xiaokang06/article/details/50616983

time    :2017-7-6
author  : zlw

others ： 在pyhton3中encode,在转码的同时还会把string 变成bytes类型，decode在解码的同时还会把bytes变回string
"""

from __future__ import division
import time

import re
from math import log


# hanzi_re = re.compile(u"[\u4E00-\u9FD5]+", re.U)
hanzi_re = re.compile(u"[\w]+", re.U)
PHRASE_MAX_LENGTH = 6


def str_decode(sentence):
    """转码"""
#    if not isinstance(sentence, unicode):
#     try:
#         sentence = sentence.decode('utf-8')
#     except UnicodeDecodeError:
#         sentence = sentence.decode('gbk', 'ignore')
    return sentence


def extract_hanzi(sentence):
    """提取汉字"""
    return hanzi_re.findall(sentence)


def cut_sentence(sentence):
    """把句子按照前后关系切分"""
    result = {}
    sentence_length = len(sentence)
    for i in range(sentence_length):
        for j in range(1, min(sentence_length - i+1, PHRASE_MAX_LENGTH + 1)):
            tmp = sentence[i: j + i]
            result[tmp] = result.get(tmp, 0) + 1
    return result

    
    
def gen_word_dict(path):
    """统计文档所有候选词，词频（包括单字）"""
    word_dict = {}
    with open(path,'r',encoding='gbk') as fp:
        for line in fp:
            utf_rdd = str_decode(line)
            hanzi_rdd = extract_hanzi(utf_rdd)   # list
            for words in hanzi_rdd:
                raw_phrase_rdd = cut_sentence(words)  # dict
                for word in raw_phrase_rdd:

                    if word in word_dict:
                        word_dict[word] += raw_phrase_rdd[word]
                    else:
                        word_dict[word] = raw_phrase_rdd[word]
    return word_dict   
    
def gen_lr_dict(word_dict,counts,thr_fq,thr_mtro):
    """统计长度>1的词的左右字出现的频数，并进行了频数和互信息筛选。
    # 得到词典：{'一个':[1208,2,8,1,15,...],'':[],...}其中[]第一元素为该字总的频数，其他元素为加上右或左边单个字后的频数
    """

    # def dict_iteritems(dict):
        # for w in dict:
            # yield (w, dict[w])

    # word_r_sort = sorted(dict_iteritems(word_dict), key=lambda x: x[0][:-1], reverse=False)
    #word_r_sort = sorted(word_dict.items(), key=lambda x: x[0][:-1], reverse=False)
    # print('dict内存:', sys.getsizeof(word_r_sort))

    l_dict = {}
    r_dict = {}
    k = 0
    for word in word_dict:
        k += 1
        if len(word) < 3: 
            continue
        wordl = word[:-1]
        ml = word_dict[wordl]
        if ml > thr_fq:  # 词频筛选
            wordl_r = wordl[1:]
            wordl_l = wordl[0]
            mul_info1 = ml * counts / (word_dict[wordl_r] * word_dict[wordl_l])
            wordl_r = wordl[-1]
            wordl_l = wordl[:-1]
            mul_info2 = ml * counts / (word_dict[wordl_r] * word_dict[wordl_l])
            mul_info = min(mul_info1, mul_info2)
            #print (wordl,mul_info)
            if mul_info > thr_mtro:  # 互信息筛选
                if wordl in l_dict:
                    l_dict[wordl].append(word_dict[word])
                else:
                    l_dict[wordl] = [ml, word_dict[word]]
           
            
        wordr = word[1:]
        mr = word_dict[wordr]
        if mr > thr_fq:  # 词频筛选
        
            wordr_r = wordr[1:]
            wordr_l = wordr[0]
            mul_info1 = mr * counts / (word_dict[wordr_r] * word_dict[wordr_l])
            wordr_r = wordr[-1]
            wordr_l = wordr[:-1]
            mul_info2 = mr * counts / (word_dict[wordr_r] * word_dict[wordr_l])
            mul_info = min(mul_info1, mul_info2)
            
            if mul_info > thr_mtro:  # 互信息筛选        
                if wordr in r_dict:
                    r_dict[wordr].append(word_dict[word])
                else:
                    r_dict[wordr] = [mr, word_dict[word]]   
        if k%1000000 == 0:
            print('---------------',k)
    return l_dict,r_dict
 
def cal_entro(r_dict):
    """计算左边熵或右边熵"""
    entro_r_dict = {}
    for word in r_dict:
        m_list = r_dict[word]

        r_list = m_list[1:]
        fm = m_list[0]

        entro_r = 0  # 右边熵
        krm = fm - sum(r_list)
        if krm > 0:
            entro_r -= 1 / fm * log(1 / fm, 2) * krm  # 右边为空时，应该增加熵

        for rm in r_list:
            entro_r -= rm / fm * log(rm / fm, 2)
        entro_r_dict[word] = entro_r
        
    return entro_r_dict
      
def entro_lr_fusion(entro_r_dict,entro_l_dict):      
    """左右熵合并"""
    entro_in_rl_dict = {}
    entro_in_r_dict = {}
    entro_in_l_dict =  entro_l_dict.copy()
    for word in entro_r_dict:
        if word in entro_l_dict:
            entro_in_rl_dict[word] = [entro_l_dict[word], entro_r_dict[word]]
            entro_in_l_dict.pop(word)
        else:
            entro_in_r_dict[word]  = entro_r_dict[word]
    return entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict
   
def entro_filter(entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict,thr_entro):
    """信息熵筛选"""
    entro_dict = {}
    l, r, rl = 0, 0, 0
    for word in entro_in_rl_dict:
        #time.sleep(0.4)
        if entro_in_rl_dict[word][0]>thr_entro and entro_in_rl_dict[word][1]>thr_entro:
            entro_dict[word] = word_dict[word]
            rl +=1
            #print (word, entro_in_rl_dict[word])

    for word in entro_in_l_dict:
        if entro_in_l_dict[word] > thr_entro:
            entro_dict[word] = word_dict[word]
            l += 1
            #print (word, entro_in_l_dict[word])

    for word in entro_in_r_dict:
        if entro_in_r_dict[word] > thr_entro:
            entro_dict[word] = word_dict[word]
            r += 1
            #print (word, entro_in_r_dict[word])

    print ('（信息熵筛选后）左右词数量：', rl, l, r)
    
    return entro_dict

    
def train_corpus_words(path):
    """读取语料文件，根据互信息、左右信息熵训练出语料词库"""
    thr_fq = 10  # 词频筛选阈值
    thr_mtro = 80  # 互信息筛选阈值
    thr_entro = 3  # 信息熵筛选阈值
    
    # 步骤1：统计文档所有候选词，词频（包括单字）
    st = time.time()
    word_dict = gen_word_dict(path)  
    et = time.time()
    print('读数耗时：',et-st)
    counts = sum(word_dict.values())  # 总词频数
    print ('总词频数：', counts,'候选词总数：',len(word_dict))
    # print('dict内存:', sys.getsizeof(word_dict))


    # 步骤2：统计长度>1的词的左右字出现的频数，并进行了频数和互信息筛选。  
    print('rl_dict is starting...')
    st = time.time()
    l_dict,r_dict = gen_lr_dict(word_dict,counts,thr_fq,thr_mtro)  # 右边存在单个字的词 的词典，值为右边字的统计（注意两个词典不一定相同，因为，右边不存在字的词不被记录）
    et = time.time()
    print ('互信息筛选耗时：',et-st)
    print( '（频数和互信息筛选后）左右候选词数量：', len(l_dict),len(r_dict))
    


    # 步骤3： 计算左右熵，得到词典：{'一个':5.37,'':,...}
    entro_r_dict = cal_entro(l_dict)  # 左边词词典 计算右边熵
    entro_l_dict = cal_entro(r_dict)  # 右边词词典 计算左边熵
    del l_dict,r_dict  # 释放内存


    # 步骤4：左右熵合并，词典：rl={'一个':[5.37,8.2],'':[左熵，右熵],...},r={'我说':5.37,'':右熵,...},l={'还行吧':3.37,'':左熵,...}
    entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict = entro_lr_fusion(entro_r_dict,entro_l_dict)
    print ('合并后存在左右熵词数量：(左右、左、右)', len(entro_in_rl_dict), len(entro_in_l_dict), len(entro_in_r_dict))
    del entro_r_dict,entro_l_dict

    # 步骤5： 信息熵筛选
    entro_dict = entro_filter(entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict,thr_entro)
    del entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict
    
    # 步骤6：输出最终满足的词，并按词频排序
    result = sorted(entro_dict.items(), key=lambda x:x[1], reverse=True)

    with open('userdict.txt', 'w',encoding='utf-8') as kf:
        for w, m in result:
            #print w, m
            kf.write(w + ' %d\n' % m)
 
    print ('\n词库训练完成！总耗时：')


if __name__ == "__main__":

    path = 'query_text.txt'
    print ('训练开始...')
    train_corpus_words(path)
    print ('training is ok !')
