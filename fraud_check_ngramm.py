import numpy as np
import regex as re
from nltk import ngrams
from collections import Counter
import regex as re
import copy
import json

def checkWord(word,dict_mark_right,dict_mark_left):
    retVal_ = []
    threshold = 0.0274352711892821
    penalty_1 = 1.12
    penalty_2 = 1.05
    most_errors_over_300 = ['ап','ар','ва','ен','ол','ор','па','по','пр','ра','ро']
    most_errors_less_300 = ['ав','ал','ам','ан','ек','ер','ит','ке','ла','ло','ми','не','нк','но','он','оп','ре','ри','рп','та','то']   
    if len(word) > 3:
        for i in range(1,len(word)-2):
            right_score = 0
            left_score = 0
            two = word[i:i+2]
            right = word[i+2]
            left = word[i-1]
            if two in dict_mark_left.keys() and left in dict_mark_left[two].keys():
                left_score = dict_mark_left[two][left]
            if two in dict_mark_right.keys() and right in dict_mark_right[two].keys():
                right_score = dict_mark_right[two][right]
            # score = (left_score*right_score)**(0.5)
            if two in most_errors_over_300:
                left_score = left_score**(penalty_1**2) 
            elif two in most_errors_less_300:
                left_score = left_score**(penalty_2**2)  
                
            retVal_.append((left_score*right_score)**(0.5))
            
    elif len(word) == 3:
        # print(i)
        right_score = 0
        left_score = 0
        two_right = word[0:2]
        two_left = word[1:3]
        right = word[2]
        left = word[0]
        if two_left in dict_mark_left.keys() and left in dict_mark_left[two_left].keys():
            left_score = dict_mark_left[two_left][left]
        if two_right in dict_mark_right.keys() and right in dict_mark_right[two_right].keys():
            right_score = dict_mark_right[two_right][right]
        if two_left in most_errors_over_300:
            left_score = left_score**penalty_1
        elif two_left in most_errors_less_300:
            left_score = left_score**penalty_2
        if two_right in most_errors_over_300:
            right_score = right_score**penalty_1
        elif two_right in most_errors_less_300:
            right_score = right_score**penalty_2
            
        retVal_.append((left_score*right_score)**(0.5))
        
    else:
        return threshold
    return np.mean(retVal_)

def checkWord3gramm(word,dict_mark,threshold):
    retVal_ = []
    most_errors_less_100 = ['авп','апо','ара','вар','дло','енг','кен','кер','оло','опр','оро','пол','пор','рар','рва','рке','роп','рор','рпо','рпр']
    most_errors_over_100 = ['фыв','апр','вап','орп','про','рол','рпа']

    if len(word) >= 3:
        for i in range(len(word)-2):
            curr = word[i:i+3]
            try:
                retVal_.append(dict_mark[curr])
            except:
                retVal_.append(threshold)
            if curr in most_errors_less_100:
                retVal_[i] = retVal_[i]**0.6
            elif curr in most_errors_over_100:
                retVal_[i] = retVal_[i]**0.5
    else:
        return threshold

    return np.median(retVal_)

temp = "123 123"
"".join(temp.split(" ")).isdigit()

def myPredict(dict_mark_right,dict_mark_left,word_list,debug=False):
    threshold = 0.0274352711892821
    mean_not_fraud = 0.14530785117709769
    sogl = ['б','в','г','д','ж','з','й','к','л','м','н','п','р','с','т','ф','х','ц','ч','ш','щ']
    gl = ['у','е','ы','а','о','э','я','и','ю']  
    one_length_word = ['а','у','к','о','я','с','и']
    two_length_word = ['из','вы','бы','гб','да','до','fm','фм','ok','ок','4g','5g','3g',
                    'еж','за','их','кг','ко','мы','шт','кг','mb','мб','хз',
                    'на','не','но','ну','об','он','мб',
                    'от','по','ту','те','ты','та','то',
                    'уж','ух','уф','ус','че','шо','тв','tv','тц',
                    'ща','юг','яд','хз','ру','fm','ne','da','vk','вк','рф']  

    full_str = " ".join(word_list)
    score_list = []
    sites = [".ru",".com",".net",".ua",".ру",".ком",".нет",".уа",",ru",",com",'.md','.мд','.io']
    if True in [w in full_str for w in sites]:
        return 1

    isdigit_flag = True
    for i,w in enumerate(word_list):
        word_list[i] = w.replace("ё","е")
        if isdigit_flag and len(re.findall(r'[a-zA-Zа-яА-Я]+',word_list[i])) > 0:
            isdigit_flag = False
    if isdigit_flag:
        return threshold


    
    for i,w in enumerate(word_list):
        while "." in word_list[i]:
            word_list[i] = word_list[i].replace(".","")

    for i,x in enumerate(word_list):
        score_list.append(checkWord(x,dict_mark_right,dict_mark_left))
        if word_list[i].isdigit():
            score_list[i] = mean_not_fraud
    for i,s in enumerate(score_list):
        if len(word_list[i]) == 1:
            if (word_list[i] in one_length_word):
                score_list[i] = mean_not_fraud
            else:
                score_list[i] = threshold
        if len(word_list[i]) == 2:
            if (word_list[i] in two_length_word):
                score_list[i] = mean_not_fraud
            else:
                score_list[i] = threshold    
        counter_sogl = 0
        counter_gl = 0
        for alpha in word_list[i]:
            if alpha in sogl:
                counter_sogl +=1
                counter_gl = 0
            if alpha in gl:
                counter_gl +=1
                counter_sogl = 0
            if counter_sogl >= 4 or counter_gl >= 4:
                # штрафуем если много согласных или гласных подряд
                score_list[i] = (score_list[i])**1.1
 
    for i,w in enumerate(word_list):
        repeatGrams = checkRepeatGrams(word_list[i])
        if repeatGrams == 2:
            score_list[i] = score_list[i]**(1.1)  
        elif repeatGrams == 3:
            score_list[i] = score_list[i]**(1.5)     
        elif repeatGrams == 4:
            score_list[i] = score_list[i]**(2)
        elif repeatGrams > 4:
            score_list[i] = score_list[i]**(3)           
    if debug:
        print(score_list)
    if len(score_list) == 0:
          return 0
    return np.mean(score_list)

def myPredict_2(dict_mark,word_list,debug=False):
    threshold = 4.094391232437392
    mean_not_fraud = 8.180009825063703
    sogl = ['б','в','г','д','ж','з','й','к','л','м','н','п','р','с','т','ф','х','ц','ч','ш','щ']
    gl = ['у','е','ы','а','о','э','я','и','ю']  
    one_length_word = ['а','у','к','о','я','с','и']
    two_length_word = ['из','вы','бы','гб','да','до','fm','фм','ok','ок','4g','5g','3g',
                    'еж','за','их','кг','ко','мы','шт','кг','mb','мб','хз',
                    'на','не','но','ну','об','он','мб',
                    'от','по','ту','те','ты','та','то',
                    'уж','ух','уф','ус','че','шо','тв','tv','тц',
                    'ща','юг','яд','хз','ру','fm','ne','da','vk','вк','рф']  

    full_str = " ".join(word_list)
    score_list = []
    sites = [".ru",".com",".net",".ua",".ру",".ком",".нет",".уа",",ru",",com",'.md','.мд','.io']
    if True in [w in full_str for w in sites]:
        return mean_not_fraud

    isdigit_flag = True
    for i,w in enumerate(word_list):
        word_list[i] = w.replace("ё","е")
        if isdigit_flag and len(re.findall(r'[a-zA-Zа-яА-Я]+',word_list[i])) > 0:
            isdigit_flag = False
    if isdigit_flag:
        return threshold




    
    for i,w in enumerate(word_list):
        while "." in word_list[i]:
            word_list[i] = word_list[i].replace(".","")

    for i,x in enumerate(word_list):
        score_list.append(checkWord3gramm(x,dict_mark,threshold))
        if word_list[i].isdigit():
            score_list[i] = mean_not_fraud

    for i,s in enumerate(score_list):
        if len(word_list[i]) == 1:
            if (word_list[i] in one_length_word):
                score_list[i] = mean_not_fraud
            else:
                score_list[i] = threshold
        if len(word_list[i]) == 2:
            if (word_list[i] in two_length_word):
                score_list[i] = mean_not_fraud
            else:
                score_list[i] = threshold    
        counter_sogl = 0
        counter_gl = 0
        for alpha in word_list[i]:
            if alpha in sogl:
                counter_sogl +=1
                counter_gl = 0
            if alpha in gl:
                counter_gl +=1
                counter_sogl = 0
            if counter_sogl >= 5 or counter_gl >= 5:
                # штрафуем если много согласных или гласных подряд
                score_list[i] = (score_list[i])**0.86
  
    for i,w in enumerate(word_list):
        repeatGrams = checkRepeatGrams(w)
        if repeatGrams == 2:
            score_list[i] = score_list[i]**(0.85)  
        elif repeatGrams == 3:
            score_list[i] = score_list[i]**(0.7)  
        elif repeatGrams == 4:
            score_list[i] = score_list[i]**(0.6)                          
        elif repeatGrams > 4:
            score_list[i] = score_list[i]**(0.45)  
    if debug:
        print(score_list)
    if len(score_list) == 0:
          return 0
    return np.mean(score_list)

def checkRepeatGrams(text):
    # находим повторяющиеся элементы типа: равраврав рарарара равпоканцрраврав
    # в будущем надо научиться искать типа: фыввыфроллор и тд
    try:
        text_clean = text
        grams2 = list(ngrams(text_clean, 2))  
        grams3 = list(ngrams(text_clean, 3))
        count2 = Counter(grams2).values()  
        count3 = Counter(grams3).values()
        retVal = max(max(count2),max(count3))
    except:
        retVal = 0
    return retVal

def myPredict_total(dict_mark_right,dict_mark_left,dict_mark,text,return_weight=True):
    threshold = 0.2
    word_list = list(re.findall(r'[ёa-zA-Zа-яА-Я0-9.]+',text))
    score_1 = myPredict(dict_mark_right,dict_mark_left,word_list)
    score_2 = myPredict_2(dict_mark,word_list)
    if return_weight:
        return score_1*score_2
    return 1 if score_1*score_2 < threshold else 0
    
def main():
    PATH = "/content/drive/MyDrive/tiburon/dicts/"
    with open(PATH + 'retVal_1_norm.json') as json_file:
        dict_1 = json.load(json_file) # отнормированные вероятности
    with open(PATH + 'retVal_2_norm.json') as json_file:
        dict_2 = json.load(json_file) # отнормированные вероятности
    with open(PATH + 'retVal_3_log.json') as json_file:
        dict_3 = json.load(json_file) # логирифмированный словарь
    
    myPredict_total(dict_1,dict_2,dict_3,"кто убил марка",False)

if __name__ == "__main__":
    main()