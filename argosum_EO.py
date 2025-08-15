import os
import re

import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords


from largemodel import llm

stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

#
nlp.add_pipe("textrank")
def readtext(path):
    with open(path, encoding='utf-8') as file_obj:
        contents = file_obj.readlines()
    return contents
def remove_surrogate_pairs(input_string):
    # 使用正则表达式匹配代理对
    surrogate_pattern = re.compile('[\ud800-\udbff][\udc00-\udfff]')

    # 使用 sub 方法将代理对替换为空字符串
    cleaned_string = surrogate_pattern.sub(r'', input_string)

    return cleaned_string
def process_opinion(opinion_path):
    document_file_list = os.listdir(opinion_path)
    opinion_list = []
    for index, item in enumerate(document_file_list):
        temp_path = os.path.join(opinion_path, item)
        text_content = eval(readtext(temp_path)[0])
        opinion_list.append(text_content)
    return opinion_list
def preprocess(text):
    new_text = []
    text = remove_surrogate_pairs(text)
    for t in text.split(" "):
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = t.replace('URL_LINK', '')
        t = '' if t.startswith('#') and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)
def getData(dir):
    # 获取不同观点
    opinion_lists = process_opinion(dir)
    cluster=[]
    for item in opinion_lists:
        data = {}
        majority_opinion = ''
        minority_opinion = ''
        Documents=[]
        if 'tweets' in item.keys():
            Documents = [preprocess(tweet).strip() for tweet in item['tweets']]
            # Documents = [tweet for tweet in item['tweets']]

        if 'majority_opinion' in item.keys():
            majority_opinion = item['majority_opinion']
        if 'minority_opinions' in item.keys() :
            minority_opinion = item['minority_opinions']
        if 'mainority_opinions' in item.keys():
            minority_opinion = item['mainority_opinions']
        if 'main_story' in item.keys():
            main_story = item['main_story']

        data['Documents'] = Documents
        data['majority_opinion'] = majority_opinion
        data['minority_opinion'] = minority_opinion
        data['main_story'] = main_story
        cluster.append(data)
    return cluster
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = re.sub(r'\W+', ' ', text)
    # 分词
    tokens = word_tokenize(text)
    return tokens

def sup_opp_rule(answer, document, sentence):
    answer = answer.replace(document, '##########doc#######').replace(sentence, '#######sen#########')

    support_words = set(["support", "supports", "assist", "assists", "approve", "approves"])
    against_words = set(
        ["diverge","diverges","against", "againsts", "not support", "oppose", "opposes", "object to", "objects to", "contradict",
         "contradicts", "resist", "resists"])
    notclaarly_words = set(["not clearly", "unclear"])

    if any(keyword in answer.lower() for keyword in notclaarly_words):
        predicted_class = "not clearly"
    elif any(keyword in answer.lower() for keyword in against_words):
        predicted_class = "diverges"
    elif any(keyword in answer.lower() for keyword in support_words) and any(
            keyword in answer.lower() for keyword in against_words):
        predicted_class = "not clearly"
    else:
        predicted_class = "support"
    print('predicted_class', predicted_class)
    return predicted_class


def construct_sup_opp_set(client_socket_sum, Documents, sentence):

    sup_set = set()
    opp_set = set()
    neu_set = set()
    for index, document in enumerate(Documents):
        prompt = f"Determine whether document '{document}' supports, diverges or not clearly against with the sentence '{sentence}'. Answer using the word below: 'support', 'diverge' or 'not clearly'.  Return the result directly without interpretation."
        print('prompt: \n', prompt)
        answer =llm.getAnswer(client_socket_sum, query=prompt)

        answer = sup_opp_rule(answer, document, sentence)
        if answer == 'support':
            sup_set.add(index)
        elif answer == 'diverges':
            opp_set.add(index)
        elif answer == 'not clearly':
            neu_set.add(index)

    return sup_set, opp_set, neu_set


def get_evdience(client_socket_sum, opinions_list, documents):
    for opinion in opinions_list:
        sup_set, opp_set, neu_set = construct_sup_opp_set(client_socket_sum, documents, opinion)
    exit_code = '!!reset'  #
    llm.getAnswer(client_socket_sum, query=exit_code)
    return sup_set, opp_set, neu_set


def continue_writing_for_neu(other_documents, opinion, phrase):
    if len(other_documents) == 0:
        return opinion

    continue_writing_prompt = (
        f" Optionally update the original summary by selecting some key opinions from the following documents "
        f" that are semantically different from the original summary'. "
        f" The original summary is '{opinion}' "
        f" the documents are : {other_documents}'. "
        f" If you feel you don't need to update it, just output the original summary. "
        f" You should output the content of the original summary, rather than telling me that the original summary is good.")

    print('continue_writing_prompt', continue_writing_prompt)
    summary = llm.getAnswer(client_socket_sum, query=continue_writing_prompt)
    return summary


def continue_writing_for_sup(other_documents, major_opinion, phrase):
    if len(other_documents) == 0:
        return major_opinion
    continue_writing_prompt = (
        f" Summarize the opinions in the following documents that are differ from the majority opinion '{major_opinion}'. "
        f" Then generate a 30-word summary that contains diverse opinions: {other_documents}. "
        f" Continue writing with the following text: '{major_opinion}, however, others ...' ")
    print('continue_writing_prompt', continue_writing_prompt)
    summary = llm.getAnswer(client_socket_sum, query=continue_writing_prompt)


    return summary

def extract_after_contrast(text):
    import re

    contrast_words = ["but", "however"]
    text_lower = text.lower()

    positions = [(word, text_lower.find(word)) for word in contrast_words if word in text_lower]
    if not positions:
        return None
    contrast_word, pos = min(positions, key=lambda x: x[1])
    pattern = re.compile(rf'\b{contrast_word}\b', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text[match.end():].strip()  #

    return None


if __name__ == "__main__":
    modelname = ""

    client_socket_sum= modelname
    testfile_dir = '../Data/MOSdata/MOS_corpus_hydrated/testing/opi'
    cluster = getData(testfile_dir)

    topic_obj = open('../Data/MOSdata/llmhints/topic.txt', 'r', encoding="utf-8")
    topic_content = topic_obj.readlines()

    tag_dir = '../Data/MOSdata/targat_includ_mainstory2.txt'
    write_flag = 1
    flag = ""
    try:
        for index, instance in enumerate(cluster):
            final_results={}

            if  index>=0 :
                print(f"\n{index + 1}.")

                documents = instance['Documents']
                final_results["documents"] = documents

                numbered_list = [f"{i + 1}. {item}" for i, item in enumerate(documents)]

                topic = topic_content[index].replace('\n', '')

                final_results["topic"] = topic

                major_prompt=  f" For the topic '{topic}', summarize the following documents to generate a  majority opinion : {numbered_list}.  Return the  answer  by the format: The majority think that ..."

                major_opinion = llm.getAnswer(client_socket_sum, query=major_prompt)

                final_results["major_opinion"] = major_opinion
                maj = re.sub(r"(?i)^the majority think\s*", "", major_opinion).strip()
                print('maj: \n', maj)
                # opinions_list=[]
                # opinions_list.append(major_opinion)
                sup_set, opp_set, neu_set = get_evdience(client_socket_sum, [maj], documents)

                print('sup set \n', sup_set)
                print('opp set \n', opp_set)
                print('neu set \n', neu_set)

                phrase = []
                summaries_final_conclusion = ''
                summaries_by_opp = ''
                summaries_by_opp_result = []
                if float(len(sup_set) / len(documents)) <= 0.9:
                    sup_list = [documents[i] for i in sup_set]
                    opp_list = [documents[i] for i in opp_set]
                    neu_list = [documents[i] for i in neu_set]

                    final_results["sup_list"] = sup_list
                    final_results["opp_list"] = opp_list
                    final_results["neu_list"] = neu_list

                    other_documents = opp_list
                    opinion = major_opinion
                    summaries_by_opp = continue_writing_for_sup(other_documents, opinion, phrase)

                    divergent_opinion = extract_after_contrast(summaries_by_opp)

                    final_results["divergent_opinion"] = divergent_opinion
                    print('summaries_by_opp \n', summaries_by_opp)

                    other_documents = neu_list
                    opinion = summaries_by_opp
                    summaries_final = continue_writing_for_neu(other_documents, opinion, phrase)
                    final_results["summaries_final"] = summaries_final

                else:
                    summaries_final = major_opinion

                with open('summary_result/' + str(flag) + 'summary_results' + '.txt', 'a+', encoding='utf-8') as file:
                    file.write(str(final_results).replace('\n', '') + '\n')


    except Exception as ex:

        import traceback
        traceback.print_exc()