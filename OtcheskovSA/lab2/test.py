import nltk
import re
import numpy as np 
  
import heapq

# execute the text here as : 
text = """Библиотека NLTK, или NLTK, — пакет библиотек и программ для символьной и статистической обработки естественного языка,
написанных на языке программирования Python. Содержит графические представления и примеры данных.
Сопровождается обширной документацией, включая книгу с объяснением основных концепций, стоящих за теми задачами обработки естественного языка,
которые можно выполнять с помощью данного пакета. хорошо подходит для студентов, изучающих компьютерную лингвистику или близкие предметы,
такие как эмпирическая лингвистика, когнитивистика, искусственный интеллект, информационный поиск и машинное обучение.
NLTK с успехом используется в качестве учебного пособия, в качестве инструмента индивидуального обучения и в качестве платформы для прототипирования и создания научно-исследовательских систем.
NLTK является свободным программным обеспечением.
Проект возглавляет Стивен Бёрд.""" 
dataset = nltk.sent_tokenize(text) 
for i in range(len(dataset)): 
    dataset[i] = dataset[i].lower() 
    dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
    dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
# Creating the Bag of Words model 
word2count = {} 
for data in dataset: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
freq_words = heapq.nlargest(100, word2count, key=word2count.get)
X = [] 
for data in dataset: 
    vector = [] 
    for word in freq_words: 
        if word in nltk.word_tokenize(data): 
            vector.append(1) 
        else: 
            vector.append(0) 
    X.append(vector) 
X = np.asarray(X) 
print(X)