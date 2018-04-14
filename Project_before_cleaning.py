# !/usr/bin/python

import pymysql
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

def open_conn():
    """open the connection before each test case"""
    conn = pymysql.connect(user='root', password='202020',
                                   host='localhost',
                                   database='yelp_db')

    return conn

def close_conn(conn):
    """close the connection after each test case"""
    conn.close()
    
def executeQuery(conn, query, commit=False):
    """ fetch result after query"""
    cursor = conn.cursor()
    query_num = query.count(";")
    if query_num > 1:
        for result in cursor.execute(query, params=None, multi=True):
            if result.with_rows:
                result = result.fetchall()
    else:
        cursor.execute(query)
        result = cursor.fetchall()
    # we commit the results only if we want the updates to the database
    # to persist.
    if commit:
        conn.commit()
    else:
        conn.rollback()
    # close the cursor used to execute the query
    cursor.close()
    return result

# ======================================Question 1==============================================
def rating_in_time_series():
    #fetch results from the database，返回的是tuple类型
    sql = "select date, stars from review where business_id='4JNXUYY8wbaaDmk3BPzlWw' order by date;"
    result = executeQuery(conn, sql)
    #retreive results as a list from the list of tuples, list(group) 出来的是[(date1, rating1),(date1, rating2)]
    result_list = [list(group) for key, group in groupby(result, key=lambda x:x[0])]
    #获取分类的时间，作为横坐标
    date_list = [key for key, group in groupby(result, key=lambda x:x[0])]
    rating_list = []
    #获取每天的 rating集合，如[[2,3,5], [3,3,3]]
    for group in result_list:
        total = []
        for ele in group:
            total.append(ele[1])
        rating_list.append(total)
    avg_rating, cnt = [0], 0
    #获取到某个时间点为止的 avg,生成一个Long term的 avg_list
    for ratings in rating_list:
        sum_uptonow = cnt*avg_rating[-1] 
        cnt += len(ratings)
        sum_after = sum(ratings) + sum_uptonow
        avg_now = float(sum_after/cnt)
        avg_rating.append(avg_now)
    avg_rating = avg_rating[1:]
    #plot results
    #横坐标是所有时间，纵坐标是到该时间点的时候的平均值
    x = date_list
    y = avg_rating
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Rating_in_time_sequence')
    plt.xlabel('Day after the first review')
    plt.ylabel('Average Rating Up to Now')
    ax1.set_ylim(ymin=0, ymax=5)
    ax1.plot(x, y)
    plt.savefig('Rating_in_time_sequence.png')
    return 


# =====================================Question 2===============================================
def Get_closest(num, ever):
    index, difference = 0, 5
    for i in range(len(ever)):
        dif = abs(float(num)-float(ever[i]))
        if dif <= difference:
            difference = dif
            index = i
    return ever[index]

def Rating_Based_Past():
    '''用过去的评价来预测将来的评价，要用到support以及confidence相关的知识，不过要把商店里的avg保留1位小数？然后由avg->这个人的评价，condifence水平，多大概率会给什么评价'''
    #sql = "select business_id, avg(stars) from review where business_id in (select distinct business_id from review where user_id = '---1lKK3aKOuomHnwAkAow') group by business_id;"
    confidence = 0.4
    
    sql = "select ROUND(convert(F1.stars,decimal),1), ROUND(F2.Avg,1) from (select business_id, stars from review where user_id='---1lKK3aKOuomHnwAkAow') as F1 natural join (select business_id, avg(stars) as Avg from review where business_id in (select distinct business_id from review where user_id = '---1lKK3aKOuomHnwAkAow') group by business_id) as F2;"
    result = executeQuery(conn, sql)
    points = np.array(result)
    #print(points)
    
    #count the frequency of every tuple (user, average) and total number of average in the form
    #{4.5:{3: 1, 4: 2, 5: 1, subtotal:4}} means 1 (4.5, 3) 2(4.5, 4) 1(4.5, 5) 4 in total
    dic_avg = dict()
    cnt = 0
    for user, average in points:
        sub = dic_avg.setdefault(str(average), dict())
        sub[user] = sub.setdefault(user, [0])
        sub[user][-1] += 1
        dic_avg[str(average)][user] = sub[user]
        dic_avg[str(average)]['sub_total'] = dic_avg[str(average)].setdefault('sub_total',0 ) + 1 
        cnt += 1
    avg_ever_been_to = list(dic_avg.keys())
    
    # find out tuples whose confidence level is higher than pre-set confidence, thus udner this confidence level we have
    # average -> user rating  
    for avg_rat in dic_avg:
        for user_rating in dic_avg[avg_rat]:
            if user_rating != 'sub_total':
                confidence_level = float(dic_avg[avg_rat][user_rating][-1])/float(dic_avg[avg_rat]['sub_total'])
                dic_avg[avg_rat][user_rating].append(confidence_level)
                print('Average Rating: %s    User Rating: %s   Frequency: %d   Support Level: %f'%(avg_rat, str(user_rating), dic_avg[avg_rat][user_rating][0], dic_avg[avg_rat][user_rating][1]))
    while True:
        average_rating = str(input('Given the average rating of this restaurant:'))
        average_rating = Get_closest(average_rating, avg_ever_been_to)
        print('According to the record of the user')
        find = False
        for user_rating in dic_avg[average_rating]:
            if user_rating != 'sub_total':
                confidence_level = dic_avg[average_rating][user_rating][-1]
                if confidence_level >= confidence:
                    find = True
                    print('The user might rate the business as: %d with the probability of %d %%'%(user_rating, confidence_level*100))
        if not find:
            print('No prediction could be made under this confidence level: %d'%(confidence))
# =============================================================================
#     fig = plt.figure()
#     compare_fig = fig.add_subplot(111)
#     compare_fig.set_title('Distribution Of Rating')
#     plt.ylabel('Rating of Users')
#     plt.xlabel('Average Rating of Others')
#     compare_fig.set_ylim(ymin=0, ymax=6)
#     y_pred = KMeans(n_clusters=5, random_state=170).fit_predict(points)
#     compare_fig.scatter(points[:,1], points[:,0], c=y_pred)
#     compare_fig.set_ylim(ymin=0, ymax=6)
#     plt.show()
# =============================================================================
# =============================================================================
#     model = LinearRegression()
#     model.fit(np.reshape(points[:,1],[len(points),1]), np.reshape(points[:,0],[len(points),1]))
#     test = list()
#     for i in range(500):
#         z = float(i)/float(100)
#         test.append(z)
#     print(test)
#     predict_test = model.predict(np.reshape(test,[len(test),1]))
#     kk = model.coef_[0][0]
#     bb = model.intercept_[0] 
#     print(kk, bb)
#     compare_fig.set_ylim(ymin=0, ymax=6)
#     compare_fig.scatter(points[:,1], points[:,0], c=y_pred)
#     compare_fig.scatter(test, predict_test,c='red', s=5)
#     plt.show()
# =============================================================================
    
    return

# =================================Question 3===================================================
def get_percentage(x_labels, ufc_labels):
    '''Computes the percentage of getting at least one response of the length range'''
    percentage = dict()
    for index in range(len(x_labels)):
        percentage[x_labels[index]] = percentage.setdefault(x_labels[index], [0, 0])
        percentage[x_labels[index]][0] += 1
        if ufc_labels[index] > 0:
            percentage[x_labels[index]][1] += 1
    x, y= [], []
    for key in percentage:
        x.append(key)
        y.append(float(percentage[key][1])/float(percentage[key][0]))
    return percentage, x, y

def get_correlation_rate(x, y, response, compare='Text'):
    '''Return the correlation of two variables'''
    print('The currelation between %s and %s Length is %f, and the confidence level is %f' %(response, compare, stats.pearsonr(x, y)[0], stats.pearsonr(x, y)[1]))
    return

def length_and_useful():
    '''评论长度与获得的useful的关系，可以是由粗糙到具体，先是（长度，获得useful的数量）的correlation'''
    '''但这种关系比较粗糙，看不出什么东西来，再往里去一点，将长度进行分类，比如<100, 100-200, 200-300这样，这里可以用到K-Means分类算法'''
    '''然后不计算useful的总数, 而是计算至少有一个useful的总数，证明有人关注了，'''
    #fetch results from the database，返回的是tuple类型
    sql = "select text, useful, funny, cool from review limit 1000000;"
    result = executeQuery(conn, sql)
    length_range = 50
    
    #retreive results as a list from the list of tuples, list(group) 出来的是[(date1, rating1),(date1, rating2)]
    x_labels = np.array([len(sub[0].split())/length_range for sub in result])
    #获取text的长度，作为横坐标，每50个单词为一类， 0-50， 50-100， 100-150这样
    useful_labels = np.array([sub[1] for sub in result])
    funny_labels = np.array([sub[2] for sub in result])
    cool_labels = np.array([sub[3] for sub in result])
    #用字典获取每个长度区间获得赞的概率
    per_get_useful, x1, y1 = get_percentage(x_labels, useful_labels)
    per_get_funny , x2, y2 = get_percentage(x_labels,  funny_labels)
    per_get_cool  , x3, y3 = get_percentage(x_labels,   cool_labels)
    
    get_correlation_rate(x1, y1, 'Useful')
    get_correlation_rate(x2, y2, 'Funny')
    get_correlation_rate(x3, y3, 'Cool')
    #print(per_get_useful)
    plt.figure(figsize=(20,10))
    l1, = plt.plot(x1, y1, color='b')
    l2, = plt.plot(x2, y2, color='y')
    l3, = plt.plot(x3, y3, color='g')
    #l2, = plt.plot(list(range(len(test_y))), test_y,  color='r')
    # 设置坐标轴的lable
    plt.xlabel('Text length range')
    plt.ylabel('Possibility to get related feedback')
    # 设置y坐标轴刻度及标签, $$是设置字体
    # 设置legend
    plt.legend(handles = [l1, l2, l3,], labels = ['Useful', 'Funny', 'Cool'], loc = 'best')
    plt.show()
    return

# ====================================================================================

def Clustering():
    '''用来对数据点进行归类，unsupervisored learning，目前还没想好用来干啥'''
    #fetch results from the database，返回的是tuple类型
    sql = "select text, useful from review limit 100000;"
    result = executeQuery(conn, sql)
    #retreive results as a list from the list of tuples, list(group) 出来的是[(date1, rating1),(date1, rating2)]
    x_labels = np.array([len(text.split()) for text, useful in result])
    #获取text的长度，作为横坐标
    y_labels = np.array([useful for text, useful in result])
    points = np.c_[x_labels, y_labels]
    print(points)
    
    #clustering
    y_pred = KMeans(n_clusters=20, random_state=170).fit_predict(points)
    fig = plt.figure()
    clustering_fig = fig.add_subplot(111)
    clustering_fig.set_title('Clustering based on text length')
    plt.xlabel('Text_length')
    plt.ylabel('Useful')
    clustering_fig.set_ylim(ymin=0, ymax=50)
    clustering_fig.scatter(points[:,0], points[:,1], c=y_pred, s=1)
    plt.show()
    return 

# =================================Question 4===================================================

def Get_duration(day_time):
    '''获取时间Mon|hh:mm-hh:mm计算并返回多少个小时，分钟用小数表示'''
    opt, endt = day_time.split('-')
    op_hour, op_min = opt.split(':')
    end_hour, end_min = endt.split(':')
    hours = int(end_hour)-int(op_hour)
    minutes = int(end_min)-int(op_min)
    if hours < 0:
        hours += 24
    if minutes < 0:
        hours -= 1
        minutes += 60
    return hours + float(float(minutes)/60)

def Get_Id_Hours_Stars(ids, hours, ratings):
    '''将传入的businessID, Hours营业时间， Ratings评价用一个字典归好类'''
    hour_rating = dict()
    for index in range(len(ids)):
        if ids[index] not in hour_rating:
            hour_rating[ids[index]] = hour_rating.setdefault(ids[index], dict())
            hour_rating[ids[index]]['Avg_Stars'] = ratings[index]
            hour_rating[ids[index]]['Operating_hours'] = 0
        hour_rating[ids[index]]['Operating_hours'] += hours[index]
    return hour_rating

def Hours_and_Rating():
    '''营业时长Operating hours与获得的Rating的关系'''
    sql = "select hours, business_id, stars from hours H inner join business B on H.business_id = B.id limit 100;"
    #现在要把它拆分再一一对应
    result = executeQuery(conn, sql)
    #print(result)
    id_labels, hours_labels, rating_labels = list(), list(), list()
    for row in result:
        id_labels.append(row[1])
        rating_labels.append(row[2])
        duration = Get_duration(row[0].split('|')[1])
        hours_labels.append(duration)
    
    hour_rating = Get_Id_Hours_Stars(id_labels, hours_labels, rating_labels)
    
    points = list()
    for business_id in hour_rating.values():
        points.append((business_id['Operating_hours'], business_id['Avg_Stars']))
    points = np.array(points)
    operating_Hours, Stars = points[:,0], points[:,1]
    get_correlation_rate(operating_Hours, Stars, 'Operating Hours', compare='Average_Stars')
# =============================================================================
#     y_labels = np.array([useful for text, useful in result])
#     #a = np.array([1,2,3,4,5,1,3,2])
#     #b = np.array([5,4,3,2,1,5,3,4])
#     print(stats.pearsonr(x_labels, y_labels))
#     #df = pd.DataFrame()
#     #df['a'] = x_labels
#     #df['b'] = y_labels
#     #print(df.corr())
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.set_title('Text_length and Useful')
#     plt.xlabel('Text_length')
#     plt.ylabel('Useful')
#     svalue = 1
#     ax1.set_ylim(ymin=0, ymax=50)
#     ax1.scatter(x_labels, y_labels, s=svalue, c='g', marker='o')
#     plt.show()
# =============================================================================
    return

if __name__ == '__main__':
    #open connection to the database
    conn = open_conn()
    #rating_in_time_series()
    #length_and_useful()
    #Rating_Based_Past()
    #Clustering()
    Hours_and_Rating()
    close_conn(conn)
