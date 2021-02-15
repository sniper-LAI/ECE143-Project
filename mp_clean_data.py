import pandas as pd
import numpy as np
from datetime import date
import datetime

def num_days(x):
    """
    return number of days from x to today
    :param x: date
    :return:
    """
    date_today = date.today()
    x = x.split('-')
    old_date = date(int(x[0]), int(x[1]), int(x[2]))
    return (date_today - old_date).days


positive_adj = ['cozy', 'spacious', 'park', 'garden', 'comfort', 'comfy', 'nice', 'beautiful', 'luxury',
                'luxurious', 'pretty', 'cute', 'artistic', 'charm', 'huge', 'hippest', 'lovely', 'retreat', 'quiet',
                'bright', 'large', 'chill', 'relaxed', 'quaint', 'sweet', 'gorgeous', 'designer', 'calm', 'convenien',
                'peaceful', 'perfect', 'private', 'prestigious', 'neat', 'amazing', 'splendid', 'immaculate',
                'spectacular', 'great', 'elegant', 'relax', 'awesome', 'fantastic', 'stylish',
                'unique', 'modern', 'stunning', 'convenience', 'gigantic', 'magnificent',
                'extraordinary', 'new', 'adorable', 'delightful', 'renovated', 'lavish', 'excellent', 'clean', 'big',
                'best', 'affordable', 'breathtaking', 'radiant', 'holiday', 'cultured', 'premier', 'marvelous',
                'trendiest', 'sunny', 'picturesque', 'glamorous', 'secure']
shared_in_name = ['share', 'sharing']
def cleanData():
    """
    Read dataset and return a cleaned dataset
    :return:
    """
    data = pd.read_csv('/Users/krishvadodaria/Downloads/archive/AB_NYC_2019.csv')
    data['last_review'] = data['last_review'].fillna('1-1-1')
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0.0)
    data['name'] = data['name'].fillna('Unnamed')
    data['adj'] = 0
    data['shared_in_name'] = 0
    data['airport_in_name'] = 0
    data['len_name'] = 0
    for adj in positive_adj:
        data['adj'][data['name'].str.lower().str.contains(adj) == True] += 1
    for shared in shared_in_name:
        data['shared_in_name'][data['name'].str.lower().str.contains(shared)] = 1
    data['airport_in_name'][data['name'].str.lower().str.contains('airport')] = 1
    data['len_name'] = data['name'].str.split().str.len()
    neigh_groups_dict = dict(zip(data['neighbourhood_group'].unique(), range(len(data['neighbourhood_group'].unique()))))
    neigh_dict = dict(zip(data['neighbourhood'].unique(), range(len(data['neighbourhood'].unique()))))
    rtype_dict = dict(zip(data['room_type'].unique(), range(len(data['room_type'].unique()))))
    data['neighbourhood_group'] = data['neighbourhood_group'].apply(lambda x: neigh_groups_dict.get(x))
    data['neighbourhood'] = data['neighbourhood'].apply(lambda x: neigh_dict.get(x))
    data['room_type'] = data['room_type'].apply(lambda x: rtype_dict.get(x))
    data['last_review'] = data['last_review'].apply(num_days)
    data = data.drop('host_name', axis = 1)
    data = data.drop('name', axis = 1)
    return data











