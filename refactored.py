import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from pylab import *
import geopandas as gpd
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from folium import plugins

from wordcloud import WordCloud,STOPWORDS
import plotly.express as px

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

# Image save path
import os
save_path = "save/visualization/"
# Create save folder
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Set sns themec
# plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Create a pandas dataframe of the Airbnb data
data = pd.read_csv('AB_NYC_2019.csv')
data.head(5)

def remove_extreme(df,attribute,lower=0.25,upper=0.75):
    """
    This function is to remove the extreme values of the specific attributes for better visualization
    :df: input dataframe
    :attribute: the attribute of the input dataframe 
    :lower: the lower percentage limit
    :upper: the upper percentage limit
    """
    
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(attribute,str)
    assert attribute in df.dtypes
    assert lower>=0 and lower<=1
    assert upper>=0 and upper<=1
    assert lower<=upper

    Q1 = df[attribute].quantile(lower)
    Q3 = df[attribute].quantile(upper)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[attribute] > lower_bound) & (df[attribute]< upper_bound)]
    return df_filtered[attribute]

def numerical_plot(df,attributes,plot_type,saved_plot):
    """
    This function is to draw six subplots for numerical attributes
    The subplots can be distplot or violinplot
    :df: input dataframe
    :attributes: the attribute list
    :plot_type: define the output plot type
    :saved_plot: the file name of the saved plot
    """
    
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(attributes,list)
    assert isinstance(plot_type,str)
    assert isinstance(saved_plot,str)
    assert plot_type in ["distplot","violinplot"]

    for attribute in attributes:
        assert attribute in df.dtypes

    sns.set_palette("muted")
    f, ax = plt.subplots(figsize=(15, 6))
    counter = 1
    for attribute in attributes:
        filtered_result = remove_extreme(df, attribute)
        subplot(2,3,counter)
        if (plot_type=="distplot"):
            sns.distplot(filtered_result)
        elif (plot_type=="violinplot"):
            sns.violinplot(y=filtered_result)
        counter += 1
    plt.tight_layout() # avoid overlap of plots
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)

# attributes = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'availability_365']
# numerical_plot(data,attributes,"distplot","plot_numerical_distribution_1.png")
# numerical_plot(data,attributes,"violinplot","plot_numerical_distribution_2.png")

def heatmap_plot(df,drop_attributes, saved_plot):
    """
    This function is to draw heatmap for the input dataframe
    :df: input dataframe
    :drop_attributes: the  list contains attributes which will not be included in the heatmap
    :saved_plot: the file name of the saved plot
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(drop_attributes,list)
    assert isinstance(saved_plot,str)
    for attribute in drop_attributes:
        assert attribute in df.dtypes

    plt.figure(figsize=(20,10))
    sns.heatmap(data.drop(drop_attributes, axis = 1 ).corr(), square=True, cmap='Blues')
    plt.title('Correlation matrix of numerical variables')
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)

# drop_attributes = ['id','host_id','latitude','longitude'] 
# heatmap_plot(data,drop_attributes,"plot_correlation_matrix.png")

def NYC_Map(file_path):
    """
    This function is to draw map for new york district 
    :file_path: input 
    """
    assert isinstance(file_path,str)
    os.path.exists(file_path)
    os.path.isfile(file_path)
    plt.figure(figsize=(10,10))
    img = plt.imread(file_path,0)
    plt.imshow(img)
    plt.axis('off')
    
# NYC_Map("/content/New_York_City_.png")

def geographical_distribution_colormap(df,attribute,file_path,saved_plot):
    """
    This function is to draw colormap for the specific attributes and show its geographical distribution
    :df: input dataframe
    :attribute: the attribute which will be plotted
    :file_path: input 
    :saved_plot: the file name of the saved plot
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(attribute,str)
    assert attribute in df.dtypes
    assert isinstance(saved_plot,str)

    assert isinstance(file_path,str)
    os.path.exists(file_path)
    os.path.isfile(file_path)

    # plt.figure(figsize=(10,10))
    img = plt.imread(file_path,0)
    plt.imshow(img, zorder=0, extent=[-74.258, -73.7, 40.49,40.92])
    ax=plt.gca()
    df.plot(kind='scatter', x='longitude', y='latitude', c=attribute, ax=ax, 
              cmap=plt.get_cmap('rainbow'), colorbar=True, alpha=0.4, zorder=5)
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)
    plt.show()


# geographical_distribution_colormap(data[data.price < 500],'price',"/content/New_York_City_.png","plot_heatmap_1.png")
# geographical_distribution_colormap(data[data.price < 500],'availability_365',"/content/New_York_City_.png","plot_heatmap_2.png")

def geographical_distribution_scatterplot(df,attribute,file_path,saved_plot):
    """
    This function is to draw scatterplot for the specific attributes and show its geographical distribution
    :df: input dataframe
    :attribute: the attribute which will be plotted
    :file_path: input 
    :saved_plot: the file name of the saved plot
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(attribute,str)
    assert attribute in df.dtypes
    assert isinstance(saved_plot,str)

    assert isinstance(file_path,str)
    os.path.exists(file_path)
    os.path.isfile(file_path)

    img = plt.imread(file_path,0)
    plt.imshow(img, zorder=0, extent=[-74.258, -73.7, 40.49,40.92])
    sns.scatterplot(x="longitude", y="latitude",hue=attribute, data=df)
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)

# geographical_distribution_scatterplot(data,"neighbourhood_group","/content/New_York_City_.png","plot_scatterplot_1.png")
# geographical_distribution_scatterplot(data,"room_type","/content/New_York_City_.png","plot_scatterplot_2.png")

def neighbourhood_group_hosts_analysis(df, saved_plot):
    """
    This function is to draw a pie chart and count plot to analyze the hosts distribution among five neighborhood group
    :df: input dataframe
    :saved_plot: the file name of the saved plot
    """

    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)

    f,ax = plt.subplots(1, 2, figsize=(15,6))
    df['neighbourhood_group'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05,0.05],
                                                        autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Neighborhood Group Pie Chart')
    # ax[0].set_ylabel('Neighborhood Group Share')

    sns.countplot('neighbourhood_group',data=df,ax=ax[1],order=df['neighbourhood_group'].value_counts().index)
    ax[1].set_title('Neighborhood Group Histogram')
    plt.tight_layout() # avoid overlap of plots
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)

# neighbourhood_group_hosts_analysis(data,"plot_count_neighbourhood_group.png")

def neighbourhood_group_price_analysis(df, saved_plot):
    """
    This function is to draw a pie chart and count plot to analyze the hosts distribution among five neighborhood group
    :df: input dataframe
    :saved_plot: the file name of the saved plot
    """

    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    viz_2 = sns.violinplot(data=df, x='neighbourhood_group', y='price')
    viz_2.set_title('Neighborhood Group Price Violin Plot')
    plt.subplot(1, 2, 2)
    sns.boxplot(x='neighbourhood_group', y='price', data=df, showfliers = False)
    plt.title('Neighborhood Group Price Boxplot')
    plt.savefig(os.path.join(save_path, "plot_price_neighberhood_group.png"), dpi=100)

# neighbourhood_group_price_analysis(data[data.price<500],"plot_price_neighberhood_group.png")

def room_type_analysis(df, saved_plot):
    """
    This function is to draw a pie chart and count plot to analyze the room_type attribute
    :df: input dataframe
    :saved_plot: the file name of the saved plot
    """

    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)

    f,ax = plt.subplots(1,2,figsize=(15, 6))
    count = data['room_type'].value_counts()
    count.plot.pie(explode=[0,0.05,0],autopct='%1.2f%%',ax=ax[0],shadow=True)
    ax[0].set_title('Room Type Share Pie Chart')

    sns.countplot(x = 'room_type',hue = "neighbourhood_group", data=data,ax=ax[1],order = count.index)
    ax[1].set_title('Room Type per Neighbourhood Group Histogram')
    plt.savefig(os.path.join(save_path, saved_plot), dpi=100)
    plt.show()
# room_type_analysis(data,"plot_room_neighberhood_group.png")

def comprehensive_analysis(df,saved_plot):
    """
    This function is to draw a bar chart shows the price of each room type per neighborhood group
    :df: input dataframe
    :saved_plot: the file name of the saved plot
    """

    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)

    plt.figure(figsize=(15, 6))
    sns.barplot(x = "neighbourhood_group", y = "price", hue = "room_type",data = data)
    plt.title("The Price of Each Room Type per Neighborhood Group")
    plt.savefig(os.path.join(save_path, "plot_room_price_neighberhood_group.png"), dpi=100)
    plt.show()

# comprehensive_analysis(data,"plot_room_price_neighberhood_group.png")

def host_listing_properties(df,saved_plot,lower,upper):
    """
    This function is to draw a pie chart shows the data distribution of host_listing_count
    :df: input dataframe
    :saved_plot: the file name of the saved plot
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)
    filtered_calculated_host_listings_count = remove_extreme(df, 'calculated_host_listings_count',lower,upper)  
    labels = filtered_calculated_host_listings_count.unique()
    sizes = filtered_calculated_host_listings_count.value_counts()*100
    ax = plt.gca()
    fig = plt.figure(figsize=(20,20))
    ax.pie(sizes, labels = labels, autopct = '%1.1f%%')
    ax.set_title('Host Listing Properties Pie Chart')
    fig.legend(labels=labels)
    plt.savefig(os.path.join(save_path, saved_plot), dpi=1000)

# host_listing_properties(data, "plot_host_listing_count.png",0.01,0.88)

def neighborhood_rooms_distribution (df,top,saved_plot):
    """
    This function is to draw two plots showing neighborhood with top number of rooms in the descending order
    :df: input dataframe
    :top： input int
    :saved_plot: the file name of the saved plot
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(saved_plot,str)
    assert isinstance(top,int)
    assert top>=1

    fig,ax = plt.subplots(1,2,figsize=(15,8))
    count = data['neighbourhood'].value_counts()
    clr = ("blue", "forestgreen", "gold", "red", "purple",'cadetblue','hotpink','orange','darksalmon','brown')
    count.sort_values(ascending=False)[:top].sort_values().plot(kind='barh', color=clr,ax=ax[0])
    ax[0].set_title("Top " +str(top)+ " neighbourhood by the number of rooms")
    ax[0].set_xlabel('The number of rooms')
    ax[0].set_ylabel('Neighbourhood ')


    groups = list(count.index)[:top]
    counts = list(count[:top])
    counts.append(count.agg(sum)-count[:top].agg('sum'))
    groups.append('Other')
    type_dict=pd.DataFrame({"group":groups,"counts":counts})
    clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
    qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
    plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
    # plt.subplots_adjust(wspace =0.5, hspace =0)
    plt.savefig(os.path.join(save_path, "plot_room_number_neighbourhood.png"), dpi=100)

# neighborhood_rooms_distribution (data,10,"plot_room_number_neighbourhood.png")

def neighborhood_sort_by_mean_price (df,top):
    """
    This function is to draw a bar plot showing neighborhood with rooms according to the mean price
    :df: input dataframe
    :top： input int
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(top,int)
    assert top>=1

    data_top_prices_by_neighbourhood = data.groupby('neighbourhood').agg({'price': 'mean'}).sort_values('price').reset_index()
    sns.barplot(y="neighbourhood", x="price", data=data_top_prices_by_neighbourhood.head(top))
    plt.title("The Neighborhood with the top "+ str(top)+" mean price")
# neighborhood_sort_by_mean_price(data,10)

def neighborhood_sort_by_max_price (df,top):
    """
    This function is to draw a bar plot showing neighborhood with rooms according to the max price
    :df: input dataframe
    :top： input int
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(top,int)
    assert top>=1

    sns.barplot(y="neighbourhood", x="price", data=df.nlargest(top,['price']))
    plt.title("The Neighborhood with the top "+str(top)+" max price")
# neighborhood_sort_by_max_price(data,10)

def make_wordcloud(words):

    """
    This function is to draw a bar plot showing neighborhood with rooms according to the mean price
    :words: input list contains all words
    """

    assert isinstance(words,pd.core.series.Series)
    for word in words:
      assert isinstance(word,str)

    text = ""
    for word in words:
        text = text + " " + word

    stopwords = set(STOPWORDS)
    word_cloud = WordCloud(width = 1000,
                       height = 800,
                       colormap='Pastel1',
                       margin = 0,
                       max_words = 300,  
                       max_font_size = 300, min_font_size = 20,  
                       background_color = "salmon",contour_width = 3).generate(text)
    plt.figure(figsize=(20,20))
    plt.imshow(word_cloud, interpolation="gaussian")
    plt.axis("off")
    plt.show()

# Remove missing values in the 'name' attribute
# make_wordcloud(data['name'].dropna())


# Most expensive Airbnbs 
# Storing the names of the 1000 luxurious airbnbs in New York in a separate variable
# expensive = data.sort_values(by = 'price', ascending = False)
# luxury = expensive.head(1000)
# words = luxury['name'].dropna()
# make_wordcloud(words)

def interactive_map(df,attribute,hover_data):
    """
    This function interactively visualizes the listed rooms according to the input attribute. 
    After clicking on the specific room, the user can see the hover data.
    The displayed icon's size is determined by the listed rooms
    :df: input dataframe
    :attribute: the attribute of the input dataframe 
    """    
    assert isinstance(df,pd.core.frame.DataFrame)
    assert isinstance(attribute,str)
    assert attribute in df.dtypes
    assert isinstance(hover_data,list)
    for hover in hover_data:
       assert hover in df.dtypes

    # Set up a scatter plot on a tile map 
    fig = px.scatter_mapbox(df, 
                        hover_data = hover_data,
                        hover_name = attribute,
                        lat="latitude", 
                        lon="longitude", 
                        color=attribute, 
                        size="price",
                        color_continuous_scale = px.colors.cyclical.Twilight, 
                        size_max = 20, 
                        opacity = .60,
                        zoom=10)
    fig.layout.mapbox.style = 'open-street-map'
    fig.update_layout(title_text = "Classify listed rooms by "+ attribute+  " in NYC<br>(Click legend to toggle borough)", height = 400)
    fig.show()
# interactive_map(data,"neighbourhood_group",['price','minimum_nights','room_type','number_of_reviews'])