# Hygiene Prediction App
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.session import set_env, info as session_info
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import joblib

unique_zipcodes = ['98101','98102','98103','98104','98105','98106','98107','98108','98109',
                  '98112','98115','98116','98117','98118','98119','98121','98122','98125','98126','98133',
                  '98134','98136','98144','98146','98166','98168','98177','98178','98188','98199']

unique_cuisines = ['Afghan', 'African', 'American (New)', 'American (Traditional)', 'Asian Fusion', 'Australian',
                    'Barbeque', 'Basque', 'Belgian', 'Brazilian', 'Breakfast & Brunch', 'British', 'Buffets',
                    'Burgers', 'Cafes', 'Cajun/Creole', 'Cambodian', 'Cantonese', 'Caribbean', 'Cheesesteaks',
                    'Chicken Wings', 'Chinese', 'Colombian', 'Comfort Food', 'Creperies', 'Cuban', 'Delis',
                    'Dim Sum', 'Diners', 'Egyptian', 'Ethiopian', 'Fast Food', 'Filipino', 'Fish & Chips', 'Fondue',
                    'Food Court', 'Food Stands', 'French', 'Gastropubs', 'German', 'Gluten-Free', 'Greek', 'Haitian',
                    'Halal', 'Hawaiian', 'Himalayan/Nepalese', 'Hot Dogs', 'Hot Pot', 'Indian', 'Indonesian', 'Irish',
                    'Italian', 'Japanese', 'Korean', 'Kosher', 'Laotian', 'Latin American', 'Lebanese', 'Live/Raw Food',
                    'Malaysian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Modern European', 'Mongolian', 'Moroccan',
                    'Pakistani', 'Persian/Iranian', 'Pizza', 'Polish', 'Puerto Rican', 'Restaurants', 'Russian', 'Salad',
                    'Salvadoran', 'Sandwiches', 'Scandinavian', 'Scottish', 'Seafood', 'Senegalese', 'Shanghainese',
                    'Soul Food', 'Soup', 'Southern', 'Spanish', 'Steakhouses', 'Sushi Bars', 'Szechuan', 'Taiwanese',
                    'Tapas Bars', 'Tapas/Small Plates', 'Tex-Mex', 'Thai', 'Trinidadian', 'Turkish', 'Vegan', 'Vegetarian',
                    'Venezuelan', 'Vietnamese']

filename = 'finalized_model.sav'
clf_model = joblib.load(filename)


scope_name = get_scope()
clear(scope_name)

def hygiene_prediction():
    data = input_group("Hygiene Prediction Input",[
        input("Provide your review comments：",name = 'review', type=TEXT,required = True),
        select("Please select a cuisine type(s)：",options = unique_cuisines,name = 'cuisine', multiple = True, type = TEXT,required = True),
        select("Please select a zipcode：",options = unique_zipcodes, name = 'zipcode',type=TEXT,required = True,multiple = False),
        select("Provide your rating:",name = 'rating', type = TEXT, required = True,options = ['1','2','3','4','5'],multiple = False)])
    data['cuisine'] = str(data['cuisine']) # need to convert list to string
    info = [[data['review'],data['cuisine'],data['zipcode'],'7',data['rating']]]
    input_df = pd.DataFrame(info, columns = ['text', 'cuisines_offered', 'zipcode','num_reviews', 'avg_rating'])
    pred = float(clf_model.predict_proba(input_df)[:,1])
    put_markdown(r""" # Hygiene Prediction Result
    """, lstrip=True)
    put_table([['Cuisines','Zipcode','Rating','Review'],
             [data['cuisine'],data['zipcode'],data['rating'],data['review']],])
    put_html('<hr')
    put_markdown( 'This restaurant is: `{:.2%}` likely to pass a hygiene inspection'.format(pred))

if __name__ == '__main__':
    #start_server(hygiene_prediction,port = 36535,debug = True)
    hygiene_prediction()
