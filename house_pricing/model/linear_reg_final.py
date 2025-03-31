import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_range_to_value(x):
    limits = x.split('-')
    if len(limits) == 2:
        try:
            return (float(limits[0]) + float(limits[1])) // 2
        except:
            return None
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for keys, data in df.groupby('location'):
        mean = np.mean(data['price_per_sqft'])
        std = np.std(data['price_per_sqft'])
        data = data[(data['price_per_sqft'] <= (mean + std)) & (data['price_per_sqft'] >= (mean - std))]
        df_out = pd.concat([df_out, data], ignore_index=True)
    return df_out

def plot_scatter_chart(df, location):   # total_sqft vs price for 2 and 3 BHK for specific location
    bhk2 = df[(df['location'] == location) & (df['bhk'] == 2)]
    bhk3 = df[(df['location'] == location) & (df['bhk'] == 3)]
    plt.figure()
    plt.scatter(bhk2['total_sqft'], bhk2['price'], marker='+', c='blue', label='2 BHK')
    plt.scatter(bhk3['total_sqft'], bhk3['price'], marker='*', c='green', label='3 BHK')
    plt.grid(True)
    plt.xlabel('sqft')
    plt.xlabel('price')
    plt.title('Price in ' + location)
    plt.legend()
    #plt.show()

def remove_bhk_outliers(df):
    exclude_indeces = []
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats_check = bhk_stats.get(bhk-1)
            if stats_check and stats_check['count'] > 5 :
                exclude_indeces = np.append(exclude_indeces, bhk_df[bhk_df['price_per_sqft'] < stats_check['mean']].index.values)
    return df.drop(exclude_indeces, axis='rows')

###### BEGINNING ######

df1 = pd.read_csv('Bengaluru_House_Data.csv')
dfx = df1.groupby('area_type')['area_type'].agg(['count'])
print(dfx)    # Is the data balanced in terms of area_type feature?

df2 = df1.drop(['area_type', 'availability', 'society', 'balcony'], axis = 'columns')
print(df2.head())    # Leaving 4 features - location, size, total_sqft and bath

print(df2.isnull().sum())
df2.dropna(inplace=True)    # Dropping rows with na values

print(df2.shape)    # 13246 rows left ---> enough data
print(df2['size'].unique())

df2['bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))   # converting string column size to numeric column - bhk
print(df2.head())

print(df2[~df2['total_sqft'].apply(is_float)].head(10))     # Column total_sqft contains int values but also ranges of int

df4 = df2
df4['total_sqft'] = df2['total_sqft'].apply(convert_range_to_value)
print(df4[~df2['total_sqft'].apply(is_float)].head(10))     # No more ranges of int
print('df4:')
print(df4.head())
print('New column')
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']    # Adding new column, *100000 to convert from lakh rupees to rupees
print(df4.head())

df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby('location')['location'].agg('count')
print(location_stats.sort_values(ascending = False))    # Exploring location feature in terms of balanced data

df5 = df4
df5['location'] = df4['location'].apply(lambda x: x if location_stats[x] > 10 else 'other')     # If specific location occurs less or equal to 10 times it's replaced with location 'other'

print(len(df5.location.unique()))    # 242 different locations are left

print(df5.head(10))
print(sum(df5['total_sqft']/df5['bhk'] < 300))
print(df5[df5['total_sqft']/df5['bhk'] < 300].head(5))

df6 = df5[~(df5['total_sqft']/df5['bhk'] < 300)]    # Filtering houses with small area of rooms
print(df6.head())
print(df6['price_per_sqft'].describe())

plot_scatter_chart(df6, 'Hebbal')

print(df6.shape)
df7 = remove_pps_outliers(df6)      # Filtering houses with price_per_sqft out of +- std_dev for specific location
print(df7.shape)

plot_scatter_chart(df7, 'Hebbal')

df8 = remove_bhk_outliers(df7)
print(df8.shape)
plot_scatter_chart(df8, 'Hebbal')

df9 = df8[df8['bhk']>df8['bath']-2]
print(df9.shape)    # 7251 rows are left

plt.figure()
plt.hist(df9['price_per_sqft'], rwidth=0.7)
plt.xlabel('Price per sqft')
plt.ylabel('Amount')
plt.grid(True)
#plt.show()

df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')     # Columns size and price_per_sqft are not needed anymore
print(df10.head())

print('########### MODEL ############')

dummies = pd.get_dummies(df10['location'])
print(dummies.head(10))

df11 = pd.concat([df10, dummies.drop(['other'], axis = 'columns')], axis = 'columns')
print(df11)
df12 = df11.drop(['location'], axis = 'columns')
print(df12.shape)

X = df12.drop(['price'], axis = 'columns')
print(X)

y = df12['price']
print(y)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, ShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

def find_best_model(X, y):
    algorithms = {
        'ridge_regression' : {
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.5, 0.25, 0.03, 0.05, 0.1],
                'max_iter': [50, 100, 500]
            }
        },
        'tree_regression': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        },
        'lasso_regression': {
            'model': Lasso(),
            'params': {
                'alpha': [0.00001, 0.001, 0.005, 0.01, 0.05],
                'selection': ['random', 'cyclic']
            }
        },
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
            }
        }
    }
    #cv = KFold(n_splits=10, shuffle = True)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    scores = []
    for model_name, config in algorithms.items():
        grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=cv, return_train_score=False, verbose=5)
        grid_search.fit(X, y)
        scores.append({
            'model': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        })
    model_stats_df = pd.DataFrame(scores)
    pd.set_option('display.max_columns', None)
    print(model_stats_df)


# find_best_model(X, y) -----> Linear regression is shown to be best model

print(X.drop(['total_sqft', 'bath', 'bhk'], axis='columns').columns=='other')

def predict_price(location, sqft, bath, bhk):
    location_id = np.where(X.drop(['total_sqft', 'bath', 'bhk'], axis='columns').columns == location, 1, 0)
    print(location_id)

    x = np.zeros(X.shape[1])
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    x[3:] = location_id
    return model.predict([x])

print(predict_price('other', 1000, 2, 5))
print(predict_price('other', 1000, 3, 6))
print(predict_price('other', 1000, 3, 7))

print(predict_price('Indira Nagar', 1000, 2,2))


# COMMENT OUT IF YOU WANT TO CREATE NEW MODEL
# import pickle
# with open('banglore_home_prices_model.pickle', 'wb') as f:
#     pickle.dump(model, f)
#
# import json
# columns = { 'data_columns': [col.lower() for col in X.columns]}
# with open('columns.json', 'w') as f:
#     f.write(json.dumps(columns))