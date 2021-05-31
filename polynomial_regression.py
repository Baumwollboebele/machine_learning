# %%
import seaborn as sns
import numpy as np
from math import sin

# %%

def f_x3(x):
    return ((1/4)*x**3)-(x**2)+((1/4)*x)+(3/2)

def f_x2(x):
    return x**2+x

def sinus(x):
    return sin(x)

def data_generator(func,values = np.arange(-10,10,0.2),n= 10):

    x =[]
    y =[]
    for _ in range(n):
        for j in values:
            x.append(j)
            y.append(func(j)+np.random.normal(func(j),0.1))
    return [x,y]


#%%

x,y = data_generator(func=sinus)
sns.scatterplot(x=x,y=y)

#ax^3+bx^3+c^x+d
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# %%
poly_x_train = [[x,x**2,x**3] for x in x_train]
poly_x_test = [[x,x**2,x**3] for x in x_test]

#%%
poly_features_test[0]

# %%
poly = PolynomialFeatures(degree=3)
poly_features_train = poly.fit_transform(poly_features_train)
poly_features_test = poly.fit_transform(poly_features_test)
# %%
poly_features_test[0]
#%%
model_poly = LinearRegression()
model_poly.fit(poly_features_train,y_train)
# %%

y_predict = model_poly.predict(poly_features_test)
# %%
mae = metrics.mean_absolute_error(y_test, y_predict)
mse = metrics.mean_squared_error(y_test, y_predict)
r2 = metrics.r2_score(y_test, y_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

# %%
sns.scatterplot(x_train,y_train)
# %%
sns.lineplot(x_test,y_predict)
# %%

x_train = np.array(x_train).reshape((len(x_train),1))
model_linear = LinearRegression()
model_linear.fit(x_train,y_train)
y_predict = model_linear.predict(np.array(x_test).reshape(len(x_test),1))

sns.lineplot(x_test,y_predict)

# %%
mae = metrics.mean_absolute_error(y_test, y_predict)
mse = metrics.mean_squared_error(y_test, y_predict)
r2 = metrics.r2_score(y_test, y_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

# %%

# %%
