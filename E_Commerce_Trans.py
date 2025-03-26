import pandas as pd 
import numpy as np
from numpy import size
import matplotlib.pyplot as plt
import seaborn as  sns
import joblib        #模型保存
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
import sys
import io
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import ElasticNet
import geopandas as gpd
import geodatasets as gds

#--------------------
# 0.全局设置
#--------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# 设置显示所有列（无省略）
pd.set_option("display.max_columns", None)    # 取消列数限制
pd.set_option("display.width", 1000)         # 设置输出宽度
# pd.set_option("display.max_colwidth", 20)  # 调整单列最大宽度
pd.options.display.float_format = "{:.2f}".format #强制保留2位小数点


#--------------------
# 1.数据加载和初步检查
#--------------------
df = pd.read_csv(".\ecommerce_transactions.csv")
print(f"\n=== 前五行数据示例 ===\n{df.head()}\n")
print('=== 基本数据信息 ===')
print(f"{df.info()}\n")


#--------------------
# 2.数据清洗
#--------------------

# 1-处理重复值
df = df.drop_duplicates()

# 2-检查分类变量唯一值
print("===唯一值检查===")
print(f"Country 唯一值:{df['Country'].unique()}")
print(f"Pay_Method 唯一值:{df['Pay_Method'].unique()}")
print(f"Category 唯一值:{df['Category'].unique()}")

#--------------------
# 3.统计分析和EDA
#--------------------
# 1-年龄分组
bins = [18, 30, 50, 70, 100]
lables = ['18-30岁', '30-50岁', '50-70岁', '70岁+']
df['Age_Group'] = pd.cut(df['Age'], bins = bins, labels= lables, right= False)
print(f"\n ===年龄分组=== \n{df['Age_Group']}")

# 2-年龄分布统计指标 
Age_Group_analysis = df.groupby('Age_Group').agg(
    Total_by_Age = ('Amount', 'sum'),       #各组总额
    Avg_by_Age =('Amount', 'mean'),         #各组均值
    Mid_by_Age = ('Amount', 'median'),      #各组中位数
    OrderCount_by_Age = ('ID', 'count')     #各组数量
).reset_index()
print(f" \n===各年龄组售额统计指标=== \n{Age_Group_analysis}")
#品类偏好分析
Category_by_Age = df.groupby(['Age_Group', 'Category']).size().unstack().fillna(0)
Category_Percent = Category_by_Age.div(Category_by_Age.sum(axis= 1), axis= 0) * 100
print(f"\n===各年龄品类偏好(按百分比%)===\n{Category_Percent}")

# 3-国家分布统计分析
Country_Group_analysis = df.groupby('Country').agg(
    Total_by_Country = ('Amount', 'sum'),             #总额
    UserCount_by_Country = ('User_Name', 'nunique'), #客户数量
    Avg_by_country = ('Amount', 'mean'),               #均值
    Mid_by_country = ('Amount', 'median'),            #中值
).sort_values('Total_by_Country', ascending= False)
print(f"\n===各国售额统计指标===\n{Country_Group_analysis}")

# 4-高频用户识别
User_Freq = df['User_Name'].value_counts().reset_index()
User_Freq.columns = ['User_Name', 'Purchase_Count']
#定义高频用户：前60% && >100
threshold_percentile = User_Freq['Purchase_Count'].quantile(0.6)
threshold_min = 500
#筛选高频用户
High_Freq_Users = User_Freq[(User_Freq['Purchase_Count'] >= threshold_percentile)
                            &
                            (User_Freq['Purchase_Count'] >= threshold_min)
                            ]
High_Freq_Users = High_Freq_Users.sort_values('Purchase_Count', ascending= False)
print(f"\n===高频用户数量 {len(High_Freq_Users)}===\n")
print(High_Freq_Users)
#合并高频用户特征
Hf_Merged = pd.merge(High_Freq_Users, df, on= 'User_Name')
Hf_Age_Dist = Hf_Merged['Age_Group'].value_counts(normalize= True) * 100
Hf_Country_Dist = Hf_Merged['Country'].value_counts(normalize= True) * 100
print(f"\n===高频用户年龄分布(百分比%)===\n{Hf_Age_Dist}")
print(f"\n===高频用户国家分布(百分比%)===\n{Hf_Country_Dist}")



# 5-支付方式&年龄画像交叉分析
pay_cross = df.pivot_table(index= 'Pay_Method',
                           columns= 'Age_Group',
                           values= 'ID',
                           aggfunc= 'count',
                           margins= True,
                           observed= False)

print(f"\n===支付方式&年龄===\n{pay_cross}")
#支付金额分布箱线图
sns.boxplot(x= 'Pay_Method', y= 'Amount', data= df)
plt.show()

# 5.1-年龄&国家&品类三维图
Age_Country_Category = df.pivot_table(index= ['Age_Group', 'Country'],
                                      columns= 'Category',
                                      values= 'Amount',
                                      aggfunc= ['sum', 'count'],
                                      fill_value= 0)
print(f"\n===年龄&国家&品类===\n{Age_Country_Category}")


# 6-RFM计算
df['Date'] = pd.to_datetime(df['Date'], errors= 'coerce')
df = df.dropna(subset= ['Date'])     #删除无效日期
snapshot_date = df['Date'].max().normalize() #日期快照

rfm = (
    df.groupby('User_Name', observed= True).agg(
        Recency = ('Date', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('ID', 'size'),
        Monetary = ('Amount', 'sum')
    ).query("Monetary > 0")
     .reset_index() 
)
print(f"\n===RFM===\n{rfm}")

# 6.1-产品关联分析

# 7- 国家-品类热力图
Country_Categoty = pd.crosstab(df['Country'], df['Category'])
sns.heatmap(Country_Categoty, annot= True, fmt= 'd')
plt.show()




#--------------------
# 4.时间序列分析
#--------------------
"""颗粒度：日"""
df['Date'] = pd.to_datetime(df['Date'])
daily_sales = df.resample('D', on= 'Date').agg({'Amount' : 'sum', 'ID':'count'})
#消除噪音（7日平均）
daily_sales['7D_MA'] = daily_sales['Amount'].rolling(window= 7).mean()
#可视化
fig = px.line(daily_sales, y= ['Amount', '7D_MA'],
              title = "每日销售趋势")
fig.show()

"""颗粒度：月 季节性分解"""
monthly_sales = df.resample('M', on= 'Date').agg({'Amount' : 'sum'})
#可视化
decomposition = STL(monthly_sales, period= 12).fit()
decomposition.plot()
plt.show()
