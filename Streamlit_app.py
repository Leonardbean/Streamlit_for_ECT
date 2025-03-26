import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
import pycountry
import numpy as np

# ================================
# 初始化界面配置
# ================================
st.set_page_config(
    page_title='数据驾驶舱',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ================================
# 数据加载与预处理
# ================================
@st.cache_data
def load_data():
    # 修改为你的实际文件路径
    df = pd.read_csv("D:\Python\projects\Streamlit_fro_ECT\ecommerce_transactions.csv")
    
    # 日期处理
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # 年龄分组
    df['Age_Group'] = pd.cut(df['Age'],
                            bins=[18, 30, 50, 70, 100],
                            labels=['18-30岁', '30-50岁', '50-70岁', '70岁+'],
                            right=False)
    return df

df = load_data()

# ================================
# 侧边栏控制面板
# ================================
with st.sidebar:
    st.header("控制面板")
    
    # 日期范围选择
    start_date = st.date_input("开始日期", value=df['Date'].min())
    end_date = st.date_input("结束日期", value=df['Date'].max())
    
    # 国家选择
    selected_countries = st.multiselect(
        "选择国家",
        options=df['Country'].unique(),
        default=df['Country'].unique()[:3]
    )
    
    # 年龄范围选择
    age_range = st.slider("年龄范围", 18, 100, (25, 65))

# ================================
# 核心指标看板
# ================================
st.header("实时业务看板")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总销售额", f"{df['Amount'].sum()/1e6:.2f} 百万")
with col2:
    st.metric("活跃用户数", df['User_Name'].nunique())
with col3:
    st.metric("商品类目数", df['Category'].nunique())

# ================================
# 跨国销售趋势分析
# ================================
with st.expander("🌍 跨国销售趋势分析", expanded=True):
    # 数据预处理
    country_ts = df.groupby(['Country', pd.Grouper(key='Date', freq='W-MON')])['Amount'].sum().unstack(0)
    
    # 控制组件
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        top_n = st.slider("显示TOP国家", 5, 50, 15)
        selected_ts_countries = st.multiselect(
            "指定国家",
            country_ts.columns.tolist(),
            default=country_ts.sum().nlargest(3).index.tolist()
        )
    with col2:
        roll_avg = st.checkbox("启用7日移动平均", True)
        log_scale = st.toggle("对数坐标轴")

    # 可视化
    fig = px.line(
        country_ts.resample('W').mean() if roll_avg else country_ts,
        x=country_ts.index,
        y=selected_ts_countries or country_ts.sum().nlargest(top_n).index,
        labels={'value': '周销售额（万元）'},
        line_shape="spline"
    )
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

# ================================
# 动态时序分析
# ================================
with st.expander("📉 动态时序分析", expanded=True):
    # ---- 双轴时序图 ----
    st.subheader("销售趋势与订单量分析")
    
    # 数据聚合
    Daily_Sales = df.resample('D', on='Date').agg({
        'Amount': 'sum',
        'ID': 'count'
    }).rename(columns={'ID': 'Order_Count'})
    
    # 创建双轴图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 添加销售额折线图（主Y轴）
    fig.add_trace(
        px.line(Daily_Sales, y='Amount', title='销售额趋势').data[0],
        secondary_y=False
    )
    
    # 添加订单量柱状图（次Y轴）
    fig.add_trace(
        px.bar(Daily_Sales, y='Order_Count', opacity=0.3).data[0],
        secondary_y=True
    )
    
    # 图表格式设置
    fig.update_layout(
        height=400,
        title_text="销售额与订单量双轴分析",
        xaxis_title="日期",
        yaxis_title="销售额",
        yaxis2_title="订单量",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- STL分解 ----
    st.subheader("销售趋势分解（STL）")
    
    # 执行STL分解
    Stl_Result = STL(Daily_Sales['Amount'], period=30).fit()
    
    # 创建分解结果DataFrame
    Components = pd.DataFrame({
        '观测值': Stl_Result.observed,
        '趋势项': Stl_Result.trend,
        '季节项': Stl_Result.seasonal,
        '残差项': Stl_Result.resid
    })
    
    # 可视化分解结果
    st.line_chart(
        Components,
        height=400,
        use_container_width=True
    )

# ================================
# 用户价值分层（RFM模型）
# ================================
with st.expander("👥 用户价值分层分析", expanded=True):
    # RFM计算
    snapshot_date = df['Date'].max().normalize()
    rfm = df.groupby('User_Name').agg(
        Recency=('Date', lambda x: (snapshot_date - x.max()).days),
        Frequency=('ID', 'size'),
        Monetary=('Amount', 'sum')
    ).query("Monetary > 0").reset_index()

    # 交互控制
    col1, col2 = st.columns(2)
    with col1:
        r_cutoff = st.slider('Recency 分段', 1, 365, (30, 90))
    with col2:
        f_cutoff = st.slider('Frequency 分段', 1, 100, (5, 10))

    # 动态分箱
    r_bins = [0] + list(r_cutoff) + [365]
    f_bins = [0] + list(f_cutoff) + [100]
    
    rfm['R_Score'] = pd.cut(rfm['Recency'], bins=r_bins, labels=[3, 2, 1])
    rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=f_bins, labels=[1, 2, 3])
    rfm['RFM_Group'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)

    # 3D可视化
    segment_map = {
        '11': '流失风险客户', '12': '一般维持客户', '13': '高频低价值客户',
        '21': '新晋潜力客户', '22': '稳定价值客户', '23': '高频中价值客户',
        '31': '重要挽留客户', '32': '核心价值客户', '33': '顶级VIP客户'
    }
    rfm['Segment'] = rfm['RFM_Group'].map(segment_map)
    
    fig = px.scatter_3d(
        rfm,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        size='Monetary',
        hover_name='User_Name',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================
# 高频用户分析模块
# ================================
def hf_user_analysis(df):
    # 核心计算
    User_Freq = df['User_Name'].value_counts().reset_index()
    User_Freq.columns = ['User_Name', 'Purchase_Count']
    
    # 交互控制
    with st.sidebar.expander("⚙️ 高频用户设置"):
        percentile = st.slider("百分比阈值", 0.5, 0.9, 0.6)
        min_purchase = st.number_input("最低购买次数", 100, 2000, 500)
    
    # 计算阈值
    threshold = User_Freq['Purchase_Count'].quantile(percentile)
    High_Freq_Users = User_Freq[
        (User_Freq['Purchase_Count'] >= threshold) &
        (User_Freq['Purchase_Count'] >= min_purchase)]
    
    # 可视化
    with st.expander("🔍 高频用户分析", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("高频用户数", f"{len(High_Freq_Users)}人")
        col2.metric("购买量占比", 
                   f"{(High_Freq_Users['Purchase_Count'].sum()/User_Freq['Purchase_Count'].sum())*100:.1f}%")
        col3.metric("平均频次", f"{High_Freq_Users['Purchase_Count'].mean():.0f}次/人")
        
        # 合并数据
        Hf_Merged = pd.merge(High_Freq_Users, df, on='User_Name')
        
        # 标签页展示
        tab1, tab2 = st.tabs(["人口特征", "地理分布"])
        with tab1:
            fig = px.pie(Hf_Merged, names='Age_Group', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.bar(Hf_Merged['Country'].value_counts(), 
                        labels={'value':'用户数量'})
            st.plotly_chart(fig, use_container_width=True)
    
    # 数据下载
    st.sidebar.download_button(
        label="📥 导出高频用户",
        data=High_Freq_Users.to_csv().encode('utf-8'),
        file_name="high_freq_users.csv"
    )

# 执行高频用户分析
hf_user_analysis(df)


# ================================
# 国家分布统计分析（热力图）
# ================================
with st.expander("🌐 国家分布统计分析", expanded=True):
    st.subheader("各国关键指标热力图")
    
    # 计算国家维度指标
    Country_Group_analysis = df.groupby('Country').agg(
        Total_by_Country=('Amount', 'sum'),
        UserCount_by_Country=('User_Name', 'nunique'),
        Avg_by_country=('Amount', 'mean'),
        Mid_by_country=('Amount', 'median')
    ).sort_values('Total_by_Country', ascending=False)
    
    # 数据标准化（Z-Score）
    country_normalized = Country_Group_analysis.apply(
        lambda x: (x - x.mean()) / x.std(), 
        axis=0
    )
    
    # 生成热力图
    fig = px.imshow(
        country_normalized.T,
        labels=dict(x="国家", y="指标", color="标准化值"),
        x=country_normalized.index,
        y=country_normalized.columns,
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    fig.update_layout(
        height=600,
        xaxis_title="国家",
        yaxis_title="统计指标"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 原始数据展示
    show_raw_data_1 = st.checkbox("显示原始数据", value= False)
    if show_raw_data_1:
        st.dataframe(
            Country_Group_analysis.style.background_gradient(cmap='Blues'),
            height=300
        )

# ================================
# 支付方式分析（热力图+箱线图） 
# ================================
with st.expander("💳 支付方式分析", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("支付方式-年龄分布热力图")
        # 生成交叉表
        pay_cross = df.pivot_table(
            index='Pay_Method',
            columns='Age_Group',
            values='ID',
            aggfunc='count',
            margins=True
        )
        
        # 热力图可视化
        fig = px.imshow(
            pay_cross.iloc[:-1, :-1],  # 排除合计行/列
            labels=dict(x="年龄组", y="支付方式", color="订单量"),
            color_continuous_scale='YlOrRd',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("支付方式金额分布箱线图")
        # 生成箱线图
        fig = px.box(
            df,
            x='Pay_Method',
            y='Amount',
            color='Pay_Method',
            points="outliers",
            log_y=True  # 对数坐标处理金额范围
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="支付方式",
            yaxis_title="金额（对数坐标）"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # 原始交叉表展示
    show_raw_data_2 = st.checkbox("查看支付方式-年龄交叉表")
    if show_raw_data_2:
        st.dataframe(
            pay_cross.style.background_gradient(cmap='YlOrRd'),
            height=300
        )



# ================================
# 年龄-国家-品类三维分析
# ================================
with st.expander("🌐📦👥 年龄-国家-品类三维分析", expanded=True):
    # 数据预处理
    df_melted = df.groupby(['Age_Group', 'Country', 'Category'], observed= False).agg(
        Total_Amount=('Amount', 'sum'),
        Order_Count=('ID', 'count')
    ).reset_index()

    # 交互控制面板
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_choice = st.radio("选择分析指标", 
                               ('总销售额', '订单量'),
                               horizontal=True)
    with col2:
        selected_categories = st.multiselect(
            "选择品类",
            options=df_melted['Category'].unique(),
            default=df_melted['Category'].unique()[:3]
        )
    with col3:
        log_scale = st.checkbox("启用对数变换", True)

    # 根据选择过滤数据
    df_filtered = df_melted[df_melted['Category'].isin(selected_categories)]
    metric_col = 'Total_Amount' if metric_choice == '总销售额' else 'Order_Count'

    # 创建分面热力图
    fig = px.density_heatmap(
        df_filtered,
        x="Country",
        y="Age_Group",
        z=metric_col,
        facet_col="Category",
        facet_col_wrap=3,
        color_continuous_scale='Viridis',
        height=800,
        title=f"{metric_choice}分布 - 按品类分层"
    )

    # 优化布局
    fig.update_layout(
        xaxis_title="国家",
        yaxis_title="年龄组",
        coloraxis_colorbar_title=metric_choice,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 调整分面标题格式
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    st.plotly_chart(fig, use_container_width=True)

    # 附加数据透视表
    if st.checkbox("显示原始数据透视表"):
        st.dataframe(
            df_melted.pivot_table(
                index=['Age_Group', 'Country'],
                columns='Category',
                values=[metric_col],
                aggfunc='sum'
            ).style.background_gradient(cmap='Blues'),
            height=400
        )

# ================================
# 启动应用
# ================================
if __name__ == "__main__":
    st.write("数据驾驶舱已就绪")