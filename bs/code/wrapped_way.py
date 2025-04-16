from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与初步探索
def load_and_explore_data(file_path):

    try:
        df = pd.read_excel(file_path)
        print(f"数据加载成功！共{df.shape[0]}行，{df.shape[1]}列。")

        # 显示基本信息
        print("\n数据前5行:")
        print(df.head())

        print("\n数据基本信息:")
        print(df.info())

        print("\n基本统计描述:")
        print(df.describe())

        # 检查缺失值
        missing = df.isnull().sum()
        print("\n缺失值统计:")
        print(missing)

        # 检查异常值
        print("\n检查潜在异常值:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"{col}: {outliers}个异常值")

        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


# 数据清洗
def clean_data(df):

    print("\n2. 数据清洗")
    if df is None:
        return None

    # 复制数据以免修改原始数据
    clean_df = df.copy()

    # 处理缺失值
    print("处理缺失值...")
    # 检查缺失比例
    missing_ratio = clean_df.isnull().mean()

    # 删除缺失比例过高的列（超过30%）
    high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
    if high_missing_cols:
        print(f"删除缺失比例过高的列: {high_missing_cols}")
        clean_df = clean_df.drop(columns=high_missing_cols)

    # 对剩余列进行缺失值填充
    numeric_cols = clean_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = clean_df.select_dtypes(include=['object', 'category']).columns

    # 数值型列用中位数填充
    for col in numeric_cols:
        if clean_df[col].isnull().sum() > 0:
            median_val = clean_df[col].median()
            clean_df[col].fillna(median_val, inplace=True)
            print(f"列 '{col}' 的缺失值用中位数 {median_val} 填充")

    # 分类型列用众数填充
    for col in categorical_cols:
        if clean_df[col].isnull().sum() > 0:
            mode_val = clean_df[col].mode()[0]
            clean_df[col].fillna(mode_val, inplace=True)
            print(f"列 '{col}' 的缺失值用众数 '{mode_val}' 填充")

    # 处理异常值
    print("\n处理异常值...")
    for col in numeric_cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 计算异常值数量
        outliers_count = ((clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)).sum()

        if outliers_count > 0:
            print(f"列 '{col}' 有 {outliers_count} 个异常值")
            # 将异常值替换为上下界
            clean_df.loc[clean_df[col] < lower_bound, col] = lower_bound
            clean_df.loc[clean_df[col] > upper_bound, col] = upper_bound
            print(f"  - 已将异常值替换为上下界 ({lower_bound}, {upper_bound})")

    # 移除重复行
    duplicates = clean_df.duplicated().sum()
    if duplicates > 0:
        print(f"\n删除 {duplicates} 个重复行")
        clean_df = clean_df.drop_duplicates()

    print(f"\n数据清洗完成。清洗后数据形状: {clean_df.shape}")
    return clean_df


# 特征编码
def encode_features(df):

    print("\n3. 特征编码")
    if df is None:
        return None, None, None

    # 创建副本
    encoded_df = df.copy()

    # 初始列分类
    numeric_cols = encoded_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"原始数值型列: {numeric_cols}")
    print(f"原始分类型列: {categorical_cols}")

    # 编码映射字典
    binary_map = {'Yes': 1, 'No': 0}
    ordinal_map_1 = {'Low': 1, 'Moderate': 2, 'High': 3}
    ordinal_map_2 = {'Poor': 1, 'Average': 2, 'Good': 3}
    freq_map = {'Never': 0, 'Occasionally': 1, 'Frequently': 2}

    replace_dict = {
        'Family_History': binary_map,
        'Chronic_Illness': binary_map,
        'Physical_Activity': ordinal_map_1,
        'Social_Support': ordinal_map_1,
        'Extracurricular_Involvement': ordinal_map_1,
        'Sleep_Quality': ordinal_map_2,
        'Diet_Quality': ordinal_map_2,
        'Substance_Use': freq_map,
        'Counseling_Service_Use': freq_map,
        'Gender': {'Male': 1, 'Female': 0},
    }

    for col, mapping in replace_dict.items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].replace(mapping)
            print(f"列 '{col}' 使用映射编码")


    # 对频率编码的列进行处理
    freq_encode_cols = ['Course', 'Relationship_Status','Residence_Type']
    for col in freq_encode_cols:
        if col in encoded_df.columns:
            freq_map = encoded_df[col].value_counts(normalize=True)
            encoded_df[col] = encoded_df[col].map(freq_map)
            print(f"列 '{col}' 使用频率编码")

    # 去除异常值（仅保留适用的连续变量列）
    from scipy.stats.mstats import winsorize
    winsorize_cols = ['Age', 'CGPA', 'Financial_Stress', 'Semester_Credit_Load']
    for col in winsorize_cols:
        encoded_df[col] = winsorize(encoded_df[col], limits=[0.05, 0.05])  # 缩尾5%

    # 最终数值型和类别型列（已全部数值化）
    final_numeric_cols = encoded_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    final_categorical_cols = []

    print(f"编码后数据维度: {encoded_df.shape}")
    return encoded_df, final_numeric_cols, final_categorical_cols

# 数据标准化
def standardize_data(df):
    scaler = StandardScaler()
    # 仅对数值型列标准化（包括原始连续变量）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
    # 合并非数值型列（如分箱后的类别列）
    non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    scaled_df = pd.concat([scaled_df, df[non_numeric_cols]], axis=1)
    return scaled_df, scaler


# 聚类函数
def perform_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    print(f"\n数据已分为 {n_clusters} 个聚类")
    return clusters

#使用随机森林进行RFE
#补充RFE特征选择函数
# def select_features_by_rfe(X, y, n_features_to_select=8):
#     """包裹式特征选择（递归特征消除）"""
#     estimator = RandomForestClassifier(random_state=42)
#     selector = RFE(estimator, n_features_to_select=n_features_to_select)
#     selector.fit(X, y)
#     selected_cols = X.columns[selector.support_]
#     print("\nRFE特征选择结果：")
#     print("重要特征排序:", sorted(zip(selector.ranking_, X.columns)))
#     return selected_cols.tolist()

def select_features_by_svm_rfe(X, y, n_features_to_select=5):
    """包裹式特征选择（递归特征消除）使用SVM"""
    # 创建SVM分类器（使用线性核以获得特征权重）
    estimator = SVC(kernel='linear', random_state=42)

    # 创建RFE选择器
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    selector.fit(X, y)

    # 获取选中的特征列
    selected_cols = X.columns[selector.support_]

    print("\nSVM-RFE特征选择结果：")
    print("重要特征排序:", sorted(zip(selector.ranking_, X.columns)))
    return selected_cols.tolist()

#主函数
def main(file_path):
    # 路径处理
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载数据
    def load_and_explore_data(file_path):
        full_path = os.path.abspath(os.path.join(current_dir, file_path))
        print(f"正在加载文件：{full_path}")
        assert os.path.exists(full_path), f"文件不存在：{full_path}"
        return pd.read_excel(full_path)

    df = load_and_explore_data(file_path)

    # 清洗数据
    clean_df = clean_data(df)

    # 编码特征
    encoded_df, numeric_cols, _ = encode_features(clean_df)

    # 标准化
    scaled_X, _ = standardize_data(encoded_df)

    # 聚类
    y_cluster = perform_clustering(scaled_X)

    # 基于聚类标签的特征选择
    print("\n基于聚类的特征选择：")
    selected_features = select_features_by_svm_rfe(scaled_X, y_cluster, n_features_to_select=10)

    print(f"\n最终选出的特征列: {selected_features}")

if __name__ == "__main__":

    data_path = os.path.join('..', 'data', 'students_mental_health_survey.xlsx')
    main(file_path=data_path)
