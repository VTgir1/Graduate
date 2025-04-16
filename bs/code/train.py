from sklearn.metrics import (precision_score, recall_score, f1_score,roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from visual2 import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.svm import SVC
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载
# 初步分析
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

# 对异常值，缺失值进行清洗处理
def clean_data(df):

    print("\n2. 数据清洗")
    if df is None:
        return None
    
    # 复制数据，避免修改原数据
    clean_df = df.copy()
    
    # 处理缺失值
    print("处理缺失值...")
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

    #进行自定义编码
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
    freq_encode_cols = ['Course', 'Relationship_Status', 'Residence_Type']
    for col in freq_encode_cols:
        if col in encoded_df.columns:
            freq_map = encoded_df[col].value_counts(normalize=True)
            encoded_df[col] = encoded_df[col].map(freq_map)
            print(f"列 '{col}' 使用频率编码")

    # 去除异常值（仅保留适用的连续变量列）
    zscore_cols = ['Age', 'CGPA', 'Financial_Stress', 'Semester_Credit_Load']
    valid_cols = [col for col in zscore_cols if col in encoded_df.columns]

    if valid_cols:
        z_scores = np.abs(stats.zscore(encoded_df[valid_cols]))
        encoded_df = encoded_df[(z_scores < 3).all(axis=1)]
        print(f"已从列 {valid_cols} 中剔除异常值，剩余样本数: {encoded_df.shape[0]}")

    # 最终数值型和类别型列（已全部数值化）
    final_numeric_cols = encoded_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    final_categorical_cols = []

    print(f"编码后数据维度: {encoded_df.shape}")
    return encoded_df, final_numeric_cols, final_categorical_cols

# 数据标准化
def standardize_data(df):
    """
    对数值型特征进行标准化
    """
    print("\n4. 数据标准化")
    if df is None:
        return None, None

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    print("数据标准化完成")
    return scaled_df, scaler

# 聚类分析，分为3类
def perform_clustering(df, n_clusters=3):

    print("\n5. 聚类分析")
    if df is None:
        return None, None
    
    # 添加类L2正则化
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42,
        n_init=10,
        tol=1e-4,
    )
    clusters = kmeans.fit_predict(df)
    
    # 将聚类标签添加到原数据中
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # po出聚类分布
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("聚类分布:")
    for cluster, count in cluster_counts.items():
        print(f"聚类 {cluster}: {count} 个样本 ({count/len(df)*100:.2f}%)")
    
    # 使用PCA降维到2D进行可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    
    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        plt.scatter(pca_result[clusters == i, 0], pca_result[clusters == i, 1], label=f'聚类 {i}')
    plt.title('聚类结果可视化 (PCA降维)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.savefig('clusters_visualization.png')
    plt.close()
    
    # 分析每个聚类的特征
    print("\n各聚类特征分析:")
    cluster_features = {}
    for i in range(n_clusters):
        cluster_data = df[clusters == i]
        cluster_features[i] = cluster_data.mean()
        print(f"聚类 {i} 的特征均值:")
        print(cluster_features[i])
    
    return df_with_clusters, kmeans

# 基于聚类标签的预测模型
def build_prediction_model(df_with_clusters):
    print("\n6. 基于聚类标签的预测模型")
    if df_with_clusters is None:
        return None, None, None

    # 通过独立模块wrapped_way，手动筛选出特征，用作训练
    features = [
        'Social_Support', 'Course', 'Counseling_Service_Use', 'Gender',
        'Financial_Stress', 'Extracurricular_Involvement', 'CGPA',
        'Relationship_Status', 'Age', 'Semester_Credit_Load',
        'Residence_Type', 'Diet_Quality'
    ]
    X = df_with_clusters[features]
    y = df_with_clusters['cluster']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 模型配置
    #采用smote平衡数据
    models = {
        #随机森林
        'Random Forest': {
            'model': imbpipeline([
                ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
                ('model', RandomForestClassifier(random_state=42))
            ]),
            'params': {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [None, 5, 10, 20, 30, 50],
                'model__min_samples_split': [2, 5, 10, 15],
                'model__min_samples_leaf': [1, 2, 4, 6],
                'model__class_weight': [None, 'balanced', {0:1,1:3,2:2}],
                'model__max_features': ['sqrt', 'log2', 0.8]
            }
        },
        #支持向量机
        'SVM': {
            'model': imbpipeline([
                ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True, random_state=42, max_iter=1000000))
            ]),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'model__kernel': ['rbf', 'linear', 'poly'],
                'model__class_weight': [None, 'balanced', {0:1,1:3,2:2}],
                'model__degree': [2, 3, 4],
                'model__coef0': [0.0, 0.5, 1.0]
            }
        },
        #逻辑回归
        'Logistic Regression': {
            'model': imbpipeline([
                ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
                ('model', LogisticRegression(random_state=42, max_iter=5000))
            ]),
            'params': {
                'model__C': np.logspace(-4, 4, 20),
                'model__penalty': ['l2', 'elasticnet'],
                'model__solver': ['saga'],
                'model__l1_ratio': [0, 0.3, 0.5, 0.7, 1],
                'model__class_weight': [None, 'balanced', {0:1,1:3,2:2}]
            }
        },
        #朴素贝叶斯
        'Naive Bayes': {
            'model': imbpipeline([
                ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
                ('scaler', StandardScaler()),
                ('model', GaussianNB())
            ]),
            'params': {
                'model__var_smoothing': np.logspace(-9, -3, 10)
            }
        }
    }

    #性能验证，自动调参
    best_models = {}
    model_scores = {}
    feature_importances = {}

    for model_name, model_info in models.items():
        print(f"\n正在训练和优化 {model_name} 模型...")

        # 参数搜索
        random_search = RandomizedSearchCV(
            estimator=model_info['model'],
            param_distributions=model_info['params'],
            n_iter=20,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        random_search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = random_search.best_estimator_
        best_models[model_name] = best_model

        # 评估指标计算
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        # 计算指标
        test_accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
        mean_cv_score = cv_scores.mean()

        # 存储结果
        model_scores[model_name] = {
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'cv_mean_score': mean_cv_score,
            'cv_scores': cv_scores,
            'best_params': random_search.best_params_
        }

        # 输出结果
        print(f"{model_name} 最佳参数: {random_search.best_params_}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC值: {auc_score:.4f}")
        print(f"交叉验证平均分数: {mean_cv_score:.4f}")

        n_classes = len(np.unique(y))
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

        # 计算OvR策略下的各分类ROC指标
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算Macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # 计算Micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 可视化
        plt.figure(figsize=(10, 8))
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        # 绘制各分类曲线
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                     label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        # 绘制Macro-average
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})',
                 color='navy', linestyle=':', linewidth=3)

        # 绘制Micro-average
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=3)

        # 基准线
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - Multiclass ROC Curve')
        plt.legend(loc="lower right", prop={'size': 8})
        plt.grid(alpha=0.3)

        # 保存和清理
        plt.savefig(f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 特征重要性
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            importance = best_model.named_steps['model'].feature_importances_
            feature_importance = pd.DataFrame({
                '特征': features,
                '重要性': importance
            }).sort_values('重要性', ascending=False)
            feature_importances[model_name] = feature_importance

    # 结果汇总
    print("\n\n所有模型性能对比：")
    results_df = pd.DataFrame.from_dict(model_scores, orient='index')
    results_df = results_df[[
        'test_accuracy', 'precision', 'recall',
        'f1', 'auc', 'cv_mean_score'
    ]]
    results_df.columns = ['准确率', '精确率', '召回率', 'F1分数', 'AUC值', '交叉验证平均分']
    print(results_df.round(4))
    results_df.to_csv('model_performance_comparison.csv', index_label='模型')

    # 最佳模型分析
    best_model_name = max(model_scores, key=lambda k: model_scores[k]['cv_mean_score'])
    best_model = best_models[best_model_name]
    print(f"\n最佳模型: {best_model_name}")

    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix(y_test, best_model.predict(X_test)),
        annot=True, fmt='d', cmap='Blues'
    )
    plt.title(f'{best_model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'confusion_matrix_{best_model_name}.png', bbox_inches='tight')
    plt.close()

    return best_model, best_model_name, model_scores

#绘制各模型混淆矩阵
def plot_all_models_confusion_matrix(model_scores, best_model_name, y_test_all, y_pred_all):

    for model_name in model_scores.keys():
        # 从存储的预测结果中获取对应模型的测试集结果
        y_test = y_test_all[model_name]
        y_pred = y_pred_all[model_name]

        # 调用visual模块中的混淆矩阵绘制函数
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            save_path=f'confusion_matrix_{model_name}.png'
        )