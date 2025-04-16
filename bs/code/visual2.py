from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#各模型混淆矩阵绘制
def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'混淆矩阵 - {model_name}')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")

    plt.show()
    plt.close()

#各模型性能对比图
def plot_model_performance_comparison(model_scores, metric='accuracy', save_path=None):

    metrics = {}

    if metric == 'accuracy':
        for model_name, scores in model_scores.items():
            metrics[model_name] = scores['test_accuracy']
    elif metric == 'cv_score':
        for model_name, scores in model_scores.items():
            metrics[model_name] = scores['cv_mean_score']

    # 转换为DataFrame
    df = pd.DataFrame(list(metrics.items()), columns=['模型', metric])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='模型', y=metric, data=df)

    # 添加数值标签
    for i, v in enumerate(df[metric]):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.title(f'模型性能对比 - {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型性能对比图已保存至: {save_path}")

    plt.show()
    plt.close()

#各模型ROC-AUC图
def plot_roc_curves(models, X_test, y_test, save_path=None):

    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            # 获取预测概率
            try:
                y_score = model.predict_proba(X_test)

                # 多分类情况下需要处理
                if y_score.shape[1] > 2:  # 多分类情况
                    # 为每个类别计算ROC曲线和AUC
                    n_classes = y_score.shape[1]

                    # 一对多方式计算ROC
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2,
                                 label=f'{model_name} (类别 {i}, AUC = {roc_auc:.4f})')
                else:  # 二分类情况
                    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2,
                             label=f'{model_name} (AUC = {roc_auc:.4f})')
            except Exception as e:
                print(f"计算 {model_name} 的ROC曲线时出错: {e}")

    # 绘制随机猜测的参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('多模型ROC曲线对比')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")

    plt.show()
    plt.close()

#聚类可视化
def visualize_clusters(df, clusters, n_clusters, feature_names=None, save_path=None):

    # 使用PCA降维到2D进行可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)

    # 创建包含聚类标签的DataFrame
    df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    df_pca['cluster'] = clusters

    # 绘制聚类散点图
    plt.figure(figsize=(12, 10))

    # 使用更丰富的颜色映射
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        plt.scatter(
            df_pca.loc[df_pca['cluster'] == i, 'PC1'],
            df_pca.loc[df_pca['cluster'] == i, 'PC2'],
            s=100,
            c=[colors[i]],
            label=f'聚类 {i}'
        )

    # 获取PCA解释方差比例
    explained_variance = pca.explained_variance_ratio_

    plt.title('聚类结果PCA可视化')
    plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})')
    plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"聚类PCA可视化已保存至: {save_path}")

    plt.show()
    plt.close()

    # 如果特征维度大于2，使用t-SNE进行可视化
    if df.shape[1] > 2:
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, df.shape[0] // 5))
            tsne_results = tsne.fit_transform(df)

            df_tsne = pd.DataFrame(data=tsne_results, columns=['t-SNE1', 't-SNE2'])
            df_tsne['cluster'] = clusters

            plt.figure(figsize=(12, 10))

            for i in range(n_clusters):
                plt.scatter(
                    df_tsne.loc[df_tsne['cluster'] == i, 't-SNE1'],
                    df_tsne.loc[df_tsne['cluster'] == i, 't-SNE2'],
                    s=100,
                    c=[colors[i]],
                    label=f'聚类 {i}'
                )

            plt.title('聚类结果t-SNE可视化')
            plt.xlabel('t-SNE维度1')
            plt.ylabel('t-SNE维度2')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)

            if save_path:
                tsne_save_path = save_path.replace('.png', '_tsne.png')
                plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
                print(f"聚类t-SNE可视化已保存至: {tsne_save_path}")

            plt.show()
            plt.close()
        except Exception as e:
            print(f"t-SNE可视化失败: {e}")

# 聚类特征概览
def plot_cluster_feature_profiles(df, clusters, n_clusters, feature_names=None, save_path=None):

    # 确保df是DataFrame
    if not isinstance(df, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'特征{i}' for i in range(df.shape[1])]
        df = pd.DataFrame(df, columns=feature_names)

    # 如果未指定特征名称，则使用所有特征
    if feature_names is None:
        feature_names = df.columns.tolist()
    else:
        # 确保所有指定的特征都在DataFrame中
        feature_names = [f for f in feature_names if f in df.columns]

    # 添加聚类标签到数据中
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters

    # 计算每个聚类的特征均值
    cluster_means = df_with_clusters.groupby('cluster')[feature_names].mean()

    # 热图显示每个聚类的特征均值
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('聚类特征均值热图')
    plt.ylabel('聚类')
    plt.xlabel('特征')
    plt.tight_layout()

    if save_path:
        heatmap_save_path = save_path.replace('.png', '_heatmap.png')
        plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
        print(f"聚类特征热图已保存至: {heatmap_save_path}")

    plt.show()
    plt.close()

    # 雷达图展示每个聚类的特征概况
    if len(feature_names) >= 3:  # 至少需要3个特征才能绘制雷达图
        # 标准化特征值以便在雷达图上比较
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        cluster_means_scaled = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            index=cluster_means.index,
            columns=cluster_means.columns
        )

        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        for cluster in range(n_clusters):
            values = cluster_means_scaled.iloc[cluster].tolist()
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, linewidth=2, label=f'聚类 {cluster}')
            ax.fill(angles, values, alpha=0.25)

        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)

        plt.title('聚类特征雷达图', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        if save_path:
            radar_save_path = save_path.replace('.png', '_radar.png')
            plt.savefig(radar_save_path, dpi=300, bbox_inches='tight')
            print(f"聚类特征雷达图已保存至: {radar_save_path}")

        plt.show()
        plt.close()

    # 箱型图比较每个聚类中的特征分布
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=feature, data=df_with_clusters)
        plt.title(f'各聚类中 {feature} 的分布')
        plt.xlabel('聚类')
        plt.ylabel(feature)

        if save_path:
            boxplot_save_path = save_path.replace('.png', f'_boxplot_{feature}.png')
            plt.savefig(boxplot_save_path, dpi=300, bbox_inches='tight')
            print(f"特征 {feature} 的箱型图已保存至: {boxplot_save_path}")

        plt.show()
        plt.close()

# 聚类中各特征分布
def plot_feature_distributions_by_cluster(df, clusters, feature_names=None, save_path=None):

    # 确保df是DataFrame
    if not isinstance(df, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'特征{i}' for i in range(df.shape[1])]
        df = pd.DataFrame(df, columns=feature_names)

    # 如果未指定特征名称，则使用所有特征
    if feature_names is None:
        feature_names = df.columns.tolist()
    else:
        # 确保所有指定的特征都在DataFrame中
        feature_names = [f for f in feature_names if f in df.columns]

    # 添加聚类标签到数据中
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters

    # 限制要可视化的特征数量，避免生成过多图表
    if len(feature_names) > 6:
        print(f"特征数量较多({len(feature_names)})，仅展示前6个特征的分布")
        feature_names = feature_names[:6]

    # 计算聚类数量
    n_clusters = len(np.unique(clusters))

    # 为每个特征创建密度图，按聚类分组
    for feature in feature_names:
        plt.figure(figsize=(12, 6))

        for i in range(n_clusters):
            subset = df_with_clusters[df_with_clusters['cluster'] == i]
            sns.kdeplot(subset[feature], label=f'聚类 {i}', fill=True, alpha=0.3)

        plt.title(f'特征 "{feature}" 在各聚类中的分布')
        plt.xlabel(feature)
        plt.ylabel('密度')
        plt.legend()

        if save_path:
            feature_save_path = save_path.replace('.png', f'_dist_{feature}.png')
            plt.savefig(feature_save_path, dpi=300, bbox_inches='tight')
            print(f"特征 {feature} 的分布图已保存至: {feature_save_path}")

        plt.show()
        plt.close()

# 聚类大小分布
def plot_cluster_size_distribution(clusters, save_path=None):

    # 计算每个聚类的样本数量
    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    # 创建聚类大小的饼图
    plt.figure(figsize=(10, 8))
    plt.pie(
        cluster_counts,
        labels=[f'聚类 {i}\n({count} 个样本, {count / len(clusters):.1%})'
                for i, count in cluster_counts.items()],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(cluster_counts)
    )
    plt.title('聚类样本分布')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"聚类大小分布图已保存至: {save_path}")

    plt.show()
    plt.close()

    # 创建聚类大小的条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [f'聚类 {i}' for i in cluster_counts.index],
        cluster_counts.values,
        color=sns.color_palette('viridis', len(cluster_counts))
    )

    # 在条形上方添加标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f'{int(height)}',
            ha='center', va='bottom'
        )

    plt.title('聚类样本数量')
    plt.xlabel('聚类')
    plt.ylabel('样本数量')

    if save_path:
        bar_save_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_save_path, dpi=300, bbox_inches='tight')
        print(f"聚类样本数量条形图已保存至: {bar_save_path}")

    plt.show()
    plt.close()

# 各模型性能对比
def plot_model_validation_comparison(model_scores, save_path=None):

    # 整理数据以便可视化
    models = []
    test_scores = []
    cv_scores = []
    cv_std = []

    for model_name, scores in model_scores.items():
        models.append(model_name)
        test_scores.append(scores['test_accuracy'])
        cv_scores.append(scores['cv_mean_score'])
        cv_std.append(scores['cv_scores'].std())

    # 将数据转换为DataFrame
    df = pd.DataFrame({
        '模型': models,
        '测试集准确率': test_scores,
        '交叉验证平均值': cv_scores,
        '交叉验证标准差': cv_std
    })

    # 绘制测试集和交叉验证性能对比
    plt.figure(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.35

    # 绘制柱状图
    plt.bar(x - width / 2, df['测试集准确率'], width, label='测试集准确率', color='cornflowerblue')
    plt.bar(x + width / 2, df['交叉验证平均值'], width, label='交叉验证平均值', color='lightcoral')

    # 添加误差棒（只对交叉验证结果）
    plt.errorbar(
        x + width / 2,
        df['交叉验证平均值'],
        yerr=df['交叉验证标准差'],
        fmt='o',
        color='black',
        ecolor='black',
        elinewidth=2,
        capsize=5
    )

    # 添加图表元素
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('模型验证性能对比')
    plt.xticks(x, df['模型'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # 添加数值标签
    for i, v in enumerate(df['测试集准确率']):
        plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    for i, v in enumerate(df['交叉验证平均值']):
        plt.text(i + width / 2, v + 0.02, f'{v:.3f}±{df["交叉验证标准差"][i]:.3f}', ha='center', va='bottom')

    plt.ylim(0, max(max(df['测试集准确率']), max(df['交叉验证平均值'])) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型验证性能对比图已保存至: {save_path}")

    plt.show()
    plt.close()

# 绘制学习曲线，评估模型的泛化能力与数据量的关系
def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), save_path=None):

    # 计算学习曲线
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        train_sizes=train_sizes,
        scoring='accuracy'
    )

    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='训练集得分')
    plt.plot(train_sizes_abs, test_mean, 'o-', color='g', label='验证集得分')

    # 填充标准差区域
    plt.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    plt.fill_between(
        train_sizes_abs,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1,
        color='g'
    )

    # 添加图表元素
    plt.title('学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存至: {save_path}")

    plt.show()
    plt.close()