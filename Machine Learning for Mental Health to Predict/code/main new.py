import matplotlib
import os
import train
matplotlib.use('Agg')
from visual2 import (
    visualize_clusters,
    plot_model_performance_comparison,
    plot_cluster_size_distribution,
    plot_model_validation_comparison,
    plot_cluster_feature_profiles)

def main():
    # 数据路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    DATA_PATH = os.path.join(current_dir, '..', 'data', 'students_mental_health_survey.xlsx')
    # 创建文件夹results
    target_dir = os.path.abspath(os.path.join(parent_dir, 'results'))
    os.makedirs(target_dir, exist_ok=True)

    # 数据加载与预处理
    print("=" * 50 + "\nSTEP 1: 数据加载与预处理\n" + "=" * 50)
    raw_df = train.load_and_explore_data(DATA_PATH)
    if raw_df is None:
        return

    cleaned_df = train.clean_data(raw_df)
    encoded_df, _, _ = train.encode_features(cleaned_df)
    scaled_df, _ = train.standardize_data(encoded_df)

    # 聚类分析
    print("\n" + "=" * 50 + "\nSTEP 2: 聚类分析\n" + "=" * 50)
    n_clusters = 3
    clustered_df, kmeans = train.perform_clustering(scaled_df, n_clusters)

    # 模型构建与评估
    print("\n" + "=" * 50 + "\nSTEP 3: 模型训练与优化\n" + "=" * 50)
    best_model, best_model_name, model_results = train.build_prediction_model(clustered_df)

    # 可视化分析
    print("\n" + "=" * 50 + "\nSTEP 4: 可视化输出\n" + "=" * 50)

    # 聚类可视化
    visualize_clusters(
        scaled_df,
        clustered_df['cluster'],
        n_clusters=n_clusters,        feature_names=scaled_df.columns.tolist(),
        save_path='../results/cluster_analysis.png'
    )

    # 聚类分布可视化
    plot_cluster_size_distribution(
        clustered_df['cluster'],
        save_path='../results/cluster_distribution.png'
    )

    # 模型性能对比
    plot_model_performance_comparison(
        model_results,
        metric='accuracy',
        save_path='../results/model_comparison.png'
    )

    # 模型验证曲线
    plot_model_validation_comparison(
        model_results,
        save_path='../results/validation_curve.png'
    )

    # 聚类特征分析
    plot_cluster_feature_profiles(
        scaled_df,
        clustered_df['cluster'],
        n_clusters=n_clusters,
        save_path='results/feature_profiles.png'
    )

    print("\n所有分析结果已保存至results目录")


if __name__ == "__main__":
    main()
