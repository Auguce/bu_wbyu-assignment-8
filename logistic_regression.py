import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

# 设置结果保存路径
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


# 生成椭圆簇数据集，加入偏移逻辑
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    """
    生成两个椭圆形的类簇，其中第二个类簇根据输入的distance参数沿x轴和y轴偏移
    """
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8],
                                  [cluster_std * 0.8, cluster_std]])

    # 生成第一个类簇 (类别0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # 生成第二个类簇 (类别1)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)

    # 偏移第二个类簇
    X2[:, 0] += distance  # 沿x轴偏移
    X2[:, 1] += distance  # 沿y轴偏移
    y2 = np.ones(n_samples)

    # 合并两个类簇数据
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


# 拟合逻辑回归模型并提取系数
def fit_logistic_regression(X, y):
    """
    使用Logistic Regression拟合数据集，并返回模型及其参数
    """
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2


def do_experiments(start, end, step_num):
    """
    执行实验：生成数据集，拟合模型，记录参数并生成结果图像
    """
    # 设置实验参数
    shift_distances = np.linspace(start, end, step_num)  # 偏移距离范围
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}  # 存储用于可视化的示例数据集

    n_samples = 8
    n_cols = 2  # 固定列数
    n_rows = (n_samples + n_cols - 1) // n_cols  # 计算需要的行数
    plt.figure(figsize=(20, n_rows * 10))  # 根据行数调整图形高度

    # 为每个偏移距离运行实验
    for i, distance in enumerate(shift_distances, 1):
        # 生成数据集
        X, y = generate_ellipsoid_clusters(distance=distance)

        # 记录必要信息
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope_list.append(-beta1 / beta2)
        intercept_list.append(-beta0 / beta2)

        # 绘制数据集
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.6)
        decision_boundary_x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        decision_boundary_y = -(beta0 + beta1 * decision_boundary_x) / beta2
        plt.plot(decision_boundary_x, decision_boundary_y, 'k-', label='Decision Boundary')

        # 计算并存储逻辑损失
        from sklearn.metrics import log_loss
        loss = log_loss(y, model.predict_proba(X))
        loss_list.append(loss)

        # 绘制信心水平的渐变轮廓
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)

        # 绘制标题和边距宽度
        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # 绘制参数与偏移距离的关系图
    plt.figure(figsize=(18, 15))
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, label="Beta0")
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, label="Beta1")
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, label="Beta2")
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, label="Slope")
    plt.title("Shift Distance vs Slope")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, label="Intercept")
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, label="Logistic Loss")
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")


if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
