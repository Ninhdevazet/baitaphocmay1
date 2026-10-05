import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

# 1. TẠO DỮ LIỆU
np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 10, axis=0) 
y = np.sin(X).ravel() + np.random.randn(30) * 0.5 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. XÂY DỰNG MÔ HÌNH VÀ CỐ TÌNH GÂY OVERFITTING (Đã thêm StandardScaler)
# Không có regularization -> Sẽ uốn éo theo từng điểm nhiễu
model_overfit = make_pipeline(PolynomialFeatures(degree=15), StandardScaler(), LinearRegression())
model_overfit.fit(X_train, y_train)

# 3. TRÁNH OVERFITTING BẰNG RIDGE REGULARIZATION (Norm 2)
# Dùng Ridge với alpha (tương đương lambda) = 1.0
model_ridge = make_pipeline(PolynomialFeatures(degree=15), StandardScaler(), Ridge(alpha=1.0))
model_ridge.fit(X_train, y_train)

# 4. CHUẨN BỊ DỮ LIỆU VẼ BIỂU ĐỒ
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot_overfit = model_overfit.predict(X_plot)
y_plot_ridge = model_ridge.predict(X_plot)

# 5. VẼ BIỂU ĐỒ
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='red', label='Training set (Dữ liệu học)')
plt.scatter(X_val, y_val, color='yellow', edgecolors='black', marker='s', label='Validation set (Dữ liệu kiểm tra)')

plt.plot(X_plot, y_plot_overfit, color='blue', label='Bị Overfitting (Không Regularization)')
plt.plot(X_plot, y_plot_ridge, color='green', linewidth=2.5, label='Tránh Overfitting (Ridge alpha=1.0)')

plt.ylim(-3, 3)
plt.legend()
plt.title("Minh họa Overfitting và Ridge (Đã chuẩn hóa dữ liệu - Fix lỗi số học)")
plt.show()
