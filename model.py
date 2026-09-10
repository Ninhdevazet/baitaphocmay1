import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Đọc dữ liệu
df = pd.read_csv('data.csv')

# 2. Tách biến độc lập (X) và biến phụ thuộc (y - Giá nhà)
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# 3. Huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

print("Đã huấn luyện xong mô hình Hồi quy tuyến tính!")
# Dự đoán thử 1 căn nhà: 3000 m2, 3 phòng ngủ, 10 năm tuổi
print("Dự đoán giá:", model.predict([[3000, 3, 10]]))