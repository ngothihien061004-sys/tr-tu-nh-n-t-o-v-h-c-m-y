import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# 1. TẠO DATA (nếu file ít dòng)
# ======================
np.random.seed(42)

dates = pd.date_range(start='2022-01-01', periods=200)

df = pd.DataFrame({
    'Date': dates,
    'Quantity': np.random.randint(1, 10, size=200),
    'Unit_Price': np.random.randint(50, 200, size=200),
    'Promotion': np.random.randint(0, 2, size=200),
    'Holiday': np.random.randint(0, 2, size=200),
    'Category': np.random.choice(['A', 'B', 'C'], size=200),
    'Store': np.random.choice(['S1', 'S2'], size=200),
})

# Revenue = Quantity * Price + noise
df['Revenue'] = df['Quantity'] * df['Unit_Price'] + np.random.normal(0, 50, 200)

# ======================
# 2. XỬ LÝ DỮ LIỆU
# ======================
df = df.sort_values(by='Date')

# Feature time
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Encode categorical
df = pd.get_dummies(df, columns=['Category', 'Store'], drop_first=True)

# ======================
# 3. TẠO X, y
# ======================
X = df.drop(['Revenue', 'Date'], axis=1)
y = df['Revenue']

# ======================
# 4. CHIA DỮ LIỆU (80% train - 20% test)
# ======================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print(f"Số mẫu train: {len(X_train)}")
print(f"Số mẫu test: {len(X_test)}")

if len(X_test) < 2:
    print("❌ Cảnh báo: Dữ liệu test quá ít, không thể tính R2 và vẽ chuẩn")

# ======================
# 5. TRAIN MODEL
# ======================
model = LinearRegression()
model.fit(X_train, y_train)

# ======================
# 6. DỰ ĐOÁN
# ======================
y_pred = model.predict(X_test)

# ======================
# 7. ĐÁNH GIÁ
# ======================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred) if len(X_test) > 1 else float('nan')

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 2) if not np.isnan(r2) else "Không tính được với dữ liệu test ít")

# ======================
# 8. VẼ BIỂU ĐỒ CHUẨN (RA ĐƯỜNG)
# ======================
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Thực tế', linewidth=2)
plt.plot(y_pred, label='Dự báo', linestyle='--', linewidth=2)

plt.title('So sánh doanh thu thực tế và dự báo')
plt.xlabel('Time Index')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()