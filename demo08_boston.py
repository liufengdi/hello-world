import numpy as np
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.metrics as sm

boston = sd.load_boston()
print(boston.data.shape, boston.data[0])
print(boston.target.shape, boston.target[0])
print(boston.feature_names)

x, y, header = boston.data, boston.target, boston.feature_names
# 打乱数据集，拆分训练集与测试集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], y[:train_size], y[train_size:]

# 训练决策树模型
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
# 输出测试结果
pred_test_y = model.predict(test_x)
score = sm.r2_score(test_y, pred_test_y)
print(score)
print(sm.mean_absolute_error(test_y, pred_test_y))

