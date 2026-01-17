import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import json
import joblib
from datetime import datetime
from pyod.models.ocsvm import OCSVM

warnings.filterwarnings('ignore')

# ---------------------- 第一步：触摸数据预处理（分配触摸ID） ----------------------
def process_touch_data(file_path, t):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 重命名列名（已修正逗号缺失问题）
    df.columns = ['事件类型', '手指位置 x 坐标', '手指位置 y 坐标', '时间戳（毫秒）', '压力', '接触面积', '滑动方向',
                  '加速度 x', '加速度 y', '加速度 z', '角速度 x', '角速度 y', '角速度 z', '磁力计 x', '磁力计 y',
                  '磁力计 z']

    touch_id = 1
    touch_data = []
    current_touch = []  # 存储当前触摸动作的所有记录（按下/滑动/抬起）
    has_press = False  # 是否已收到“按下”
    has_release = False  # 是否已收到“抬起”
    has_slide = False  # 是否已收到“滑动中”

    # 定义有效触摸事件（排除无触摸）
    valid_events = ['按下', '滑动中', '抬起']

    for index, row in df.iterrows():
        event_type = row['事件类型']

        if event_type == '按下':
            if not has_press:
                # 首次按下：初始化当前触摸动作
                has_press = True
                current_touch = [row]
            else:
                # 连续按下：直接添加到当前触摸动作
                current_touch.append(row)

        elif event_type == '滑动中':
            if has_press:  # 只有已按下时，滑动才有效
                has_slide = True
                current_touch.append(row)  # 保留所有滑动记录

        elif event_type == '抬起':
            if has_press:  # 只有已按下时，抬起才有效
                has_release = True
                current_touch.append(row)  # 保留所有抬起记录

        else:  # 非触摸事件（如“无触摸”）
            if has_press and has_release and has_slide:
                # 若已完成一个完整动作（按下→滑动→抬起），则分配触摸ID
                start_time = current_touch[0]['时间戳（毫秒）'] - t
                end_time = current_touch[-1]['时间戳（毫秒）'] + t
                # 提取该动作的所有数据：时间范围+有效事件（排除无触摸）
                touch_df = df[(df['时间戳（毫秒）'] >= start_time) &
                              (df['时间戳（毫秒）'] <= end_time) &
                              (df['事件类型'].isin(valid_events))].copy()
                touch_df['触摸 ID'] = touch_id
                touch_data.append(touch_df)
                touch_id += 1
                # 重置状态，准备下一个动作
                has_press = has_release = has_slide = False
                current_touch = []

    # 处理最后一个完整动作（循环结束后可能残留）
    if has_press and has_release and has_slide and current_touch:
        start_time = current_touch[0]['时间戳（毫秒）'] - t
        end_time = current_touch[-1]['时间戳（毫秒）'] + t
        # 同样添加有效事件筛选，排除无触摸
        touch_df = df[(df['时间戳（毫秒）'] >= start_time) &
                      (df['时间戳（毫秒）'] <= end_time) &
                      (df['事件类型'].isin(valid_events))].copy()
        touch_df['触摸 ID'] = touch_id
        touch_data.append(touch_df)

    # 合并所有触摸动作数据
    return pd.concat(touch_data, ignore_index=True) if touch_data else pd.DataFrame()

# ---------------------- 第二步：特征提取函数 ----------------------
def extract_time_domain_features(data):
    """提取时域特征：均值、最小值、最大值、方差、复杂度"""
    if len(data) == 0:
        return [np.nan] * 5
    mean = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)
    var = np.var(data)
    complexity = np.sqrt(np.sum((np.diff(data) ** 2))) if len(data) > 1 else 0
    return [mean, min_val, max_val, var, complexity]


def extract_position_features(touch_core_data):
    """提取位置特征：起始点(x,y)、结束点(x,y)"""
    if len(touch_core_data) == 0:
        return [np.nan] * 4
    start_x = touch_core_data.iloc[0]['手指位置 x 坐标']
    start_y = touch_core_data.iloc[0]['手指位置 y 坐标']
    end_x = touch_core_data.iloc[-1]['手指位置 x 坐标']
    end_y = touch_core_data.iloc[-1]['手指位置 y 坐标']
    return [start_x, start_y, end_x, end_y]


def extract_length_features(touch_core_data):
    """提取长度特征：位移长度、移动长度、比值"""
    if len(touch_core_data) < 2:
        return [np.nan, np.nan, np.nan]

    # 位移长度（起始到结束的欧氏距离）
    start_x, start_y = touch_core_data.iloc[0]['手指位置 x 坐标'], touch_core_data.iloc[0]['手指位置 y 坐标']
    end_x, end_y = touch_core_data.iloc[-1]['手指位置 x 坐标'], touch_core_data.iloc[-1]['手指位置 y 坐标']
    displacement = math.hypot(end_x - start_x, end_y - start_y)

    # 移动长度（相邻点欧氏距离之和）
    move_length = 0.0
    for i in range(len(touch_core_data) - 1):
        x1, y1 = touch_core_data.iloc[i]['手指位置 x 坐标'], touch_core_data.iloc[i]['手指位置 y 坐标']
        x2, y2 = touch_core_data.iloc[i + 1]['手指位置 x 坐标'], touch_core_data.iloc[i + 1]['手指位置 y 坐标']
        move_length += math.hypot(x2 - x1, y2 - y1)

    # 比值（避免除零）
    ratio = displacement / move_length if move_length != 0 else 0
    return [displacement, move_length, ratio]


def extract_angle_features(touch_core_data):
    """提取角度特征：两个角度序列的统计值"""
    if len(touch_core_data) < 1:
        return [np.nan] * 8  # 两个序列各4个统计值

    x_coords = touch_core_data['手指位置 x 坐标'].values
    y_coords = touch_core_data['手指位置 y 坐标'].values
    n = len(touch_core_data)

    # 角度序列1：每个点到原点（屏幕左下角）的连线与x轴夹角（弧度）
    angle1 = [math.atan2(y, x) for x, y in zip(x_coords, y_coords)]

    # 角度序列2：相邻点连线与x轴夹角（弧度）
    angle2 = []
    for i in range(n - 1):
        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]
        angle2.append(math.atan2(dy, dx))

    # 计算统计值（处理空序列）
    def get_angle_stats(angle_seq):
        if len(angle_seq) == 0:
            return [np.nan] * 4
        return [np.mean(angle_seq), np.max(angle_seq), np.min(angle_seq), np.var(angle_seq)]

    stats1 = get_angle_stats(angle1)
    stats2 = get_angle_stats(angle2) if len(angle2) > 0 else [np.nan] * 4
    return stats1 + stats2


def extract_time_features(tstart, tend):
    """提取时间特征：持续时间（tend - tstart）"""
    if pd.isna(tstart) or pd.isna(tend):
        return [np.nan]
    duration = tend - tstart
    return [duration]


def extract_speed_features(touch_core_data):
    """提取速度特征：相邻点速度序列的统计值"""
    if len(touch_core_data) < 2:
        return [np.nan] * 4

    times = touch_core_data['时间戳（毫秒）'].values
    x_coords = touch_core_data['手指位置 x 坐标'].values
    y_coords = touch_core_data['手指位置 y 坐标'].values
    n = len(touch_core_data)

    speeds = []
    for i in range(n - 1):
        dt = times[i + 1] - times[i]  # 时间差（毫秒）
        if dt == 0:  # 避免除零
            continue
        dx = x_coords[i + 1] - x_coords[i]
        dy = y_coords[i + 1] - y_coords[i]
        distance = math.hypot(dx, dy)
        speed = distance / dt  # 速度（单位：坐标单位/毫秒）
        speeds.append(speed)

    if len(speeds) == 0:
        return [np.nan] * 4
    return [np.mean(speeds), np.max(speeds), np.min(speeds), np.var(speeds)]


def extract_pressure_features(touch_core_data):
    """提取压力特征：压力序列的统计值"""
    if len(touch_core_data) == 0:
        return [np.nan] * 4
    pressure = touch_core_data['压力'].values
    return [np.mean(pressure), np.max(pressure), np.min(pressure), np.var(pressure)]


def extract_area_features(touch_core_data):
    """提取接触面积特征：面积序列的统计值"""
    if len(touch_core_data) == 0:
        return [np.nan] * 4
    area = touch_core_data['接触面积'].values
    return [np.mean(area), np.max(area), np.min(area), np.var(area)]


def extract_direction_features(touch_core_data):
    """提取方向特征：仅保留移动角度（弧度），移除方向类别"""
    if len(touch_core_data) < 2:
        return [np.nan]

    start_x, start_y = touch_core_data.iloc[0]['手指位置 x 坐标'], touch_core_data.iloc[0]['手指位置 y 坐标']
    end_x, end_y = touch_core_data.iloc[-1]['手指位置 x 坐标'], touch_core_data.iloc[-1]['手指位置 y 坐标']

    dx = end_x - start_x
    dy = end_y - start_y

    # 计算角度（与x轴夹角，弧度）
    if dx == 0 and dy == 0:  # 无移动
        angle = 0.0
    else:
        angle = math.atan2(dy, dx)  # 范围：[-π, π]
        # 转换为[0, 2π)
        angle = angle if angle >= 0 else angle + 2 * math.pi
    return [angle]  # 仅返回角度，移除方向字符串


# ---------------------- 第三步：主处理函数（合并预处理和特征提取） ----------------------
def process_touch_and_extract_features(raw_file_path, t):
    # 1. 预处理原始数据，分配触摸ID（使用t作为预处理时间扩展）
    touch_id_df = process_touch_data(raw_file_path, t)
    if touch_id_df.empty:
        print("没有检测到完整的触摸动作，无法提取特征")
        return pd.DataFrame()

    # 2. 确保数据按触摸ID和时间戳排序
    touch_id_df = touch_id_df.sort_values(by=['触摸 ID', '时间戳（毫秒）']).reset_index(drop=True)

    all_features = []
    sample_counts = []  # 存储每个触摸ID的采样个数（备用，主要在循环内直接添加到特征）
    # 获取所有唯一的触摸ID
    touch_ids = touch_id_df['触摸 ID'].unique()

    for touch_id in touch_ids:
        # 3. 获取当前触摸ID的所有数据（包含前后扩展数据）
        touch_all_data = touch_id_df[touch_id_df['触摸 ID'] == touch_id].copy()
        if len(touch_all_data) < 1:
            continue

        # 4. 确定tstart（第一次按下时间）和tend（最后一次抬起时间）
        press_events = touch_all_data[touch_all_data['事件类型'] == '按下']
        release_events = touch_all_data[touch_all_data['事件类型'] == '抬起']

        if len(press_events) == 0 or len(release_events) == 0:
            continue  # 跳过没有完整按下/抬起的触摸动作

        tstart = press_events['时间戳（毫秒）'].min()
        tend = release_events['时间戳（毫秒）'].max()

        # 5. 提取触摸核心数据（tstart到tend范围内）
        touch_core_data = touch_all_data[
            (touch_all_data['时间戳（毫秒）'] >= tstart) &
            (touch_all_data['时间戳（毫秒）'] <= tend)
            ].copy()

        if len(touch_core_data) < 1:
            continue

        # 记录当前触摸ID的采样个数（原始核心数据行数，无去重）
        sample_count = len(touch_core_data)
        sample_counts.append(sample_count)

        # 6. 提取各类特征
        pos_feats = extract_position_features(touch_core_data)
        len_feats = extract_length_features(touch_core_data)
        angle_feats = extract_angle_features(touch_core_data)
        time_feats = extract_time_features(tstart, tend)
        speed_feats = extract_speed_features(touch_core_data)
        pressure_feats = extract_pressure_features(touch_core_data)
        area_feats = extract_area_features(touch_core_data)
        dir_feats = extract_direction_features(touch_core_data)  # 仅含角度，无方向

        # 合并触摸信息特征（保留 sample_count 到特征列表）
        touch_feats = [touch_id, sample_count] + pos_feats + len_feats + angle_feats + time_feats + \
                      speed_feats + pressure_feats + area_feats + dir_feats

        # 7. 提取传感器特征（仅时域，无频域；使用t作为特征提取的时间扩展）
        sensor_start = tstart - t
        sensor_end = tend + t
        sensor_data = touch_id_df[
            (touch_id_df['时间戳（毫秒）'] >= sensor_start) &
            (touch_id_df['时间戳（毫秒）'] <= sensor_end)
            ].copy()

        # 传感器列表
        sensors = [
            ('加速度 x', sensor_data['加速度 x'].values),
            ('加速度 y', sensor_data['加速度 y'].values),
            ('加速度 z', sensor_data['加速度 z'].values),
            ('角速度 x', sensor_data['角速度 x'].values),
            ('角速度 y', sensor_data['角速度 y'].values),
            ('角速度 z', sensor_data['角速度 z'].values),
            ('磁力计 x', sensor_data['磁力计 x'].values),
            ('磁力计 y', sensor_data['磁力计 y'].values),
            ('磁力计 z', sensor_data['磁力计 z'].values)
        ]

        sensor_feats = []
        for name, data in sensors:
            time_feats = extract_time_domain_features(data)
            sensor_feats.extend(time_feats)  # 仅添加时域特征，移除频域强度

        # 8. 合并所有特征
        all_features.append(touch_feats + sensor_feats)

    # 9. 定义特征列名（保留"采样个数"列）
    touch_columns = [
        '触摸 ID',
        '采样个数',  # 保留列：每个触摸动作的原始核心数据条数（无去重）
        # 位置特征
        '起始x', '起始y', '结束x', '结束y',
        # 长度特征
        '位移长度', '移动长度', '位移/移动比值',
        # 角度特征
        '角度1_均值', '角度1_最大值', '角度1_最小值', '角度1_方差',
        '角度2_均值', '角度2_最大值', '角度2_最小值', '角度2_方差',
        # 时间特征
        '持续时间（毫秒）',
        # 速度特征
        '速度_均值', '速度_最大值', '速度_最小值', '速度_方差',
        # 压力特征
        '压力_均值', '压力_最大值', '压力_最小值', '压力_方差',
        # 面积特征
        '面积_均值', '面积_最大值', '面积_最小值', '面积_方差',
        # 方向特征（仅保留角度）
        '移动角度（弧度）'
    ]

    # 传感器特征列名（仅时域，无频域强度）
    sensor_columns = []
    sensor_names = ['加速度 x', '加速度 y', '加速度 z', '角速度 x', '角速度 y', '角速度 z', '磁力计 x', '磁力计 y',
                    '磁力计 z']
    time_domain_names = ['均值', '最小值', '最大值', '方差', '复杂度']
    for sensor in sensor_names:
        for td_name in time_domain_names:
            sensor_columns.append(f'{sensor}_{td_name}')

    all_columns = touch_columns + sensor_columns

    # 10. 生成特征DataFrame
    result_df = pd.DataFrame(all_features, columns=all_columns)
    # 附加采样个数列表（方便后续单独调用）
    result_df.attrs['sample_counts'] = sample_counts
    return result_df


# ---------------------- 数据预处理（保留核心逻辑，简化输出） ----------------------
def preprocess_data(pos_file_path, neg_file_path, test_pos_ratio=0.2, random_state=None):
    NON_FEATURE_COLS = ['触摸 ID', '采样个数', '标签']

    # 读取数据并标记标签
    df_pos = pd.read_csv(pos_file_path, encoding='utf-8-sig')
    df_pos['标签'] = 0  # 正常样本
    df_neg = pd.read_csv(neg_file_path, encoding='utf-8-sig')
    df_neg['标签'] = 1  # 异常样本

    # 提取特征列（74维）
    feature_cols = [col for col in df_pos.columns if col not in NON_FEATURE_COLS]
    if len(feature_cols) != 74:
        print(f"⚠️ 特征数量为{len(feature_cols)}维（预期74维）")

    # 划分训练集（仅正样本用于训练）和测试集（正样本测试+负样本）
    X_pos = df_pos[feature_cols]
    X_pos_train, X_pos_test, _, y_pos_test = train_test_split(
        X_pos, df_pos['标签'], test_size=test_pos_ratio, random_state=random_state
    )
    X_neg_test = df_neg[feature_cols]
    X_test = pd.concat([X_pos_test, X_neg_test], ignore_index=True)
    y_test = np.concatenate([y_pos_test, df_neg['标签'].values], axis=0)

    # 缺失值填充（如果需要）
    imputer = SimpleImputer(strategy='mean')
    X_pos_train_filled = imputer.fit_transform(X_pos_train)
    X_test_filled = imputer.transform(X_test)

    # 标准化（核心：训练集拟合，后续预测必须用相同参数）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_pos_train_filled)  # 训练数据
    X_test_scaled = scaler.transform(X_test_filled)  # 测试数据

    print(f"✅ 预处理完成：训练集{X_train_scaled.shape}，测试集{X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_test, scaler, imputer, feature_cols


# ---------------------- 模型调优（保留，可选） ----------------------
def tune_ocsvm_hyperparams(X_train):
    param_grid = {'nu': [ 0.005,0.01,0.05,0.15,0.1,0.2], 'gamma': [0.01,0.05,0.1,1.0,2.0,3.0]}
    ocsvm = OCSVM(kernel='rbf')

    def unsupervised_score(estimator, X):
        scores = estimator.decision_function(X)
        pseudo_y = np.zeros(len(scores))
        pseudo_y[np.argsort(scores)[-int(0.05 * len(scores)):]] = 1
        return roc_auc_score(pseudo_y, scores)

    grid = GridSearchCV(ocsvm, param_grid, cv=3, scoring=unsupervised_score, n_jobs=-1)
    grid.fit(X_train)
    print(f"最优参数：{grid.best_params_}")
    return grid.best_params_
#  C:\Users\Lenovo\Desktop\2048_AllData_20251110_174244.csv

# ---------------------- 模型训练（直接用pyod.OCSVM） ---------------------- 
def train_ocsvm(X_train, best_params=None):
    if best_params:
        ocsvm = OCSVM(**best_params, kernel='rbf')
    else:
        ocsvm = OCSVM(nu=0.05, gamma=0.01, kernel='rbf')  # 默认参数
    ocsvm.fit(X_train)
    print("✅ OCSVM模型训练完成")
    return ocsvm
# 0.1,0.01 可以

# ---------------------- 模型评估（简化） ----------------------
def evaluate_ocsvm(model, X_test, y_test):
    scores = model.decision_function(X_test)
    y_pred = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, scores)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统用 SimHei（黑体）
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 输出详细评估信息
    normal_count = np.sum(y_test == 0)
    anomaly_count = np.sum(y_test == 1)
    
    normal_correct = cm[0, 0]  # 正常样本被正确识别
    normal_wrong = cm[0, 1]    # 正常样本被错误识别为异常
    anomaly_wrong = cm[1, 0]   # 异常样本被错误识别为正常
    anomaly_correct = cm[1, 1] # 异常样本被正确识别
    
    print(f"评估结果：AUC-ROC={auc_roc:.4f}")
    print(f"\n测试集详情:")
    print(f"  正常样本: {normal_count}个")
    print(f"  异常样本: {anomaly_count}个")
    print(f"\n识别结果:")
    print(f"  正常样本识别正确: {normal_correct}个 ({normal_correct/normal_count*100:.1f}%)")
    print(f"  正常样本识别错误: {normal_wrong}个 ({normal_wrong/normal_count*100:.1f}%)")
    print(f"  异常样本识别错误: {anomaly_wrong}个 ({anomaly_wrong/anomaly_count*100:.1f}%)")
    print(f"  异常样本识别正确: {anomaly_correct}个 ({anomaly_correct/anomaly_count*100:.1f}%)")
    
    # 添加统计信息
    normal_scores = scores[y_test == 0]
    anomaly_scores = scores[y_test == 1]
    
    print(f"\n分数统计信息:")
    print(f"  正常样本分数范围: [{np.min(normal_scores):.4f}, {np.max(normal_scores):.4f}]")
    print(f"  正常样本分数均值: {np.mean(normal_scores):.4f}")
    print(f"  异常样本分数范围: [{np.min(anomaly_scores):.4f}, {np.max(anomaly_scores):.4f}]")
    print(f"  异常样本分数均值: {np.mean(anomaly_scores):.4f}")
    
    # 显示决策阈值信息
    print(f"\nOne-Class SVM 决策机制:")
    print(f"  默认决策阈值: 0")
    print(f"  分数 > 0: 预测为正常样本")
    print(f"  分数 <= 0: 预测为异常样本")
    print(f"  |分数| 越大: 距离决策边界越远，分类置信度越高")
    print(f"  注意：OCSVM 根据分数的正负进行分类，而不是绝对值大小")
    print(f"  注意：异常样本不一定有最低分数，它们可能分布在决策边界的任一侧")
    
    # 显示模型的决策阈值
    print(f"\n模型决策阈值详情:")
    print(f"  OCSVM 内置参数阈值: {model.threshold_:.6f}")
    print(f"  用于预测的决策阈值: 0 (固定值)")
    print(f"  说明：OCSVM 在 predict() 方法中使用 0 作为阈值进行分类")
    print(f"  原因：0 是数学上标准的决策边界，表示样本到超平面的距离为0")
    
    # 解释两种阈值的区别
    print(f"\n阈值概念解释:")
    print(f"  内置阈值 ({model.threshold_:.6f}): 模型内部参数，用于计算决策函数")
    print(f"  预测阈值 (0): 标准分类边界，sklearn 固定使用此值进行分类")
    print(f"  无论内置阈值为何值，预测始终以 0 为界进行分类")
    
    print(f"\n当前数据分布:")
    print(f"  正常样本中分数 > 0 的比例: {np.sum(normal_scores > 0) / len(normal_scores) * 100:.1f}%")
    print(f"  异常样本中分数 <= 0 的比例: {np.sum(anomaly_scores <= 0) / len(anomaly_scores) * 100:.1f}%")
    
    # 显示决策边界解释
    print(f"\n决策边界解释:")
    print(f"  分数 = 0: 决策边界（超平面）")
    print(f"  分数 > 0: 在正常样本区域一侧")
    print(f"  分数 < 0: 在正常样本区域另一侧")
    print(f"  |分数| 越大: 距离决策边界越远，置信度越高")
    
    # 计算最优决策阈值（基于约登指数）
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    youden_j = tpr - fpr
    optimal_threshold_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_idx]
    
    print(f"\n最优决策阈值分析:")
    print(f"  基于约登指数的最优阈值: {optimal_threshold:.4f}")
    print(f"  对应的TPR: {tpr[optimal_threshold_idx]:.4f}")
    print(f"  对应的FPR: {fpr[optimal_threshold_idx]:.4f}")
    
    # 添加异常分数可视化
    # 分离正常和异常样本的分数
    normal_scores = scores[y_test == 0]
    anomaly_scores = scores[y_test == 1]
    
    # 创建正常样本分数直方图
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.7, label='正常样本', color='blue')
    plt.xlabel('异常分数')
    plt.ylabel('样本数量')
    plt.title('正常样本的异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 创建异常样本分数直方图
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='异常样本', color='red')
    plt.xlabel('异常分数')
    plt.ylabel('样本数量')
    plt.title('异常样本的异常分数分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 创建对比直方图
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.7, label='正常样本', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='异常样本', color='red')
    plt.xlabel('异常分数')
    plt.ylabel('样本数量')
    plt.title('正常与异常样本的异常分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 创建箱线图
    plt.figure(figsize=(8, 6))
    plt.boxplot([normal_scores, anomaly_scores], labels=['正常样本', '异常样本'])
    plt.ylabel('异常分数')
    plt.title('正常与异常样本的异常分数箱线图')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 创建散点图显示分数分布
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(normal_scores)), normal_scores, alpha=0.7, label='正常样本', color='blue')
    plt.scatter(range(len(normal_scores), len(normal_scores) + len(anomaly_scores)), anomaly_scores, alpha=0.7, label='异常样本', color='red')
    plt.xlabel('样本索引')
    plt.ylabel('异常分数')
    plt.title('正常与异常样本的异常分数散点图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return auc_roc, cm


# ---------------------- 保存模型和标准化参数（核心：完全贴合你的示例） ----------------------
def save_model_and_scaler(ocsvm_model, scaler, model_path="ocsvm_model.pkl", scaler_path="scaler_params.json"):
    # 1. 直接保存完整OCSVM模型（含所有参数，无需提取）
    joblib.dump(ocsvm_model, model_path)
    print(f"✅ 模型已保存到：{model_path}")

    # 2. 保存标准化器的均值和方差（供Android Studio用，和你给的示例一致）
    scaler_params = {
        "mean": scaler.mean_.tolist(),  # 均值
        #"var": scaler.var_.tolist()  # 方差（也可用scale_=标准差，根据需要选）
        "scale": scaler.scale_.tolist()  # 标准差（如果Android需要用标准差，解开这行）
    }
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, ensure_ascii=False, indent=2)
    print(f"✅ 标准化参数已保存到：{scaler_path}")


# ---------------------- 导出模型参数供Java使用 ----------------------
def export_model_for_java(ocsvm_model, output_path="ocsvm_params.json"):
    """
    导出OCSVM模型参数为JSON格式，供Java使用
    """
    model_params = {
        "support_vectors": ocsvm_model.support_vectors_.tolist(),
        "dual_coef": ocsvm_model.dual_coef_.tolist(),
        "gamma": float(ocsvm_model.gamma),
        "intercept": ocsvm_model.intercept_.tolist(),
        "kernel": ocsvm_model.kernel,
        "n_features": ocsvm_model.support_vectors_.shape[1],
        "n_support_vectors": ocsvm_model.support_vectors_.shape[0]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_params, f, ensure_ascii=False, indent=2)
    print(f"✅ 模型参数已导出为JSON格式：{output_path}")


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    print("===== 触摸数据特征提取和OCSVM训练工具 =====")
    
    # # 获取正样本数据路径
    # while True:
    #     pos_raw_file_path = input("请输入正样本原始数据CSV文件路径（示例：C://Users//Lenovo//Desktop//2048_AllData_20251110_164941.csv）：").strip()
    #     if not pos_raw_file_path:
    #         print("路径不能为空，请重新输入！")
    #         continue
    #     if not pos_raw_file_path.endswith(".csv"):
    #         print("请输入CSV格式的文件路径（以.csv结尾）！")
    #         continue
    #     break
    pos_raw_file_path = r"C://Users//Lenovo//Desktop//2048_AllData_20260109_144408.csv"
    # 获取负样本数据路径
    # while True:
    #     neg_raw_file_path = input("请输入负样本原始数据CSV文件路径（示例：C://Users//Lenovo//Desktop//2048_AllData_20251110_171538.csv）：").strip()
    #     if not neg_raw_file_path:
    #         print("路径不能为空，请重新输入！")
    #         continue
    #     if not neg_raw_file_path.endswith(".csv"):
    #         print("请输入CSV格式的文件路径（以.csv结尾）！")
    #         continue
    #     break

    neg_raw_file_path=r"C://Users//Lenovo//Desktop//2048_AllData_20251110_171538.csv"

    # 设置时间扩展量t为0
    t = 0
    
    # 设置输出路径
    pos_output_path = r"C:\Users\Lenovo\Desktop\dataset_all_features.csv"
    neg_output_path = r"C:\Users\Lenovo\Desktop\dataset_all_features_fu.csv"
    
    print(f"\n开始处理正样本数据（时间扩展量：{t}ms）...")
    pos_feature_df = process_touch_and_extract_features(pos_raw_file_path, t)
    
    # 保存正样本特征结果
    if not pos_feature_df.empty:
        pos_feature_df.to_csv(pos_output_path, index=False, encoding='utf-8-sig')
        print(f"\n正样本特征提取完成！")
        print(f"保存路径：{pos_output_path}")
        print(f"共提取 {len(pos_feature_df)} 个完整触摸动作的特征")
        print(f"特征总数：{len(pos_feature_df.columns)} 个")
    else:
        print("\n正样本特征提取失败，未生成有效数据")
        exit()
    
    print(f"\n开始处理负样本数据（时间扩展量：{t}ms）...")
    neg_feature_df = process_touch_and_extract_features(neg_raw_file_path, t)
    
    # 保存负样本特征结果
    if not neg_feature_df.empty:
        neg_feature_df.to_csv(neg_output_path, index=False, encoding='utf-8-sig')
        print(f"\n负样本特征提取完成！")
        print(f"保存路径：{neg_output_path}")
        print(f"共提取 {len(neg_feature_df)} 个完整触摸动作的特征")
        print(f"特征总数：{len(neg_feature_df.columns)} 个")
    else:
        print("\n负样本特征提取失败，未生成有效数据")
        exit()
        
    # 数据路径
    POS_DATA = pos_output_path
    NEG_DATA = neg_output_path

    # 步骤1：预处理
    X_train, X_test, y_test, scaler, imputer, feature_cols = preprocess_data(POS_DATA, NEG_DATA)

    # 步骤2：调优（可选，设为False则用默认参数）
    DO_TUNE = False
    best_params = tune_ocsvm_hyperparams(X_train) if DO_TUNE else None

    # 步骤3：训练模型
    ocsvm_model = train_ocsvm(X_train, best_params)

    # 步骤4：评估
    evaluate_ocsvm(ocsvm_model, X_test, y_test)

    # 步骤5：保存（核心：完全按你的示例方式保存）
    save_model_and_scaler(ocsvm_model, scaler)
    
    # 步骤6：导出模型参数供Java使用
    export_model_for_java(ocsvm_model)