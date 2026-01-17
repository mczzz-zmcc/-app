package com.example.infer;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.work.Data;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkInfo;
import androidx.work.WorkManager;
import androidx.work.WorkerParameters;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class MainActivity extends AppCompatActivity {

    // 所有布局控件
    private Button btnStart, btnStop, btnSave, btnReset, btnDelete, btnInfer;
    private TextView tvData, tvScore, tvInferResult;
    private LinearLayout gameContainer, mainContainer;
    private Spinner spinnerFrequency;

    // 游戏相关
    private GameManager gameManager;
    private int totalScore = 0;
    private float touchStartX = 0f, touchStartY = 0f;
    private static final int MIN_SLIDE_DISTANCE = 50;

    // 身份验证核心变量
    private static final int REQUIRED_TOUCH_COUNT = 3;
    private List<AllDataModel> inferRawData = new ArrayList<>();
    private List<Float> inferScores = new ArrayList<>();
    private OCSVMModel ocsvmModel;
    private boolean isInferring = false;
    private final float ANOMALY_THRESHOLD = 0.5f;
    private Handler collectHandler;
    private Runnable collectTimeoutRunnable;

    // 传感器+频率配置
    private SensorManager sensorManager;
    private Sensor accelerometer, gyroscope, magnetometer;
    private boolean isSensorRegistered = false;
    private final int[] FIXED_HZ_OPTIONS = {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    private List<Integer> supportedHzList = new ArrayList<>();
    private int selectedHz = 50;
    private int sensorDelay;

    // 线程安全与临时数据
    private final ReentrantLock dataLock = new ReentrantLock();
    private volatile MotionEvent currentMotionEvent = null;
    private HandlerThread sensorThread;
    private Handler sensorHandler;
    private String currentTouchType = "无触摸";
    private String currentTouchDirection = "无";
    private float currentPressure = 0f;
    private float currentSize = 0f;
    private float currentTouchX = 0f;
    private float currentTouchY = 0f;

    // 核心配置：触摸时间范围管理
    private final long FILTER_TIME_T = 0;
    private List<long[]> touchIntervals = new ArrayList<>();
    private long currentTouchStart = 0;
    private long currentTouchMaxLiftTime = 0;
    private boolean isCurrentTouchFinished = true;
    private static final long COLLECT_DURATION = 10000;
    private static final long LIFT_WAIT_DELAY = 0;
    private int currentActionId = 0;

    // CSV保存相关（当前会话的特征CSV路径，用于后续读取）
    private String collectSessionId;
    private String rawDataCsvPath;
    private String featureCsvPath; // 原始特征CSV路径
    private String processedFeatureCsvPath; // 处理后（填充+标准化）特征CSV路径
    private static final int BATCH_SAVE_SIZE = 200;
    private boolean isBatchSaving = false;


    // 统一数据模型
    static class AllDataModel {
        String touchType;
        String touchDirection;
        float x, y;
        float pressure;
        float size;
        long time;
        float accX, accY, accZ;
        float gyroX, gyroY, gyroZ;
        float magX, magY, magZ;
        int actionId;

        public AllDataModel(
                String touchType, String touchDirection,
                float x, float y, float pressure, float size,
                long time, int actionId,
                float accX,float  accY, float accZ,
                float gyroX, float gyroY, float gyroZ,
                float magX, float magY, float magZ
        ) {
            this.touchType = touchType;
            this.touchDirection = touchDirection;
            this.x = x;
            this.y = y;
            this.pressure = pressure;
            this.size = size;
            this.time = time;
            this.actionId = actionId;
            this.accX = accX;
            this.accY = accY;
            this.accZ = accZ;
            this.gyroX = gyroX;
            this.gyroY = gyroY;
            this.gyroZ = gyroZ;
            this.magX = magX;
            this.magY = magY;
            this.magZ = magZ;
        }

        public long getTime() {
            return time;
        }

        public String toCsv() {
            return String.format(Locale.ENGLISH, "%s,%f,%f,%d,%f,%f,%s," +
                            "%f,%f,%f,%f,%f,%f,%f,%f,%f",
                    touchType, x, y, time, pressure, size, touchDirection,
                    accX, accY, accZ, gyroX, gyroY, gyroZ, magX, magY, magZ);
        }
    }

    // 74维特征数据模型
    static class FeatureModel {
        String sessionId;
        String touchSessionId;
        long collectTime;
        int sampleCount;
        long startTime;
        long endTime;
        float[] features;

        public FeatureModel(String sessionId, String touchSessionId, long collectTime,
                            int sampleCount, long startTime, long endTime, float[] features) {
            this.sessionId = sessionId;
            this.touchSessionId = touchSessionId;
            this.collectTime = collectTime;
            this.sampleCount = sampleCount;
            this.startTime = startTime;
            this.endTime = endTime;
            this.features = features;
        }

        public String toCsv() {
            StringBuilder sb = new StringBuilder();
            sb.append(sessionId).append(",");
            sb.append(touchSessionId).append(",");
            sb.append(collectTime).append(",");
            sb.append(sampleCount).append(",");
            sb.append(startTime).append(",");
            sb.append(endTime).append(",");
            for (int i = 0; i < features.length; i++) {
                sb.append(features[i]);
                if (i < features.length - 1) {
                    sb.append(",");
                }
            }
            return sb.toString();
        }

        public static String getCsvHeader() {
            StringBuilder sb = new StringBuilder();
            sb.append("会话ID,触摸会话ID,提取时间戳,采样数,采样起始时间(毫秒),采样结束时间(毫秒),");
            String[] featureNames = {
                    "起始x", "起始y", "结束x", "结束y", "位移长度", "移动长度", "位移/移动比值",
                    "角度1_均值", "角度1_最大值", "角度1_最小值", "角度1_方差",
                    "角度2_均值", "角度2_最大值", "角度2_最小值", "角度2_方差",
                    "持续时间（毫秒）",
                    "速度_均值", "速度_最大值", "速度_最小值", "速度_方差",
                    "压力_均值", "压力_最大값", "压力_最小값", "压力_方差",
                    "面积_均值", "面积_最大값", "面积_最小값", "面积_方差",
                    "移动角度（弧度）",
                    "加速度 x_均值", "加速度 x_最小값", "加速度 x_最大값", "加速度 x_方差", "加速度 x_复杂度",
                    "加速度 y_均值", "加速度 y_最小값", "加速度 y_最大값", "加速度 y_方差", "加速度 y_复杂度",
                    "加速度 z_均值", "加速度 z_最小값", "加速度 z_最大값", "加速度 z_方差", "加速度 z_复杂度",
                    "角速度 x_均值", "角速度 x_最小값", "角速度 x_最大값", "角速度 x_方差", "角速度 x_复杂度",
                    "角速度 y_均值", "角速度 y_最小값", "角速度 y_最大값", "角速度 y_方差", "角速度 y_复杂度",
                    "角速度 z_均值", "角速度 z_最小값", "角速度 z_最大값", "角速度 z_方差", "角速度 z_复杂度",
                    "磁力计 x_均值", "磁力计 x_最小값", "磁力计 x_最大값", "磁力计 x_方差", "磁力计 x_复杂度",
                    "磁力计 y_均值", "磁力计 y_最小값", "磁力计 y_最大값", "磁力计 y_方差", "磁力计 y_复杂度",
                    "磁力计 z_均值", "磁力计 z_最小값", "磁力计 z_最大값", "磁力计 z_方差", "磁力计 z_复杂度"
            };
            for (int i = 0; i < 74; i++) {
                sb.append(featureNames[i]);
                if (i < 73) {
                    sb.append(",");
                }
            }
            return sb.toString();
        }
    }

    // 触摸会话类
    class TouchSession {
        private List<AllDataModel> dataList = new ArrayList<>();

        public void addData(AllDataModel data) {
            dataList.add(data);
        }

        public List<AllDataModel> getDataList() {
            return dataList;
        }

        public long getStartTime() {
            return dataList.get(0).getTime();
        }

        public long getEndTime() {
            return dataList.get(dataList.size() - 1).getTime();
        }
    }

    // OCSVM模型类（新增填充均值参数）
    class OCSVMModel {
        private float[][] supportVectors;
        private float[] dualCoef;
        private float intercept;
        private float gamma;
        private float[] scalerMean;
        private float[] scalerStd;
        private float[] imputerMean; // Python训练集的缺失值填充均值

        public void loadParams(android.content.res.AssetManager assetManager) throws Exception {
            InputStream is = assetManager.open("ocsvm_params.json");
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String jsonStr = new String(buffer, StandardCharsets.UTF_8);
            JSONObject json = new JSONObject(jsonStr);

            // 加载支持向量
            JSONArray svJson = json.getJSONArray("support_vectors");
            supportVectors = new float[svJson.length()][];
            for (int i = 0; i < svJson.length(); i++) {
                JSONArray vecJson = svJson.getJSONArray(i);
                if (vecJson.length() != 74) {
                    throw new Exception("支持向量维度错误：预期74，实际" + vecJson.length());
                }
                supportVectors[i] = new float[vecJson.length()];
                for (int j = 0; j < vecJson.length(); j++) {
                    supportVectors[i][j] = (float) vecJson.getDouble(j);
                }
            }

            // 加载对偶系数
            JSONArray dcJson = json.getJSONArray("dual_coef").getJSONArray(0);
            dualCoef = new float[dcJson.length()];
            for (int i = 0; i < dcJson.length(); i++) {
                dualCoef[i] = (float) dcJson.getDouble(i);
            }

            // 加载截距项和gamma
            intercept = (float) json.getJSONArray("intercept").getDouble(0);
            gamma = (float) json.getDouble("gamma");

            // 加载标准化均值和标准差
            // 从scaler_params.json中加载mean和var并计算std
            InputStream scalerIs = assetManager.open("scaler_params.json");
            byte[] scalerBuffer = new byte[scalerIs.available()];
            scalerIs.read(scalerBuffer);
            scalerIs.close();
            String scalerJsonStr = new String(scalerBuffer, StandardCharsets.UTF_8);
            JSONObject scalerJson = new JSONObject(scalerJsonStr);

            JSONArray meanJson = scalerJson.getJSONArray("mean");
            if (meanJson.length() != 74) {
                throw new Exception("mean长度错误：预期74，实际" + meanJson.length());
            }
            scalerMean = new float[meanJson.length()];
            for (int i = 0; i < meanJson.length(); i++) {
                scalerMean[i] = (float) meanJson.getDouble(i);
            }

            // 直接从scaler_params.json中的scale获取std（标准差）
            JSONArray stdJson = scalerJson.getJSONArray("scale");
            // stdJson已通过scale获取，长度应与meanJson一致
            if (stdJson.length() != 74) {
                throw new Exception("std长度错误：预期74，实际" + stdJson.length());
            }
            scalerStd = new float[stdJson.length()];
            for (int i = 0; i < stdJson.length(); i++) {
                scalerStd[i] = (float) stdJson.getDouble(i);
            }
            // 加载缺失值填充均值（新增）
            // 从scaler_params.json中加载imputer_mean
            JSONArray imputerMeanJson = scalerJson.getJSONArray("mean");
            if (imputerMeanJson.length() != 74) {
                throw new Exception("imputer_mean长度错误：预期74，实际" + imputerMeanJson.length());
            }
            imputerMean = new float[imputerMeanJson.length()];
            for (int i = 0; i < imputerMeanJson.length(); i++) {
                imputerMean[i] = (float) imputerMeanJson.getDouble(i);
            }
        }

        public float infer(float[] features) {
            if (features.length != 74) {
                throw new IllegalArgumentException("特征维度错误：预期74，实际" + features.length);
            }

            // 特征已经预处理过，无需再次标准化
            double decisionScore = 0;
            for (int i = 0; i < supportVectors.length; i++) {
                double distSq = 0;
                for (int j = 0; j < features.length; j++) {
                    double diff = features[j] - supportVectors[i][j];
                    distSq += diff * diff;
                }
                
                // 添加调试信息：检查距离计算
                if (i == 0) {
                    android.util.Log.d("OCSVM_DEBUG", "第一个支持向量距离计算 - distSq: " + distSq + ", gamma: " + gamma);
                }
                
                double k = Math.exp(-gamma * distSq);
                
                // 添加调试信息：检查核函数结果
                if (i == 0) {
                    android.util.Log.d("OCSVM_DEBUG", "第一个支持向量核函数结果 - k: " + k);
                }
                
                decisionScore += dualCoef[i] * k;
            }
            decisionScore += intercept;
            
            // 添加调试信息
            android.util.Log.d("OCSVM_DEBUG", "输入特征前5个值: " + 
                features[0] + ", " + features[1] + ", " + features[2] + ", " + features[3] + ", " + features[4]);
            android.util.Log.d("OCSVM_DEBUG", "决策分数: " + decisionScore + ", gamma: " + gamma + ", intercept: " + intercept);
            
            return (float) -decisionScore;
        }

        // 新增getter方法
        public float[] getImputerMean() {
            return imputerMean;
        }

        public float[] getScalerMean() {
            return scalerMean;
        }

        public float[] getScalerStd() {
            return scalerStd;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        collectHandler = new Handler(Looper.getMainLooper());
        bindViews();
        enableGameContainerTouch();
        initSensors();
        initFrequencySpinner();
        gameContainer.post(this::initGame);
        initOCSVMModel();
        bindButtonEvents();
        tvData.setText("");
    }

    private void enableGameContainerTouch() {
        if (gameContainer != null) {
            gameContainer.setClickable(true);
            gameContainer.setFocusable(true);
            gameContainer.setFocusableInTouchMode(true);
            Log.d("SensorInferDebug", "[TOUCH_ENABLE] gameContainer触摸属性已启用");
        } else {
            Log.e("SensorInferDebug", "[TOUCH_ENABLE] gameContainer绑定失败，无法启用触摸");
        }
    }

    private void bindViews() {
        mainContainer = findViewById(R.id.main_container);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        btnReset = findViewById(R.id.btnReset);
        btnDelete = findViewById(R.id.btnDelete);
        btnInfer = findViewById(R.id.btnInfer);
        tvData = findViewById(R.id.tvData);
        tvScore = findViewById(R.id.tvScore);
        tvInferResult = findViewById(R.id.tvInferResult);
        gameContainer = findViewById(R.id.gameContainer);
        spinnerFrequency = findViewById(R.id.spinnerFrequency);
    }

    private void initOCSVMModel() {
        try {
            ocsvmModel = new OCSVMModel();
            ocsvmModel.loadParams(getAssets());
            if (ocsvmModel.supportVectors == null || ocsvmModel.dualCoef == null || ocsvmModel.imputerMean == null) {
                throw new Exception("模型关键参数未初始化");
            }
            tvInferResult.append("✅ OCSVM模型加载成功（含填充和标准化参数）\n");
            // 添加调试信息
            tvInferResult.append("   支持向量数量: " + ocsvmModel.supportVectors.length + "\n");
            tvInferResult.append("   gamma参数: " + ocsvmModel.gamma + "\n");
            tvInferResult.append("   intercept参数: " + ocsvmModel.intercept + "\n");
            if (ocsvmModel.supportVectors.length > 0) {
                tvInferResult.append("   第一个支持向量前5个值: " + 
                    ocsvmModel.supportVectors[0][0] + ", " + ocsvmModel.supportVectors[0][1] + ", " + ocsvmModel.supportVectors[0][2] + ", " + 
                    ocsvmModel.supportVectors[0][3] + ", " + ocsvmModel.supportVectors[0][4] + "\n");
            }
        } catch (Exception e) {
            tvInferResult.append("❌ OCSVM模型加载失败：" + e.getMessage() + "\n");
            e.printStackTrace();
            ocsvmModel = null;
        }
    }

    private void bindButtonEvents() {
        btnStart.setEnabled(false);
        btnStart.setOnClickListener(v -> {});

        btnStop.setEnabled(false);
        btnStop.setOnClickListener(v -> {});

        btnDelete.setEnabled(false);
        btnDelete.setOnClickListener(v -> {});

        btnReset.setOnClickListener(v -> {
            initGame();
            totalScore = 0;
            tvScore.setText("得分: 0");
        });

        btnInfer.setOnClickListener(v -> start10sCollectAndInfer());

        gameContainer.setOnTouchListener((v, event) -> {
            handleTouchEvent(event);
            handleGameSlide(event);
            return false;
        });
    }

    private void start10sCollectAndInfer() {
        if (ocsvmModel == null) {
            Toast.makeText(this, "OCSVM模型未加载，无法验证", Toast.LENGTH_SHORT).show();
            return;
        }
        if (isInferring) {
            Toast.makeText(this, "验证中，请等待...", Toast.LENGTH_SHORT).show();
            return;
        }

        initCsvPaths(); // 初始化当前会话的CSV路径（含时间戳）
        isInferring = true;
        dataLock.lock();
        try {
            inferRawData.clear();
            inferScores.clear();
            touchIntervals.clear();
            currentTouchStart = 0;
            currentTouchMaxLiftTime = 0;
            isCurrentTouchFinished = true;
            currentActionId = 0;
        } finally {
            dataLock.unlock();
        }

        tvInferResult.setText("🔍 开始身份验证\n1. 10秒内请在游戏区完成完整滑动\n2. 必须包含滑动动作才会被统计为有效会话\n3. 时间到后自动分析...\n");
        tvInferResult.append("💾 原始特征CSV路径：" + featureCsvPath + "\n");
        tvInferResult.append("💾 处理后特征CSV路径：" + processedFeatureCsvPath + "\n");
        showCollectCountdown();

        if (!isSensorRegistered) {
            try {
                if (accelerometer != null) {
                    sensorManager.registerListener(sensorListener, accelerometer, sensorDelay, sensorHandler);
                }
                if (gyroscope != null) {
                    sensorManager.registerListener(sensorListener, gyroscope, sensorDelay, sensorHandler);
                }
                if (magnetometer != null) {
                    sensorManager.registerListener(sensorListener, magnetometer, sensorDelay, sensorHandler);
                }
                isSensorRegistered = true;
                tvInferResult.append("✅ 传感器已启动（" + selectedHz + "Hz）\n");
            } catch (Exception e) {
                Log.e("SensorInferDebug", "[SENSOR_REG] 传感器注册失败：" + e.getMessage(), e);
                tvInferResult.append("❌ 传感器启动失败：" + e.getMessage() + "\n");
                isInferring = false;
                return;
            }
        }

        collectTimeoutRunnable = this::extractFeaturesAndInferAfterCollect;
        collectHandler.postDelayed(collectTimeoutRunnable, COLLECT_DURATION);
    }

    // 初始化CSV路径（含处理后文件路径）
    private void initCsvPaths() {
        collectSessionId = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        File csvDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        rawDataCsvPath = new File(csvDir, "2048_InferRawData_" + collectSessionId + ".csv").getAbsolutePath();
        featureCsvPath = new File(csvDir, "2048_InferFeature_" + collectSessionId + ".csv").getAbsolutePath();
        processedFeatureCsvPath = featureCsvPath.replace(".csv", "_processed.csv"); // 处理后文件名
    }

    private void showCollectCountdown() {
        new Thread(() -> {
            for (int i = 10; i > 0; i--) {
                int count = i;
                tvInferResult.post(() -> tvInferResult.append("⏳ 剩余收集时间：" + count + "秒\n"));
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                    break;
                }
            }
        }).start();
    }

    private void extractFeaturesAndInferAfterCollect() {
        isInferring = false;
        if (isSensorRegistered) {
            try {
                sensorManager.unregisterListener(sensorListener);
                isSensorRegistered = false;
            } catch (Exception e) {
                Log.e("SensorInferDebug", "[SENSOR_UNREG] 传感器注销失败：" + e.getMessage(), e);
            }
        }

        tvInferResult.post(() -> tvInferResult.append("\n⏹️ 10秒收集结束，开始预处理数据...\n"));

        List<AllDataModel> storedData = new ArrayList<>();
        List<long[]> storedIntervals = new ArrayList<>();
        dataLock.lock();
        try {
            storedData.addAll(inferRawData);
            storedIntervals.addAll(touchIntervals);
            tvInferResult.post(() -> tvInferResult.append("📊 已读取存储数据：" + storedData.size() + " 条，触摸动作：" + storedIntervals.size() + " 个\n"));
        } finally {
            dataLock.unlock();
        }

        int dataCountBeforeDedup = storedData.size();
        List<AllDataModel> deduplicatedData = deduplicateRawData(storedData);
        tvInferResult.post(() -> tvInferResult.append("✅ 原始数据去重完成：去重前" + dataCountBeforeDedup + "条，去重后" + deduplicatedData.size() + "条\n"));

        saveRawDataToCsv(deduplicatedData);

        List<long[]> correctedIntervals = correctTouchIntervals(deduplicatedData);
        tvInferResult.post(() -> tvInferResult.append("✅ 触摸区间校正完成：原始" + storedIntervals.size() + "个，校正后" + correctedIntervals.size() + "个\n"));

        List<TouchSession> touchSessions = preprocessInferData(deduplicatedData, correctedIntervals);
        tvInferResult.post(() -> tvInferResult.append("✅ 提取到 " + touchSessions.size() + " 个有效触摸会话\n"));

        if (touchSessions.size() < REQUIRED_TOUCH_COUNT) {
            tvInferResult.post(() -> {
                tvInferResult.append("❌ 有效滑动不足（需" + REQUIRED_TOUCH_COUNT + "个，实际" + touchSessions.size() + "个）\n");
                tvInferResult.append("请重新点击「身份验证」\n");
            });
            return;
        }

        int useSessionCount = touchSessions.size();
        tvInferResult.post(() -> tvInferResult.append("\n📈 开始分析" + useSessionCount + "个有效会话的触摸特征...\n"));

        new Thread(() -> {
            try {
                List<FeatureModel> allFeatureModels = new ArrayList<>();

                // 1. 提取特征并保存到列表
                for (int i = 0; i < useSessionCount; i++) {
                    TouchSession session = touchSessions.get(i);
                    float[] features = extractTouchFeatures(session);
                    if (features == null) {
                        final int idx = i + 1;
                        tvInferResult.post(() -> tvInferResult.append("⚠️ 第" + idx + "个触摸动作特征提取失败\n"));
                        continue;
                    }

                    int sampleCount = session.getDataList().size();
                    long startTime = session.getStartTime();
                    long endTime = session.getEndTime();

                    FeatureModel featureModel = new FeatureModel(
                            collectSessionId,
                            "Touch_" + (i + 1),
                            System.currentTimeMillis(),
                            sampleCount,
                            startTime,
                            endTime,
                            features
                    );
                    allFeatureModels.add(featureModel);
                }

                // 2. 同步保存原始特征到CSV
                boolean saveSuccess = saveFeatureDataToCsvSync(allFeatureModels);
                if (!saveSuccess) {
                    tvInferResult.post(() -> tvInferResult.append("❌ 特征保存失败，终止推理\n"));
                    return;
                }

                // 3. 处理特征CSV（填充+标准化）
                // 恢复为带参数的调用，传入模型中的训练集参数
                boolean processSuccess = processFeatureCsv(
                        featureCsvPath,
                        processedFeatureCsvPath,
                        ocsvmModel.getImputerMean(),  // 模型中的填充均值（训练集参数）
                        ocsvmModel.getScalerMean(),   // 模型中的标准化均值（训练集参数）
                        ocsvmModel.getScalerStd()     // 模型中的标准化标准差（训练集参数）
                );
                if (!processSuccess) {
                    tvInferResult.post(() -> tvInferResult.append("❌ 特征预处理失败，终止推理\n"));
                    return;
                }

                // 4. 从处理后的CSV读取特征
                List<float[]> csvFeaturesList = readFeaturesFromCsv(processedFeatureCsvPath);
                if (csvFeaturesList.isEmpty()) {
                    tvInferResult.post(() -> tvInferResult.append("❌ 未从处理后的CSV读取到有效特征，终止推理\n"));
                    return;
                }

                // 5. 用处理后的特征进行推理
                for (int i = 0; i < csvFeaturesList.size(); i++) {
                    float[] featuresFromCsv = csvFeaturesList.get(i);
                    
                    // 添加调试信息：显示部分特征值
                    StringBuilder debugInfo = new StringBuilder();
                    debugInfo.append("第").append(i+1).append("个样本特征值：");
                    for (int f = 0; f < Math.min(5, featuresFromCsv.length); f++) {
                        debugInfo.append(String.format("f%d=%.4f, ", f, featuresFromCsv[f]));
                    }
                    debugInfo.append("...\n");
                    tvInferResult.post(() -> tvInferResult.append(debugInfo.toString()));
                    
                    float anomalyScore = ocsvmModel.infer(featuresFromCsv);
                    inferScores.add(anomalyScore);

                    final int idx = i + 1;
                    final float score = anomalyScore;
                    tvInferResult.post(() -> tvInferResult.append(
                            "第" + idx + "次分析（处理后CSV特征）：异常分数=" + String.format("%.4f", score) + "\n"
                    ));
                }

                // 6. 判定结果
                int normalCount = 0;
                for (float score : inferScores) {
                    if (score < ANOMALY_THRESHOLD) normalCount++;
                }
                boolean isSelf = normalCount > csvFeaturesList.size() / 2;

                final int finalNormalCount = normalCount;
                tvInferResult.post(() -> {
                    tvInferResult.append("\n✅ 身份验证完成\n");
                    tvInferResult.append("📊 分析结果汇总：\n");
                    tvInferResult.append("   异常分数列表：" + inferScores.stream()
                            .map(s -> String.format("%.4f", s))
                            .collect(Collectors.joining(", ")) + "\n");
                    tvInferResult.append("   正常会话数：" + finalNormalCount + " | 异常会话数：" + (csvFeaturesList.size() - finalNormalCount) + "\n");
                    tvInferResult.append("   判定阈值：" + (csvFeaturesList.size() / 2.0) + "\n");
                    tvInferResult.append("   正常判断次数：" + finalNormalCount + "/" + csvFeaturesList.size() + "\n");
                    tvInferResult.append("   最终结论：" + (isSelf ? "✅ 判定为本人" : "❌ 判定为非本人") + "\n");
                    tvInferResult.append("💾 原始特征CSV：" + featureCsvPath + "\n");
                    tvInferResult.append("💾 处理后特征CSV：" + processedFeatureCsvPath + "\n");
                });

            } catch (Exception e) {
                e.printStackTrace();
                tvInferResult.post(() -> {
                    tvInferResult.append("❌ 验证失败：" + e.getMessage() + "\n");
                });
            }
        }).start();
    }

    // 新增：特征预处理（均值填充+标准化）
    // 修改后的特征预处理方法：用JSON中的训练集参数（填充+标准化）
    // 恢复参数：接收模型中的训练集参数（imputerMean、scalerMean、scalerStd）
    private boolean processFeatureCsv(
            String rawFeatureCsvPath,
            String processedCsvPath,
            float[] imputerMean,    // 模型中的填充均值（训练集参数）
            float[] scalerMean,     // 模型中的标准化均值（训练集参数）
            float[] scalerStd       // 模型中的标准化标准差（训练集参数）
    ) {
        try {
            File rawCsvFile = new File(rawFeatureCsvPath);
            if (!rawCsvFile.exists()) {
                tvInferResult.post(() -> tvInferResult.append("❌ 原始特征CSV不存在：" + rawFeatureCsvPath + "\n"));
                return false;
            }

            List<String> lines = Files.readAllLines(rawCsvFile.toPath(), StandardCharsets.UTF_8);
            if (lines.size() < 2) {
                tvInferResult.post(() -> tvInferResult.append("❌ 原始特征CSV数据为空：" + rawFeatureCsvPath + "\n"));
                return false;
            }

            List<String> processedLines = new ArrayList<>();

            // 生成表头（每个特征：标准化值 + 训练集均值 + 训练集标准差）
            StringBuilder headerSb = new StringBuilder();
            headerSb.append("会话ID,触摸会话ID,提取时间戳,采样数,采样起始时间(毫秒),采样结束时间(毫秒),");
            for (int j = 0; j < 74; j++) {
                String[] featureNames = {
                        "起始x", "起始y", "结束x", "结束y", "位移长度", "移动长度", "位移/移动比值",
                        "角度1_均值", "角度1_最大值", "角度1_最小值", "角度1_方差",
                        "角度2_均值", "角度2_最大值", "角度2_最小值", "角度2_方差",
                        "持续时间（毫秒）",
                        "速度_均值", "速度_最大值", "速度_最小值", "速度_方差",
                        "压力_均值", "压力_最大值", "压力_最小值", "压力_方差",
                        "面积_均值", "面积_最大值", "面积_最小值", "面积_方差",
                        "移动角度（弧度）",
                        "加速度 x_均值", "加速度 x_最小值", "加速度 x_最大值", "加速度 x_方差", "加速度 x_复杂度",
                        "加速度 y_均值", "加速度 y_最小值", "加速度 y_最大值", "加速度 y_方差", "加速度 y_复杂度",
                        "加速度 z_均值", "加速度 z_最小值", "加速度 z_最大值", "加速度 z_方差", "加速度 z_复杂度",
                        "角速度 x_均值", "角速度 x_最小值", "角速度 x_最大值", "角速度 x_方差", "角速度 x_复杂度",
                        "角速度 y_均值", "角速度 y_最小值", "角速度 y_最大值", "角速度 y_方差", "角速度 y_复杂度",
                        "角速度 z_均值", "角速度 z_最小值", "角速度 z_最大值", "角速度 z_方差", "角速度 z_复杂度",
                        "磁力计 x_均值", "磁力计 x_最小值", "磁力计 x_最大值", "磁力计 x_方差", "磁力计 x_复杂度",
                        "磁力计 y_均值", "磁力计 y_最小值", "磁力计 y_最大值", "磁力计 y_方差", "磁力计 y_复杂度",
                        "磁力计 z_均值", "磁力计 z_最小值", "磁力计 z_最大值", "磁力计 z_方差", "磁力计 z_复杂度"
                };
                headerSb.append(featureNames[j]).append("_标准化,");
                headerSb.append(featureNames[j]).append("_训练集均值,");
                headerSb.append(featureNames[j]).append("_训练集标准差");
                if (j < 73) {
                    headerSb.append(",");
                }
            }
            processedLines.add(headerSb.toString());

            // 遍历每一行数据，用模型参数处理
            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split(",");
                if (parts.length != 6 + 74) {
                    final int lineNum = i;
                    tvInferResult.post(() -> tvInferResult.append("⚠️ 特征CSV格式错误，行" + lineNum + "列数=" + parts.length + "，跳过\n"));
                    continue;
                }

                // 拼接前6个元数据列
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < 6; j++) {
                    sb.append(parts[j]).append(",");
                }

                // 处理74维特征（用模型中的训练集参数）
                for (int j = 0; j < 74; j++) {
                    int currentJ = j; // 解决lambda变量问题
                    float featureVal;

                    // 缺失值处理：用模型中的训练集均值填充
                    int featureIndex = 6 + j; // 定位到特征值的索引
                    if (parts[featureIndex] == null || parts[featureIndex].isEmpty() || parts[featureIndex].equalsIgnoreCase("NaN")) {
                        featureVal = imputerMean[j];
                        tvInferResult.post(() -> tvInferResult.append(
                                "⚠️ 特征" + (currentJ + 1) + "缺失，用训练集均值" + String.format("%.4f", imputerMean[currentJ]) + "填充\n"
                        ));
                    } else {
                        try {
                            featureVal = Float.parseFloat(parts[featureIndex]);
                        } catch (NumberFormatException e) {
                            featureVal = imputerMean[j];
                            tvInferResult.post(() -> tvInferResult.append(
                                    "⚠️ 特征" + (currentJ + 1) + "值无效，用训练集均值填充\n"
                            ));
                        }
                    }

                    // 标准化：用模型中的训练集均值和标准差
                    float scaledVal;
                    if (scalerStd[j] == 0) {
                        scaledVal = 0f;
                    } else {
                        scaledVal = (featureVal - scalerMean[j]) / scalerStd[j];
                    }

                    // 追加：标准化值 + 训练集均值 + 训练集标准差
                    sb.append(String.format(Locale.ENGLISH, "%.4f", scaledVal)).append(",");
                    sb.append(String.format(Locale.ENGLISH, "%.4f", scalerMean[j])).append(",");
                    sb.append(String.format(Locale.ENGLISH, "%.4f", scalerStd[j]));

                    // 添加调试信息
                    if (i == 1 && j < 5) { // 只输出第一行前5个特征的处理信息
                        final int index = j; // 创建effectively final变量
                        final float originalVal = featureVal;
                        final float scaledValue = scaledVal;
                        tvInferResult.post(() -> tvInferResult.append(
                            "特征" + index + "处理: 原始=" + String.format("%.4f", originalVal) + 
                            ", 标准化=" + String.format("%.4f", scaledValue) + 
                            ", 均值=" + String.format("%.4f", scalerMean[index]) + 
                            ", 标准差=" + String.format("%.4f", scalerStd[index]) + "\n"));
                    }

                    if (j < 73) {
                        sb.append(",");
                    }
                }

                processedLines.add(sb.toString());
            }

            // 保存处理后的CSV
            File processedCsvFile = new File(processedCsvPath);
            if (!processedCsvFile.getParentFile().exists()) {
                processedCsvFile.getParentFile().mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(processedCsvFile), StandardCharsets.UTF_8))) {
                writer.write('\ufeff'); // UTF-8 BOM
                for (String processedLine : processedLines) {
                    writer.write(processedLine + "\n");
                }
            }

            tvInferResult.post(() -> tvInferResult.append("✅ 特征预处理完成（用训练集参数）！路径：" + processedCsvPath + "\n"));
            return true;

        } catch (IOException e) {
            e.printStackTrace();
            tvInferResult.post(() -> tvInferResult.append("❌ 处理特征CSV失败：" + e.getMessage() + "\n"));
            return false;
        }
    }

    private List<long[]> correctTouchIntervals(List<AllDataModel> deduplicatedData) {
        List<long[]> correctedIntervals = new ArrayList<>();
        if (deduplicatedData.isEmpty()) return correctedIntervals;

        Set<Integer> actionIds = new HashSet<>();
        for (AllDataModel data : deduplicatedData) {
            actionIds.add(data.actionId);
        }

        for (int actionId : actionIds) {
            if (actionId == 0) continue;

            List<AllDataModel> actionData = new ArrayList<>();
            for (AllDataModel data : deduplicatedData) {
                if (data.actionId == actionId) {
                    actionData.add(data);
                }
            }
            if (actionData.isEmpty()) continue;

            Collections.sort(actionData, (a, b) -> Long.compare(a.time, b.time));

            long tstart = 0;
            for (AllDataModel data : actionData) {
                if ("按下".equals(data.touchType)) {
                    tstart = data.time;
                    break;
                }
            }
            if (tstart == 0) continue;

            long tend = 0;
            for (int i = actionData.size() - 1; i >= 0; i--) {
                AllDataModel data = actionData.get(i);
                if ("抬起".equals(data.touchType)) {
                    tend = data.time;
                    break;
                }
            }
            if (tend == 0) continue;

            correctedIntervals.add(new long[]{tstart, tend});
            Log.d("IntervalFix", "动作ID=" + actionId + " 校正区间：tstart=" + tstart + ", tend=" + tend);
        }

        Collections.sort(correctedIntervals, (a, b) -> Long.compare(a[0], b[0]));
        return correctedIntervals;
    }

    private List<AllDataModel> deduplicateRawData(List<AllDataModel> rawData) {
        if (rawData == null || rawData.size() < 2) {
            if (rawData != null && rawData.size() == 1) {
                Collections.sort(rawData, (a, b) -> Long.compare(a.time, b.time));
            }
            return rawData;
        }

        List<AllDataModel> deduplicatedData = new ArrayList<>();
        deduplicatedData.add(rawData.get(0));

        for (int i = 1; i < rawData.size(); i++) {
            AllDataModel prev = deduplicatedData.get(deduplicatedData.size() - 1);
            AllDataModel curr = rawData.get(i);

            boolean sameTime = curr.time == prev.time;
            boolean xDiffSmall = Math.abs(curr.x - prev.x) < 0.1f;
            boolean yDiffSmall = Math.abs(curr.y - prev.y) < 0.1f;

            if (!(sameTime && xDiffSmall && yDiffSmall)) {
                deduplicatedData.add(curr);
            } else {
                Log.d("DedupDebug", "去重一条重复数据：时间戳=" + curr.time);
            }
        }

        Collections.sort(deduplicatedData, (a, b) -> Long.compare(a.time, b.time));
        Log.d("SortDebug", "数据排序完成：共" + deduplicatedData.size() + "条");
        return deduplicatedData;
    }

    // 同步保存原始数据CSV
    private void saveRawDataToCsv(List<AllDataModel> rawData) {
        if (rawData.isEmpty()) {
            tvInferResult.post(() -> tvInferResult.append("⚠️ 无原始数据可保存\n"));
            return;
        }

        tvInferResult.post(() -> tvInferResult.append("💾 开始保存原始数据（" + rawData.size() + "条）...\n"));

        File tempDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS + "/temp");
        if (tempDir == null || !tempDir.exists() && !tempDir.mkdirs()) {
            tvInferResult.post(() -> tvInferResult.append("❌ 无法创建临时目录\n"));
            return;
        }

        String tempFileName = "temp_raw_" + collectSessionId + ".json";
        File tempFile = new File(tempDir, tempFileName);

        Gson gson = new Gson();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile, Charset.forName("UTF-8")))) {
            writer.write(gson.toJson(rawData, new TypeToken<List<AllDataModel>>() {}.getType()));
        } catch (IOException e) {
            tvInferResult.post(() -> tvInferResult.append("❌ 原始数据临时文件写入失败：" + e.getMessage() + "\n"));
            e.printStackTrace();
            return;
        }

        Data inputData = new Data.Builder()
                .putString("tempFilePath", tempFile.getAbsolutePath())
                .putString("targetCsvPath", rawDataCsvPath)
                .putBoolean("isRawData", true)
                .build();

        OneTimeWorkRequest saveRequest = new OneTimeWorkRequest.Builder(SaveDataWorker.class)
                .setInputData(inputData)
                .build();

        WorkManager.getInstance(this).enqueue(saveRequest);
        WorkManager.getInstance(this)
                .getWorkInfoByIdLiveData(saveRequest.getId())
                .observe(this, workInfo -> {
                    if (workInfo != null) {
                        if (workInfo.getState() == WorkInfo.State.SUCCEEDED) {
                            tvInferResult.append("✅ 原始数据保存成功：" + rawDataCsvPath + "\n");
                            if (tempFile.exists()) {
                                tempFile.delete();
                            }
                        } else if (workInfo.getState() == WorkInfo.State.FAILED) {
                            String error = workInfo.getOutputData().getString("error") != null ?
                                    workInfo.getOutputData().getString("error") : "未知错误";
                            tvInferResult.append("❌ 原始数据保存失败：" + error + "\n");
                        }
                    }
                });
    }

    // 同步保存特征数据到CSV
    private boolean saveFeatureDataToCsvSync(List<FeatureModel> featureModels) {
        if (featureModels.isEmpty()) {
            tvInferResult.post(() -> tvInferResult.append("⚠️ 无特征数据可保存\n"));
            return false;
        }

        try {
            File csvFile = new File(featureCsvPath);
            if (!csvFile.getParentFile().exists()) {
                csvFile.getParentFile().mkdirs();
            }

            try (BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(new FileOutputStream(csvFile), StandardCharsets.UTF_8))) {
                writer.write('\ufeff'); // UTF-8 BOM
                writer.write(FeatureModel.getCsvHeader() + "\n");
                for (FeatureModel model : featureModels) {
                    writer.write(model.toCsv() + "\n");
                }
            }

            tvInferResult.post(() -> tvInferResult.append("✅ 原始特征数据已同步保存到：" + featureCsvPath + "\n"));
            return true;
        } catch (IOException e) {
            tvInferResult.post(() -> tvInferResult.append("❌ 特征数据同步保存失败：" + e.getMessage() + "\n"));
            e.printStackTrace();
            return false;
        }
    }

    // 从CSV读取74维特征
    private List<float[]> readFeaturesFromCsv(String csvPath) {
        List<float[]> featuresList = new ArrayList<>();
        try {
            File csvFile = new File(csvPath);
            if (!csvFile.exists()) {
                tvInferResult.post(() -> tvInferResult.append("❌ 特征CSV文件不存在：" + csvPath + "\n"));
                return featuresList;
            }

            List<String> lines = Files.readAllLines(csvFile.toPath(), StandardCharsets.UTF_8);
            if (lines.size() < 2) {
                tvInferResult.post(() -> tvInferResult.append("❌ 特征CSV数据为空：" + csvPath + "\n"));
                return featuresList;
            }

            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i).trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split(",");
                // 修复：处理后的CSV列数 = 6（元数据） + 74×3（每个特征3列）
                if (parts.length != 6 + 74 * 3) {
                    final int lineNumber = i;
                    tvInferResult.post(() -> tvInferResult.append("⚠️ 特征CSV格式错误，行" + lineNumber + "列数=" + parts.length + "，预期" + (6 + 74*3) + "列，跳过\n"));
                    continue;
                }

                float[] features = new float[74];
                // 修复：读取每个特征的「标准化值」列（第6+3j列，j从0到73）
                for (int j = 0; j < 74; j++) {
                    try {
                        // 处理后的CSV列顺序：元数据(6列) → 特征1_标准化(6)、特征1_均值(7)、特征1_标准差(8) → 特征2_标准化(9)、...
                        int scaledValIndex = 6 + j * 3; // 标准化值所在列的索引
                        features[j] = Float.parseFloat(parts[scaledValIndex]);
                    } catch (NumberFormatException e) {
                        features[j] = 0f;
                        final int featureIndex = j;
                        tvInferResult.post(() -> tvInferResult.append("⚠️ 特征" + (featureIndex + 1) + "解析失败：" + parts[6 + featureIndex * 3] + "\n"));
                    }
                }
                featuresList.add(features);
            }

            tvInferResult.post(() -> tvInferResult.append("✅ 从CSV读取特征：" + featuresList.size() + "条，每条74维\n"));
        } catch (IOException e) {
            tvInferResult.post(() -> tvInferResult.append("❌ 读取特征CSV失败：" + e.getMessage() + "\n"));
            e.printStackTrace();
        }
        return featuresList;
    }

    private List<TouchSession> preprocessInferData(List<AllDataModel> dataCopy, List<long[]> intervalsCopy) {
        List<TouchSession> sessions = new ArrayList<>();
        if (intervalsCopy.isEmpty()) return sessions;

        Collections.sort(intervalsCopy, (a, b) -> Long.compare(a[0], b[0]));

        for (int i = 0; i < intervalsCopy.size(); i++) {
            long[] interval = intervalsCopy.get(i);
            long tstart = interval[0];
            long tend = interval[1];

            if (i < intervalsCopy.size() - 1) {
                long nextTstart = intervalsCopy.get(i + 1)[0];
                if (tend >= nextTstart) {
                    tend = nextTstart - 1;
                    Log.w("TouchFix", "修正重叠区间：原tend=" + interval[1] + " → 新tend=" + tend);
                }
            }

            TouchSession session = new TouchSession();
            boolean hasPress = false;
            boolean hasSlide = false;
            boolean hasRelease = false;

            for (AllDataModel data : dataCopy) {
                if (data == null) continue;

                boolean inTimeRange = data.getTime() >= tstart && data.getTime() <= tend;
                boolean isEffectiveType = "按下".equals(data.touchType)
                        || "滑动中".equals(data.touchType)
                        || "抬起".equals(data.touchType);

                if (!inTimeRange || !isEffectiveType) {
                    continue;
                }

                if ("按下".equals(data.touchType)) hasPress = true;
                if ("滑动中".equals(data.touchType)) hasSlide = true;
                if ("抬起".equals(data.touchType)) hasRelease = true;

                session.addData(data);
                Log.d("PreprocessDebug", "[SESSION_" + i + "] 添加数据：类型=" + data.touchType);
            }

            if (hasPress && hasSlide && hasRelease && !session.getDataList().isEmpty()) {
                sessions.add(session);
                Log.d("PreprocessDebug", "[SESSION_ADD] 有效会话" + i + "：数据条数=" + session.getDataList().size());
            } else {
                Log.d("PreprocessDebug", "[SESSION_SKIP] 会话" + i + "无效");
            }
        }

        return sessions;
    }

    private float[] extractTouchFeatures(TouchSession session) {
        List<AllDataModel> dataList = session.getDataList();
        if (dataList.size() < 2) return null;

        List<Float> features = new ArrayList<>();
        AllDataModel first = dataList.get(0);
        AllDataModel last = dataList.get(dataList.size() - 1);

        // 1. 基础位置特征（4个）
        features.add(first.x);
        features.add(first.y);
        features.add(last.x);
        features.add(last.y);

        // 2. 位移与移动长度特征（3个）
        double displacement = Math.hypot(last.x - first.x, last.y - first.y);
        double moveLength = 0;
        for (int i = 0; i < dataList.size() - 1; i++) {
            AllDataModel curr = dataList.get(i);
            AllDataModel next = dataList.get(i + 1);
            moveLength += Math.hypot(next.x - curr.x, next.y - curr.y);
        }
        double displacementMoveRatio = moveLength == 0 ? 0 : displacement / moveLength;
        features.add((float) displacement);
        features.add((float) moveLength);
        features.add((float) displacementMoveRatio);

        // 3. 角度1特征（4个）
        List<Float> angle1List = new ArrayList<>();
        for (AllDataModel data : dataList) {
            float angle = (float) Math.atan2(data.y, data.x);
            if (angle < 0) angle += 2 * (float) Math.PI;
            angle1List.add(angle);
        }
        features.add((float) calculateMean(angle1List));
        features.add((float) calculateMax(angle1List));
        features.add((float) calculateMin(angle1List));
        features.add((float) calculateVariance(angle1List, calculateMean(angle1List)));

        // 4. 角度2特征（4个）
        List<Float> angle2List = new ArrayList<>();
        for (int i = 0; i < dataList.size() - 1; i++) {
            AllDataModel curr = dataList.get(i);
            AllDataModel next = dataList.get(i + 1);
            float dx = next.x - curr.x;
            float dy = next.y - curr.y;
            float angle = (float) Math.atan2(dy, dx);
            angle2List.add(angle);
        }
        features.add((float) calculateMean(angle2List));
        features.add((float) calculateMax(angle2List));
        features.add((float) calculateMin(angle2List));
        features.add((float) calculateVariance(angle2List, calculateMean(angle2List)));

        // 5. 时间特征（1个）
        features.add((float) (last.time - first.time));

        // 6. 速度特征（4个）
        List<Float> speedList = new ArrayList<>();
        for (int i = 0; i < dataList.size() - 1; i++) {
            AllDataModel curr = dataList.get(i);
            AllDataModel next = dataList.get(i + 1);
            long timeDiff = next.time - curr.time;
            if (timeDiff == 0) continue;
            double distance = Math.hypot(next.x - curr.x, next.y - curr.y);
            speedList.add((float) (distance / timeDiff));
        }
        features.add((float) calculateMean(speedList));
        features.add((float) calculateMax(speedList));
        features.add((float) calculateMin(speedList));
        features.add((float) calculateVariance(speedList, calculateMean(speedList)));

        // 7. 压力特征（4个）
        List<Float> pressureList = dataList.stream().map(d -> d.pressure).collect(Collectors.toList());
        features.add((float) calculateMean(pressureList));
        features.add((float) calculateMax(pressureList));
        features.add((float) calculateMin(pressureList));
        features.add((float) calculateVariance(pressureList, calculateMean(pressureList)));

        // 8. 面积特征（4个）
        List<Float> areaList = dataList.stream().map(d -> d.size).collect(Collectors.toList());
        features.add((float) calculateMean(areaList));
        features.add((float) calculateMax(areaList));
        features.add((float) calculateMin(areaList));
        features.add((float) calculateVariance(areaList, calculateMean(areaList)));

        // 9. 移动角度特征（1个）
        float dx = last.x - first.x;
        float dy = last.y - first.y;
        float moveAngle = (float) Math.atan2(dy, dx);
        if (moveAngle < 0) moveAngle += 2 * (float) Math.PI;
        features.add(moveAngle);

        // 10. 加速度特征（X/Y/Z各5个，共15个）
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.accX).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.accY).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.accZ).collect(Collectors.toList())));

        // 11. 角速度特征（X/Y/Z各5个，共15个）
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.gyroX).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.gyroY).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.gyroZ).collect(Collectors.toList())));

        // 12. 磁力计特征（X/Y/Z各5个，共15个）
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.magX).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.magY).collect(Collectors.toList())));
        features.addAll(getSensorAxisFeatures(dataList.stream().map(d -> d.magZ).collect(Collectors.toList())));

        if (features.size() != 74) {
            Log.e("FeatureError", "特征数量不匹配：实际" + features.size() + "个，预期74个");
            return null;
        }

        float[] featureArray = new float[74];
        for (int i = 0; i < 74; i++) {
            featureArray[i] = features.get(i);
        }
        return featureArray;
    }

    private List<Float> getSensorAxisFeatures(List<Float> values) {
        List<Float> axisFeatures = new ArrayList<>();
        if (values.isEmpty()) {
            axisFeatures.add(0f);
            axisFeatures.add(0f);
            axisFeatures.add(0f);
            axisFeatures.add(0f);
            axisFeatures.add(0f);
            return axisFeatures;
        }

        double mean = calculateMean(values);
        double min = calculateMin(values);
        double max = calculateMax(values);
        double variance = calculateVariance(values, mean);
        double complexity = calculateComplexity(values);

        axisFeatures.add((float) mean);
        axisFeatures.add((float) min);
        axisFeatures.add((float) max);
        axisFeatures.add((float) variance);
        axisFeatures.add((float) complexity);
        return axisFeatures;
    }

    private double calculateMean(List<Float> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (float v : values) sum += v;
        return sum / values.size();
    }

    private double calculateMax(List<Float> values) {
        if (values.isEmpty()) return 0.0;
        double max = values.get(0);
        for (float v : values) if (v > max) max = v;
        return max;
    }

    private double calculateMin(List<Float> values) {
        if (values.isEmpty()) return 0.0;
        double min = values.get(0);
        for (float v : values) if (v < min) min = v;
        return min;
    }

    private double calculateVariance(List<Float> values, double mean) {
        if (values.isEmpty()) return 0.0;
        double sumSq = 0.0;
        for (float v : values) sumSq += Math.pow(v - mean, 2);
        return sumSq / values.size();
    }

    private double calculateComplexity(List<Float> values) {
        if (values.size() < 2) return 0.0;
        double sumDiffSq = 0.0;
        for (int i = 0; i < values.size() - 1; i++) {
            double diff = values.get(i + 1) - values.get(i);
            sumDiffSq += diff * diff;
        }
        return Math.sqrt(sumDiffSq);
    }

    private void initSensors() {
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        int accMaxFreq = getSensorMaxFrequency(accelerometer);
        int gyroMaxFreq = getSensorMaxFrequency(gyroscope);
        int magMaxFreq = getSensorMaxFrequency(magnetometer);
        int deviceMaxFreq = Math.min(Math.min(accMaxFreq, gyroMaxFreq), magMaxFreq);

        supportedHzList.clear();
        for (int hz : FIXED_HZ_OPTIONS) {
            if (hz <= deviceMaxFreq) {
                supportedHzList.add(hz);
            }
        }
        if (!supportedHzList.contains(50)) {
            supportedHzList.add(50);
        }

        sensorThread = new HandlerThread("SensorThread");
        sensorThread.start();
        sensorHandler = new Handler(sensorThread.getLooper());
    }

    private void initFrequencySpinner() {
        List<String> displayOptions = new ArrayList<>();
        for (int hz : supportedHzList) {
            displayOptions.add(hz + " Hz");
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, displayOptions);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerFrequency.setAdapter(adapter);

        int defaultPos = supportedHzList.indexOf(50);
        if (defaultPos == -1) {
            defaultPos = supportedHzList.size() - 1;
        }
        spinnerFrequency.setSelection(defaultPos);
        selectedHz = 50;
        sensorDelay = 1000000 / selectedHz;

        spinnerFrequency.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedHz = 50;
                sensorDelay = 1000000 / selectedHz;
                tvInferResult.append("已选择采样频率：" + selectedHz + "Hz（延迟：" + sensorDelay + "微秒）\n");
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    private final SensorEventListener sensorListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (!isInferring || event == null || event.values == null) return;

            float accX = 0, accY = 0, accZ = 0;
            float gyroX = 0, gyroY = 0, gyroZ = 0;
            float magX = 0, magY = 0, magZ = 0;
            switch (event.sensor.getType()) {
                case Sensor.TYPE_LINEAR_ACCELERATION:
                    accX = event.values[0];
                    accY = event.values[1];
                    accZ = event.values[2];
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    gyroX = event.values[0];
                    gyroY = event.values[1];
                    gyroZ = event.values[2];
                    break;
                case Sensor.TYPE_MAGNETIC_FIELD:
                    magX = event.values[0];
                    magY = event.values[1];
                    magZ = event.values[2];
                    break;
            }

            dataLock.lock();
            try {
                if (!inferRawData.isEmpty()) {
                    AllDataModel latestData = inferRawData.get(inferRawData.size() - 1);
                    if (System.currentTimeMillis() - latestData.time < 1000) {
                        AllDataModel updatedData = new AllDataModel(
                                latestData.touchType,
                                latestData.touchDirection,
                                latestData.x,
                                latestData.y,
                                latestData.pressure,
                                latestData.size,
                                latestData.time,
                                latestData.actionId,
                                accX, accY, accZ,
                                gyroX, gyroY, gyroZ,
                                magX, magY, magZ
                        );
                        inferRawData.set(inferRawData.size() - 1, updatedData);
                    }
                }
            } finally {
                dataLock.unlock();
            }

            float touchX = currentTouchType.equals("无触摸") ? 0 : currentTouchX;
            float touchY = currentTouchType.equals("无触摸") ? 0 : currentTouchY;
            AllDataModel data = new AllDataModel(
                    currentTouchType != null ? currentTouchType : "无触摸",
                    currentTouchDirection != null ? currentTouchDirection : "无",
                    touchX, touchY, currentPressure, currentSize,
                    System.currentTimeMillis(),
                    currentActionId,
                    accX, accY, accZ, gyroX, gyroY, gyroZ, magX, magY, magZ
            );

            dataLock.lock();
            try {
                inferRawData.add(data);
            } finally {
                dataLock.unlock();
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {}
    };

    private void handleTouchEvent(MotionEvent event) {
        if (event == null) return;

        long time = System.currentTimeMillis();
        float rawX = event.getRawX();
        float rawY = event.getRawY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                mainContainer.removeCallbacks(liftFinishRunnable);
                isCurrentTouchFinished = false;
                currentActionId++;
                currentTouchStart = time;
                currentTouchMaxLiftTime = 0;
                Log.d("TouchFix", "新动作开始（ID=" + currentActionId + "）：tstart=" + currentTouchStart);

                currentTouchType = "按下";
                currentTouchDirection = "无";
                currentTouchX = rawX;
                currentTouchY = rawY;
                currentPressure = event.getPressure();
                currentSize = event.getSize();
                currentMotionEvent = MotionEvent.obtain(event);

                saveTouchDataToCache("按下", "无", rawX, rawY, currentPressure, currentSize, time);
                break;

            case MotionEvent.ACTION_MOVE:
                currentTouchType = "滑动中";
                currentTouchX = rawX;
                currentTouchY = rawY;
                currentPressure = event.getPressure();
                currentSize = event.getSize();
                currentMotionEvent = MotionEvent.obtain(event);

                saveTouchDataToCache("滑动中", "无", rawX, rawY, currentPressure, currentSize, time);
                Log.d("TouchFix", "[滑动中] 动作ID=" + currentActionId);
                break;

            case MotionEvent.ACTION_UP:
                currentTouchType = "抬起";
                currentTouchDirection = getSlideDirection(event.getX(), event.getY());
                currentTouchX = rawX;
                currentTouchY = rawY;
                currentPressure = event.getPressure();
                currentSize = event.getSize();
                currentMotionEvent = null;

                currentTouchMaxLiftTime = time;
                Log.d("TouchFix", "[抬起] 动作ID=" + currentActionId + "，时间=" + time);

                saveTouchDataToCache("抬起", currentTouchDirection, rawX, rawY, currentPressure, currentSize, time);

                mainContainer.removeCallbacks(liftFinishRunnable);
                mainContainer.postDelayed(liftFinishRunnable, LIFT_WAIT_DELAY);
                break;

            default:
                if (!"按下".equals(currentTouchType) && !"滑动中".equals(currentTouchType) && !"抬起".equals(currentTouchType)) {
                    currentTouchType = "无触摸";
                    currentMotionEvent = null;
                }
                break;
        }
    }

    private void saveTouchDataToCache(String touchType, String direction, float x, float y, float pressure, float size, long time) {
        if (!isInferring) return;

        AllDataModel touchData = new AllDataModel(
                touchType, direction,
                x, y, pressure, size,
                time,
                currentActionId,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f
        );

        dataLock.lock();
        try {
            inferRawData.add(touchData);
        } finally {
            dataLock.unlock();
        }
    }

    private void handleGameSlide(MotionEvent event) {
        if (event == null) return;

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touchStartX = event.getX();
                touchStartY = event.getY();
                break;
            case MotionEvent.ACTION_UP:
                getSlideDirection(event.getX(), event.getY());
                break;
        }
    }

    private String getSlideDirection(float endX, float endY) {
        float dx = endX - touchStartX;
        float dy = endY - touchStartY;

        if (Math.abs(dx) > Math.abs(dy)) {
            if (Math.abs(dx) > dpToPx(MIN_SLIDE_DISTANCE)) {
                GameManager.Direction dir = dx > 0 ? GameManager.Direction.RIGHT : GameManager.Direction.LEFT;
                if (gameManager != null) gameManager.slide(dir);
                return dx > 0 ? "右滑" : "左滑";
            }
        } else {
            if (Math.abs(dy) > dpToPx(MIN_SLIDE_DISTANCE)) {
                GameManager.Direction dir = dy > 0 ? GameManager.Direction.DOWN : GameManager.Direction.UP;
                if (gameManager != null) gameManager.slide(dir);
                return dy > 0 ? "下滑" : "上滑";
            }
        }
        return "无效滑动";
    }

    private void initGame() {
        if (gameManager != null) gameManager = null;
        gameManager = new GameManager(this, gameContainer);
        gameManager.setOnGameUpdateListener(addedScore -> {
            totalScore += addedScore;
            tvScore.setText("得分: " + totalScore);
        });
        gameManager.initAfterMeasure();
    }

    private int dpToPx(int dp) {
        return (int) (dp * getResources().getDisplayMetrics().density);
    }

    private int getSensorMaxFrequency(Sensor sensor) {
        if (sensor == null) return 5;
        float minDelay = sensor.getMinDelay();
        if (minDelay <= 0) return 50;
        int maxFreq = (int) (1000000 / minDelay);
        return Math.max(5, Math.min(maxFreq, 100));
    }

    private final Runnable liftFinishRunnable = () -> {
        if (!isCurrentTouchFinished && currentTouchStart > 0 && currentTouchMaxLiftTime > 0 && currentActionId > 0) {
            long finalTend = currentTouchMaxLiftTime;
            Log.d("TouchFix", "动作ID=" + currentActionId + " 结束：tend=" + finalTend);

            dataLock.lock();
            try {
                touchIntervals.add(new long[]{currentTouchStart, finalTend});
            } finally {
                dataLock.unlock();
            }

            mainContainer.postDelayed(() -> {
                currentTouchType = "无触摸";
                currentTouchDirection = "无";
                currentTouchX = 0f;
                currentTouchY = 0f;
                currentPressure = 0f;
                currentSize = 0f;
                currentTouchStart = 0;
                currentTouchMaxLiftTime = 0;
                isCurrentTouchFinished = true;
            }, 30);
        }
    };

    @Override
    protected void onStop() {
        super.onStop();
        isInferring = false;

        if (collectHandler != null && collectTimeoutRunnable != null) {
            collectHandler.removeCallbacks(collectTimeoutRunnable);
        }

        dataLock.lock();
        try {
            touchIntervals.clear();
            inferRawData.clear();
            currentTouchStart = 0;
            currentTouchMaxLiftTime = 0;
            isCurrentTouchFinished = true;
            currentActionId = 0;
        } finally {
            dataLock.unlock();
        }

        if (isSensorRegistered) {
            try {
                sensorManager.unregisterListener(sensorListener);
                isSensorRegistered = false;
            } catch (Exception e) {
                Log.e("SensorInferDebug", "[SENSOR_UNREG] 应用退后台，传感器注销失败：" + e.getMessage(), e);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        isInferring = false;

        mainContainer.removeCallbacks(liftFinishRunnable);

        if (collectHandler != null) {
            collectHandler.removeCallbacksAndMessages(null);
            collectHandler = null;
        }
        collectTimeoutRunnable = null;

        dataLock.lock();
        try {
            touchIntervals.clear();
            inferRawData.clear();
            inferScores.clear();
            currentTouchStart = 0;
            currentTouchMaxLiftTime = 0;
            isCurrentTouchFinished = true;
            currentActionId = 0;
        } finally {
            dataLock.unlock();
        }

        if (sensorManager != null && isSensorRegistered) {
            try {
                sensorManager.unregisterListener(sensorListener);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if (sensorThread != null) {
            sensorThread.quitSafely();
            try {
                sensorThread.join(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        currentMotionEvent = null;
    }

    public static class SaveDataWorker extends androidx.work.Worker {
        public SaveDataWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
            super(context, workerParams);
        }

        @NonNull
        @Override
        public Result doWork() {
            try {
                Data inputData = getInputData();
                String tempFilePath = inputData.getString("tempFilePath");
                String targetCsvPath = inputData.getString("targetCsvPath");
                boolean isRawData = inputData.getBoolean("isRawData", true);

                if (tempFilePath == null || targetCsvPath == null) {
                    return Result.failure(new Data.Builder().putString("error", "参数缺失").build());
                }
                File tempFile = new File(tempFilePath);
                if (!tempFile.exists()) {
                    return Result.failure(new Data.Builder().putString("error", "临时文件不存在").build());
                }
                File targetCsvFile = new File(targetCsvPath);

                List<String> lines = Files.readAllLines(tempFile.toPath(), Charset.forName("UTF-8"));
                if (lines.isEmpty()) {
                    return Result.failure(new Data.Builder().putString("error", "临时文件无数据").build());
                }

                Gson gson = new Gson();
                FileOutputStream fos = new FileOutputStream(targetCsvFile, true);
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fos, StandardCharsets.UTF_8), 8192);

                try {
                    if (!targetCsvFile.exists()) {
                        fos.write(0xef);
                        fos.write(0xbb);
                        fos.write(0xbf);

                        if (isRawData) {
                            writer.write("事件类型,X坐标,Y坐标,时间戳(毫秒),压力,面积,滑动方向,动作ID," +
                                    "加速度X,加速度Y,加速度Z,角速度X,角速度Y,角速度Z,磁场X,磁场Y,磁场Z\n");
                        } else {
                            writer.write(FeatureModel.getCsvHeader() + "\n");
                        }
                    }

                    if (isRawData) {
                        List<AllDataModel> rawData = gson.fromJson(lines.get(0), new TypeToken<List<AllDataModel>>() {}.getType());
                        for (AllDataModel data : rawData) {
                            if (data != null) {
                                writer.write(data.toCsv() + "\n");
                            }
                        }
                    } else {
                        List<FeatureModel> featureData = gson.fromJson(lines.get(0), new TypeToken<List<FeatureModel>>() {}.getType());
                        for (FeatureModel data : featureData) {
                            if (data != null && data.features != null && data.features.length == 74) {
                                writer.write(data.toCsv() + "\n");
                            }
                        }
                    }

                    writer.flush();
                    return Result.success();
                } catch (Exception e) {
                    e.printStackTrace();
                    return Result.failure(new Data.Builder().putString("error", e.getMessage()).build());
                } finally {
                    writer.close();
                    fos.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
                return Result.failure(new Data.Builder().putString("error", "保存失败：" + e.getMessage()).build());
            }
        }
    }
}