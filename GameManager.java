package com.example.infer;

import android.content.Context;
import android.graphics.Typeface;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GameManager {
    private static final int SIZE = 4;
    private static final int GAP = 8; // 格子间距（dp）
    private Block[][] grid = new Block[SIZE][SIZE]; // 引用外部Block类
    private LinearLayout gameContainer;
    private Context context;
    private Random random = new Random();
    private OnGameUpdateListener listener;

    // 游戏得分更新回调接口
    public interface OnGameUpdateListener {
        void onUpdate(int score);
    }

    public GameManager(Context context, LinearLayout gameContainer) {
        this.context = context;
        this.gameContainer = gameContainer;
    }

    // 初始化网格（强制正方形，解决尺寸漂移问题）
    public void initAfterMeasure() {
        gameContainer.post(() -> {
            // 1. 计算正方形容器的边长（取屏幕较短边的80%，避免超出屏幕）
            int screenWidth = getScreenWidth();
            int screenHeight = getScreenHeight();
            int maxSide = (int) (Math.min(screenWidth, screenHeight) * 0.9);

            // 2. 固定游戏容器为正方形（避免重置后尺寸改变）
            ViewGroup.LayoutParams containerParams = gameContainer.getLayoutParams();
            containerParams.width = maxSide;
            containerParams.height = maxSide;
            gameContainer.setLayoutParams(containerParams);

            // 3. 初始化4x4网格（均匀分布）
            initGrid(maxSide);

            // 4. 初始生成1个2（游戏规则）
            addRandomBlock();
        });
    }

    // 初始化网格布局（核心：均匀分配格子尺寸）
    // 初始化网格布局（核心：让4x4格子整体上下左右边距相同）
    private void initGrid(int containerSize) {
        gameContainer.removeAllViews();
        // 关键1：统一游戏框的内边距（上下左右相同，比如12dp，控制格子整体的边距）
        int containerPadding = dpToPx(12);
        gameContainer.setPadding(containerPadding, containerPadding, containerPadding, containerPadding);

        int totalGap = dpToPx(GAP) * (SIZE - 1); // 格子总间距（4个格子有3个间距，不影响整体边距）
        // 计算每个格子的尺寸（容器尺寸 - 2*统一内边距 - 总间距）/4，确保格子均匀分布
        int cellSize = (containerSize - 2 * containerPadding - totalGap) / SIZE;

        // 生成4行，每行4个格子
        for (int y = 0; y < SIZE; y++) {
            LinearLayout row = new LinearLayout(context);
            LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, cellSize);
            // 关键2：行之间的间距（仅中间行有，上下无，不影响整体上下边距）
            if (y > 0) {
                rowParams.setMargins(0, dpToPx(GAP), 0, 0); // 仅上行与下行之间有间距
            }
            row.setLayoutParams(rowParams);
            row.setOrientation(LinearLayout.HORIZONTAL);

            // 每行生成4个格子
            for (int x = 0; x < SIZE; x++) {
                TextView tv = new TextView(context);
                LinearLayout.LayoutParams cellParams = new LinearLayout.LayoutParams(
                        cellSize, cellSize);
                // 关键3：列之间的间距（仅中间列有，左右无，不影响整体左右边距）
                if (x > 0) {
                    cellParams.setMarginStart(dpToPx(GAP)); // 仅左列与右列之间有间距
                }
                // 格子样式（居中、粗体等，不变）
                tv.setLayoutParams(cellParams);
                tv.setGravity(Gravity.CENTER);
                tv.setTextSize(24);
                tv.setTypeface(Typeface.DEFAULT_BOLD);
                tv.setBackgroundColor(0xffcdc1b4);

                row.addView(tv);
                grid[x][y] = new Block(x, y, 0, tv);
            }
            gameContainer.addView(row);
        }
    }

    // 获取屏幕宽度（无过时API）
    private int getScreenWidth() {
        WindowManager wm = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics metrics = new DisplayMetrics();
        wm.getDefaultDisplay().getMetrics(metrics);
        return metrics.widthPixels;
    }

    // 获取屏幕高度（无过时API）
    private int getScreenHeight() {
        WindowManager wm = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics metrics = new DisplayMetrics();
        wm.getDefaultDisplay().getMetrics(metrics);
        return metrics.heightPixels;
    }

    // 随机生成一个值为2的格子
    public void addRandomBlock() {
        List<Block> emptyBlocks = new ArrayList<>();
        // 查找所有空格子（值为0）
        for (int x = 0; x < SIZE; x++) {
            for (int y = 0; y < SIZE; y++) {
                if (grid[x][y].getValue() == 0) {
                    emptyBlocks.add(grid[x][y]);
                }
            }
        }
        // 随机选择一个空格子设置为2
        if (!emptyBlocks.isEmpty()) {
            Block block = emptyBlocks.get(random.nextInt(emptyBlocks.size()));
            block.setValue(2); // 调用外部Block类的setValue方法（自动更新显示）
        }
    }

    // 滑动逻辑（上下左右）
    public boolean slide(Direction direction) {
        boolean moved = false;
        int score = 0;

        switch (direction) {
            case LEFT:
                for (int y = 0; y < SIZE; y++) {
                    List<Integer> values = new ArrayList<>();
                    // 提取当前行非零值
                    for (int x = 0; x < SIZE; x++) {
                        if (grid[x][y].getValue() != 0) {
                            values.add(grid[x][y].getValue());
                        }
                    }
                    // 合并相邻相同值
                    List<Integer> merged = mergeValues(values);
                    // 计算得分
                    score += calculateScore(values, merged);
                    // 回填到网格（有变化才标记移动）
                    if (!values.equals(merged)) {
                        moved = true;
                        fillGridRow(y, merged, Direction.LEFT);
                    }
                }
                break;

            case RIGHT:
                for (int y = 0; y < SIZE; y++) {
                    List<Integer> values = new ArrayList<>();
                    // 从右到左提取非零值
                    for (int x = SIZE - 1; x >= 0; x--) {
                        if (grid[x][y].getValue() != 0) {
                            values.add(grid[x][y].getValue());
                        }
                    }
                    // 合并相邻相同值
                    List<Integer> merged = mergeValues(values);
                    // 计算得分
                    score += calculateScore(values, merged);
                    // 回填到网格（有变化才标记移动）
                    if (!values.equals(merged)) {
                        moved = true;
                        fillGridRow(y, merged, Direction.RIGHT);
                    }
                }
                break;

            case UP:
                for (int x = 0; x < SIZE; x++) {
                    List<Integer> values = new ArrayList<>();
                    // 从上到下提取非零值
                    for (int y = 0; y < SIZE; y++) {
                        if (grid[x][y].getValue() != 0) {
                            values.add(grid[x][y].getValue());
                        }
                    }
                    // 合并相邻相同值
                    List<Integer> merged = mergeValues(values);
                    // 计算得分
                    score += calculateScore(values, merged);
                    // 回填到网格（有变化才标记移动）
                    if (!values.equals(merged)) {
                        moved = true;
                        fillGridColumn(x, merged, Direction.UP);
                    }
                }
                break;

            case DOWN:
                for (int x = 0; x < SIZE; x++) {
                    List<Integer> values = new ArrayList<>();
                    // 从下到上提取非零值
                    for (int y = SIZE - 1; y >= 0; y--) {
                        if (grid[x][y].getValue() != 0) {
                            values.add(grid[x][y].getValue());
                        }
                    }
                    // 合并相邻相同值
                    List<Integer> merged = mergeValues(values);
                    // 计算得分
                    score += calculateScore(values, merged);
                    // 回填到网格（有变化才标记移动）
                    if (!values.equals(merged)) {
                        moved = true;
                        fillGridColumn(x, merged, Direction.DOWN);
                    }
                }
                break;
        }

        // 移动后生成新格子并回调得分
        if (moved) {
            addRandomBlock();
            if (listener != null) {
                listener.onUpdate(score);
            }
        }
        return moved;
    }

    // 合并相邻相同值（通用方法）
    private List<Integer> mergeValues(List<Integer> values) {
        List<Integer> merged = new ArrayList<>();
        for (int i = 0; i < values.size(); i++) {
            if (i < values.size() - 1 && values.get(i).equals(values.get(i + 1))) {
                // 合并两个相同值（翻倍）
                merged.add(values.get(i) * 2);
                i++; // 跳过下一个，避免连续合并
            } else {
                merged.add(values.get(i));
            }
        }
        // 补零到4个元素
        while (merged.size() < SIZE) {
            merged.add(0);
        }
        return merged;
    }

    // 计算合并得分
    private int calculateScore(List<Integer> original, List<Integer> merged) {
        int originalSum = original.stream().mapToInt(Integer::intValue).sum();
        int mergedSum = merged.stream().mapToInt(Integer::intValue).sum();
        return mergedSum - originalSum;
    }

    // 回填行数据
    private void fillGridRow(int rowIndex, List<Integer> values, Direction direction) {
        if (direction == Direction.LEFT) {
            for (int x = 0; x < SIZE; x++) {
                grid[x][rowIndex].setValue(values.get(x)); // 调用外部Block的setValue
            }
        } else if (direction == Direction.RIGHT) {
            for (int x = 0; x < SIZE; x++) {
                int valueIndex = SIZE - 1 - x;
                grid[x][rowIndex].setValue(values.get(valueIndex)); // 调用外部Block的setValue
            }
        }
    }

    // 回填列数据
    private void fillGridColumn(int colIndex, List<Integer> values, Direction direction) {
        if (direction == Direction.UP) {
            for (int y = 0; y < SIZE; y++) {
                grid[colIndex][y].setValue(values.get(y)); // 调用外部Block的setValue
            }
        } else if (direction == Direction.DOWN) {
            for (int y = 0; y < SIZE; y++) {
                int valueIndex = SIZE - 1 - y;
                grid[colIndex][y].setValue(values.get(valueIndex)); // 调用外部Block的setValue
            }
        }
    }

    // 重置游戏（重新初始化，保持尺寸不变）
    public void resetGame() {
        initAfterMeasure();
        if (listener != null) {
            listener.onUpdate(0);
        }
    }

    // dp转px（适配不同屏幕密度）
    private int dpToPx(int dp) {
        return (int) (dp * context.getResources().getDisplayMetrics().density);
    }

    // 设置得分更新监听器
    public void setOnGameUpdateListener(OnGameUpdateListener listener) {
        this.listener = listener;
    }

    // 滑动方向枚举
    public enum Direction {
        UP, DOWN, LEFT, RIGHT
    }
}