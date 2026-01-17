package com.example.infer;

import android.widget.TextView;

public class Block {
    private int x;
    private int y;
    private int value;
    private TextView tv;

    public Block(int x, int y, int value, TextView tv) {
        this.x = x;
        this.y = y;
        this.value = value;
        this.tv = tv;
        updateView();
    }

    public int getX() { return x; }
    public int getY() { return y; }
    public int getValue() { return value; }

    public void setValue(int value) {
        this.value = value;
        updateView();
    }

    // 更新格子显示（颜色+文字）
    private void updateView() {
        if (value == 0) {
            tv.setText("");
            tv.setBackgroundColor(0xffcdc1b4);
        } else {
            tv.setText(String.valueOf(value));
            // 不同数值对应不同背景色（经典2048配色）
            switch (value) {
                case 2: tv.setBackgroundColor(0xffeee4da); tv.setTextColor(0xff776e65); break;
                case 4: tv.setBackgroundColor(0xffede0c8); tv.setTextColor(0xff776e65); break;
                case 8: tv.setBackgroundColor(0xfff2b179); tv.setTextColor(0xfffff4e9); break;
                case 16: tv.setBackgroundColor(0xfff59563); tv.setTextColor(0xfffff4e9); break;
                case 32: tv.setBackgroundColor(0xfff67c5f); tv.setTextColor(0xfffff4e9); break;
                case 64: tv.setBackgroundColor(0xfff65e3b); tv.setTextColor(0xfffff4e9); break;
                case 128: tv.setBackgroundColor(0xffedcf72); tv.setTextColor(0xfffff4e9); break;
                case 256: tv.setBackgroundColor(0xffedcc61); tv.setTextColor(0xfffff4e9); break;
                case 512: tv.setBackgroundColor(0xffedc850); tv.setTextColor(0xfffff4e9); break;
                case 1024: tv.setBackgroundColor(0xffedc53f); tv.setTextColor(0xfffff4e9); break;
                case 2048: tv.setBackgroundColor(0xffedc22e); tv.setTextColor(0xfffff4e9); break;
                default: tv.setBackgroundColor(0xff3c3a32); tv.setTextColor(0xfffff4e9); break;
            }
        }
    }
}