package com.example.infer;

public class TouchData {
    private String type;
    private float x;
    private float y;
    private long time;
    private float pressure;
    private float size;

    public TouchData(String type, float x, float y, long time, float pressure, float size) {
        this.type = type;
        this.x = x;
        this.y = y;
        this.time = time;
        this.pressure = pressure;
        this.size = size;
    }

    public String getType() { return type; }
    public float getX() { return x; }
    public float getY() { return y; }
    public long getTime() { return time; }
    public float getPressure() { return pressure; }
    public float getSize() { return size; }

    public String toDisplayText() { return ""; }
    public String toCsv() { return ""; }
}
