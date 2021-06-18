package com.example.test4.mymqtt;

public interface MqttListener {
    void onConnected();
    void onFail();
    void onLost();
    void onReceive(String topic,String message);
    void onSendSucc();
}
