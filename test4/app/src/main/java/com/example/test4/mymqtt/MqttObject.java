package com.example.test4.mymqtt;

public class MqttObject {
    private String topic;
    private String message;

    MqttObject(){

    }

    MqttObject(String topic,String message){
        this.topic=topic;
        this.message=message;
    }

    public String getTopic() {
        return topic;
    }

    public void setTopic(String topic) {
        this.topic = topic;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}
