package com.example.test4.mymqtt;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.os.MessageQueue;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.test4.R;

import org.eclipse.paho.client.mqttv3.MqttException;

import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class MqttService extends Service implements MqttListener {

    private class CheckMqttThread  extends TimerTask {
        @Override
        public void run() {
            handler.sendEmptyMessage(MESSAGE_CHECK);
        }
    }


    private static final int MESSAGE_CHECK=0;
//    configuration of all mqtt
    private static MqttManeger mqttManeger;
//    interfaces list
    private static List<MqttListener> mqttListenerList=new ArrayList<>();
//     used to check if need to update
    private CheckMqttThread my_thread;
//    a timer used to count some period to ensure the MQTT is still online
    private Timer timer=new Timer(true);

    private Integer tt=0;

//    a handler used to connect MQTT--ensure the connection
    private Handler handler=new Handler(){
        @Override
        public void handleMessage(@NonNull Message msg) {
            if(msg.what==MESSAGE_CHECK){
                if(mqttManeger!=null&&!mqttManeger.isHas_connected()){
                    mqttManeger.conneMqtt();
                }
            }
        }
    };

//    mainly used to change the topic of the subscribe msg
//    -->> want to use only one back service
    private static String cur_topic=null;
    public static void setCur_topic(String cur_topic) {
        MqttService.cur_topic = cur_topic;
    }

//    initialize the MQTT manager and connect the server
    @Override
    public void onCreate() {
        super.onCreate();
        mqttManeger=new MqttManeger(this);
        mqttManeger.conneMqtt();
    }

//    used to show notification in the outside UI-->info push
    private Notification get_notification(){
        Notification notification=null;
        NotificationManager notificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
//        设置Notification的ChannelID,否则不能正常显示
//        ensure the API version-->the minimum version of the project is 25, very embarrassed
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel mChannel = new NotificationChannel("mqtt", getResources().getString(R.string.app_name), NotificationManager.IMPORTANCE_HIGH);
            notificationManager.createNotificationChannel(mChannel);
            notification = new Notification.Builder(getApplicationContext(), "mqtt").build();
        }else {
            Notification.Builder builder = new Notification.Builder(this)
                    .setSmallIcon(R.mipmap.ic_launcher)
                    .setContentTitle(getResources().getString(R.string.app_name))
                    .setContentText("");
            notification = builder.build();
        }
        return notification;
    }

//
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
//        check if connect the MQTT server on time
        if(my_thread==null){
            my_thread=new CheckMqttThread();
            timer.scheduleAtFixedRate(my_thread,2000,10000);
        }

//        start the foreground service
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.O){
            startForeground(1,get_notification());
        }

        return Service.START_STICKY;
    }


//    add mqttlis-->different operation for different infomation
    public static void add_mqttListener(MqttListener mqttListener){
        if(!mqttListenerList.contains(mqttListener)){
            mqttListenerList.add(mqttListener);
        }
    }

    public static void rm_mqttListener(MqttListener mqttListener){
        mqttListenerList.remove(mqttListener);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }


    /**************** MQTT Listener *****************/
//    just a summary of all mqtt listeners
//    different pages has differnet listeners for different usage
    @Override
    public void onConnected() {
        Log.i("mqtt-test","Has connected MQTT server!");
        if(mqttManeger!=null){
            try {
                mqttManeger.subcribe_msg("#",1);
            } catch (MqttException e) {
                e.printStackTrace();
            }
        }
        for (MqttListener mqttListener:mqttListenerList){
            mqttListener.onConnected();
        }
    }

    @Override
    public void onFail() {
        Log.i("mqtt","Fail to connect MQTT server");
        if(mqttManeger!=null){
            mqttManeger.reconnMqtt();
        }
        for(MqttListener mqttListener:mqttListenerList){
            mqttListener.onFail();
        }
    }

    @Override
    public void onLost() {
        Log.i("mqtt","Lost MQTT server");
        if(mqttManeger!=null){
            mqttManeger.reconnMqtt();
        }
        for(MqttListener mqttListener:mqttListenerList){
            mqttListener.onLost();
        }
    }

    @Override
    public void onReceive(String topic, String message) {
        Log.i("mqtt","Receive msg from MQTT server:"+message);
        for(MqttListener mqttListener:mqttListenerList){
            mqttListener.onReceive(topic,message);
        }
    }

    @Override
    public void onSendSucc() {
        Log.i("mqtt","Succeed to send MSG!");
        for(MqttListener mqttListener:mqttListenerList){
            mqttListener.onSendSucc();
        }
    }
}
