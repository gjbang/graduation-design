package com.example.test4.mymqtt;

import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;

import org.eclipse.paho.client.mqttv3.IMqttActionListener;
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.IMqttToken;
import org.eclipse.paho.client.mqttv3.MqttAsyncClient;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;


public class MqttManeger {

    /*********** Variable ***********/
    private String TAG="mqtt";

//    basic information of the mqtt connection
    private  String host=MqttServerConfig.addr;
    private  String client_id=MqttServerConfig.client_id;
    private  String user_name=MqttServerConfig.user;
    private  String user_passwd=MqttServerConfig.passwd;

//    client
    private MqttAsyncClient mqttAsyncClient=null;
//    connection options
    private MqttConnectOptions mqttConnectOptions=new MqttConnectOptions();
//    an interface used to implement different action for events of mqtt
    private MqttListener mqttListener;
//    record if the connection setup
    private boolean has_connected=false;
//    count the retry times
    private int retry_times=0;


//    create thread to operate each events of mqtt
    private Handler handler=new Handler(new Handler.Callback() {
        @Override
        public boolean handleMessage(@NonNull Message msg) {
            switch (msg.arg1){
                case MqttTag.MQTT_CONNECTED:
                    Log.i("mqtt-test","manager!");
                    mqttListener.onConnected();
                    retry_times=0;
                    break;
                case MqttTag.MQTT_FAIL:
                    mqttListener.onFail();
                    break;
                case MqttTag.MQTT_LOST:
                    mqttListener.onLost();
                    break;
                case MqttTag.MQTT_RECEIVE:
                    MqttObject obj=(MqttObject) msg.obj;
                    mqttListener.onReceive(obj.getTopic(),obj.getMessage());
                    break;
                case MqttTag.MQTT_SENDSUCC:
                    mqttListener.onSendSucc();
                    break;
            }
            return true;
        }
    });

//    A default class of mqtt, which can listen whether mqtt succeed to connect
    private IMqttActionListener iMqttActionListener=new IMqttActionListener() {
        @Override
        public void onSuccess(IMqttToken asyncActionToken) {
            has_connected=true;
            Message msg=new Message();
            msg.arg1=MqttTag.MQTT_CONNECTED;
//            Log.i("mqtt-test","s-manager!");
            handler.sendMessage(msg);
        }

        @Override
        public void onFailure(IMqttToken asyncActionToken, Throwable exception) {
            has_connected=false;
            Message msg=new Message();
            msg.arg1=MqttTag.MQTT_FAIL;
            handler.sendMessage(msg);
        }
    };

//    A default class of mqtt, used to handle 3 events
    private MqttCallback mqttCallback=new MqttCallback() {
        @Override
        public void connectionLost(Throwable cause) {
            has_connected=false;
            Message msg=new Message();
            msg.arg1=MqttTag.MQTT_LOST;
            handler.sendMessage(msg);
        }

        @Override
        public void messageArrived(String topic, MqttMessage message) throws Exception {
            Message msg=new Message();
            msg.arg1=MqttTag.MQTT_RECEIVE;
            msg.obj=new MqttObject(topic,new String(message.getPayload()));
            handler.sendMessage(msg);
        }

        @Override
        public void deliveryComplete(IMqttDeliveryToken token) {
            Message msg=new Message();
            msg.arg1=MqttTag.MQTT_SENDSUCC;
            handler.sendMessage(msg);
        }
    };

    /*************** Method ***************/

//    Constructor of this class
    public MqttManeger(MqttListener mqttListener){
        this.mqttListener=mqttListener;
    }

//    information initialize functions
    public void setMqttInfo(String host,String client_id){
        this.host=host;
        this.client_id=client_id;
    }

//    set the connection setting
    public void setMqttConnectOptions(){
        mqttConnectOptions.setCleanSession(true);
        mqttConnectOptions.setConnectionTimeout(10);
        mqttConnectOptions.setKeepAliveInterval(100);
        mqttConnectOptions.setUserName("axv");
        mqttConnectOptions.setPassword("123456".toCharArray());
    }

    public void setConnectionOptions(boolean cleanSession,int timeout,int alive_time){
        mqttConnectOptions.setCleanSession(cleanSession);
        mqttConnectOptions.setConnectionTimeout(timeout);
        mqttConnectOptions.setKeepAliveInterval(alive_time);
    }

//    connect Mqtt
    public void conneMqtt(){
        try{
            setMqttConnectOptions();

            Log.e("bugs-null","value:"+this.host);
            mqttAsyncClient=new MqttAsyncClient(this.host,this.client_id,new MemoryPersistence());
            mqttAsyncClient.connect(mqttConnectOptions,null,iMqttActionListener);
            mqttAsyncClient.setCallback(mqttCallback);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }

//    disconnect Mqtt
    public void disconnMqtt(){
        try {
            mqttAsyncClient.disconnect();
            has_connected=false;
            mqttAsyncClient=null;
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }


//    reconnect Mqtt
    public void reconnMqtt(){
        if(retry_times++<5){
            disconnMqtt();
            conneMqtt();
        }else{
            Log.e(TAG,"Cannot connect the MQTT server!");
        }
    }

//    publish msg to MQTT server
    public void publish_msg(String topic,String msg, int QoS) throws MqttException {
        if(has_connected){
            mqttAsyncClient.publish(topic,msg.getBytes(),QoS,false);
        }else{
            Log.e(TAG,"Client hasn't connect the server");
        }
    }

//    public void publish_msg(String)

//    subscribe msg for MQTT server
    public void subcribe_msg(String topic,int QoS) throws MqttException {
        if(has_connected){
            mqttAsyncClient.subscribe(topic,QoS);
        }else{
            Log.e(TAG,"Client hasn't connect the server");
        }
    }

//    some getter
    public boolean isHas_connected(){
        return has_connected;
    }
    public  String getClient_id(){
        return client_id;
    }
}
