package com.example.test4.ui.dashboard;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;

import com.example.test4.R;
import com.example.test4.mymqtt.MqttListener;
import com.example.test4.mymqtt.MqttService;
import com.google.gson.Gson;

import org.eclipse.paho.client.mqttv3.MqttMessage;

import java.util.ArrayList;
import java.util.List;

import static androidx.core.content.ContextCompat.startForegroundService;

public class DashboardFragment extends Fragment implements MqttListener {

    private DashboardViewModel dashboardViewModel;

    private TextView dht_temp_t,dht_humd_t,dht_time_t;

//    initialize the UI View
    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        dashboardViewModel =
                ViewModelProviders.of(this).get(DashboardViewModel.class);
        View root = inflater.inflate(R.layout.fragment_dashboard, container, false);

        dht_temp_t=root.findViewById(R.id.dht_temp_txt);
//        dht_humd_t=root.findViewById(R.id.dht_humd_txt);
        dht_time_t=root.findViewById(R.id.dht_time_txt);
        return root;
    }


//    Some vars used to operate the data
    private List<String> messageList=new ArrayList<>();
    private String str="";


    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

//        add MQTT listener
        MqttService.add_mqttListener(this);
//        need to bind the UI activity and the background service
        Intent intent=new Intent(getActivity(),MqttService.class);
//        change the current msg-->different page has different topic
//        TODO: may use '#' to receive all msg, and different listener to select the topic they need
        MqttService.setCur_topic("dht_data");

//        TODO: this function is only used for >=API 8.0, so may be need to support other version's API
        startForegroundService(getActivity(),intent);

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        MqttService.rm_mqttListener(this);
    }

    @Override
    public void onConnected() {

    }

    @Override
    public void onFail() {

    }

    @Override
    public void onLost() {

    }

    @Override
    public void onReceive(String topic, String message) {
        if(topic.equals("fall")){
            Log.i("bugs-null","msg:"+message);
            if(!message.isEmpty()){
                String[] data=message.split("#");
                dht_time_t.setText(data[0]);
                dht_temp_t.setText(data[1]);
//                dht_humd_t.setText(data[1].split(",")[1]);
            }

        }
    }

    @Override
    public void onSendSucc() {

    }
}