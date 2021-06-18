package com.example.test4.ui.notifications;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;

import com.example.test4.R;
import com.example.test4.mymqtt.MqttListener;
import com.example.test4.mymqtt.MqttService;

import static androidx.core.content.ContextCompat.startForegroundService;

public class NotificationsFragment extends Fragment implements MqttListener {

    private NotificationsViewModel notificationsViewModel;

    private ImageView img_face_i;
    private TextView face_label_t;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        notificationsViewModel =
                ViewModelProviders.of(this).get(NotificationsViewModel.class);
        View root = inflater.inflate(R.layout.fragment_notifications, container, false);

        img_face_i=root.findViewById(R.id.img_face);
        face_label_t=root.findViewById(R.id.face_label);

        return root;
    }


    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

//        add MQTT listener
        MqttService.add_mqttListener(this);
//        need to bind the UI activity and the background service
        Intent intent=new Intent(getActivity(),MqttService.class);
//        change the current msg-->different page has different topic
//        TODO: may use '#' to receive all msg, and different listener to select the topic they need
        MqttService.setCur_topic("pic_data");

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
        if(topic.equals("pic")){
            Log.i("face-test","msg:receive data:");
            if(!message.isEmpty()){
                String label=message.substring(0,message.indexOf('#'));
//                message=message.substring(message.indexOf('#')+1);
//                String label=message.substring(0,message.indexOf('#'));
                String code=message.substring(message.indexOf('#')+1);
                byte[] decode= Base64.decode(code,Base64.DEFAULT);
                Bitmap decodedByte = BitmapFactory.decodeByteArray(decode, 0, decode.length);
                img_face_i.setImageBitmap(decodedByte);
                face_label_t.setText(label);
            }
        }
    }

    @Override
    public void onSendSucc() {

    }
}