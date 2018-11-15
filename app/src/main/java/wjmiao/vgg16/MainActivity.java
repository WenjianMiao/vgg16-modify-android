package wjmiao.vgg16;

import android.Manifest;
import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.NumberPicker;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.TreeMap;

import static java.security.AccessController.getContext;


public class MainActivity extends AppCompatActivity {

    // Timely Stop Threshold: threshold
    float threshold = (float) 0.8;
    NumberPicker nPicker1,nPicker2;
    TextView result_tv1;
    TextView result_tv2;
    TextView result_tv3;
    TextView result_tv4;
    Button xor_bt;
    String modelFile0="ModifyConv10/converted_model0.tflite";
    String modelFile1="ModifyConv10/converted_model1.tflite";
    String modelFile2="ModifyConv10/converted_model2.tflite";
    String modelFile3="ModifyConv10/converted_model3.tflite";

    // tflite0: input->observing
    // tflite1: input->mid_feature_map
    // tflite2: mid_feature_map->original output
    // tflite3: input->original output
    Interpreter tflite0;
    Interpreter tflite1;
    Interpreter tflite2;
    Interpreter tflite3;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result_tv1=findViewById(R.id.result_tv1);
        result_tv2=findViewById(R.id.result_tv2);
        result_tv3=findViewById(R.id.result_tv3);
        result_tv4=findViewById(R.id.result_tv4);
        xor_bt=findViewById(R.id.bt);

        try {
            tflite0 = new Interpreter(loadModelFile(this, modelFile0));
            tflite1 = new Interpreter(loadModelFile(this, modelFile1));
            tflite2 = new Interpreter(loadModelFile(this, modelFile2));
            tflite3 = new Interpreter(loadModelFile(this, modelFile3));
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        final android.content.Context pp = this;

        xor_bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    AssetManager assetManager = getAssets();

                    // network input and ground_truth
                    float [][][][] imgs = new float[1][32][32][3];
                    float [][] labels = new float[10000][100];

                    // network inference result
                    float [][][][] mid_feature_map = new float[1][2][2][512]; // may be changed
                    float [][] output_logits = new float[1][100];

                    float mean_red = (float) 129.74308525390626;
                    float mean_green = (float) 124.285218359375;
                    float mean_blue = (float) 112.6952638671875;
                    float std_red = (float) 68.40415141388044;
                    float std_green = (float) 65.62775279419219;
                    float std_blue = (float) 70.65942155331254;

                    int total_accuracy = 0;
                    int total_stop = 0;

                    int original_accuracy = 0;

                    // time record: duration1: our_method  duration2: original_method
                    long startTime;
                    long duration1 = 0;
                    long duration2 = 0;

                    // read labels
                    int permission = ActivityCompat.checkSelfPermission(pp, Manifest.permission.WRITE_EXTERNAL_STORAGE);
                    if (permission != PackageManager.PERMISSION_GRANTED) {
                        // We don't have permission so prompt the user
                        ActivityCompat.requestPermissions(
                                (Activity)pp,
                                PERMISSIONS_STORAGE,
                                REQUEST_EXTERNAL_STORAGE
                        );
                    }

                    String label_path="/test_label.txt";
                    label_path = Environment.getExternalStorageDirectory().getAbsolutePath() + label_path;
                    Scanner scanner = new Scanner(new File(label_path));

                    for (int i=0; i<10000; i++) {
                        int label = scanner.nextInt();
                        labels[i][label] = (float) 1.0;
                    }

                    // main task loop
                    for (int i=0; i<500; i++) {

                        // read image
                        InputStream inputStream = assetManager.open("image/"+Integer.toString(i)+".png");
                        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                        int [][] rgbvalue = new int[bitmap.getHeight()][bitmap.getWidth()];
                        for(int j=0; j<bitmap.getHeight();j++) {
                            for(int k=0; k<bitmap.getWidth();k++) {
                                rgbvalue[j][k] = bitmap.getPixel(k,j);
                                int red = Color.red(rgbvalue[j][k]);
                                int green = Color.green(rgbvalue[j][k]);
                                int blue = Color.blue(rgbvalue[j][k]);
                                imgs[0][j][k][0] = (float) red;
                                imgs[0][j][k][0] = (imgs[0][j][k][0] - mean_red) / std_red;
                                imgs[0][j][k][1] = (float) green;
                                imgs[0][j][k][1] = (imgs[0][j][k][1] - mean_green) / std_green;
                                imgs[0][j][k][2] = (float) blue;
                                imgs[0][j][k][2] = (imgs[0][j][k][2] - mean_blue) / std_blue;
                            }
                        }

                        // Do the inference of observing layer
                        startTime = System.currentTimeMillis();
                        tflite0.run(imgs, output_logits);
                        duration1 = duration1 + System.currentTimeMillis() - startTime;

                        if( TimelyStop(output_logits[0]) ){
                            if (Top5Accuracy(output_logits[0], labels[i])) {
                                total_accuracy++;
                                total_stop++;
                            }
                        }

                        else {

                            // Simulate that it has already run to the middle layer
                            tflite1.run(imgs, mid_feature_map);

                            // Do the inference of the full network
                            startTime = System.currentTimeMillis();
                            tflite2.run(mid_feature_map, output_logits);
                            duration1 = duration1 + System.currentTimeMillis() - startTime;

                            // test if the result is accurate in top-5
                            if (Top5Accuracy(output_logits[0], labels[i])) {
                                total_accuracy++;
                            }
                        }

                        // Simulate the original method
                        startTime = System.currentTimeMillis();
                        tflite3.run(imgs, output_logits);
                        duration2 = duration2 + System.currentTimeMillis() - startTime;

                        if (Top5Accuracy(output_logits[0], labels[i])) {
                            original_accuracy++;
                        }

                    }

                    // Show results
                    result_tv1.setText(String.valueOf(duration2));
                    result_tv2.setText(String.valueOf(duration1));
                    result_tv3.setText(String.valueOf(original_accuracy));
                    result_tv4.setText(String.valueOf(total_accuracy));

                } catch (IOException e) {
                    e.printStackTrace();
                }

            }

        });

    }

    private MappedByteBuffer loadModelFile(Activity activity,String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private boolean Top5Accuracy(float [] logits, float [] labels){
        float [] array = logits.clone();
        Arrays.sort(array);
        float top1=array[99], top2=array[98], top3=array[97],top4=array[96],top5=array[95];
        int index1=0,index2=0,index3=0,index4=0,index5=0;
        for(int l=0; l<100; l++){
            if(logits[l] == top1) {
                index1 = l;
                continue;
            }
            if(logits[l] == top2) {
                index2 = l;
                continue;
            }
            if(logits[l] == top3) {
                index3 = l;
                continue;
            }
            if(logits[l] == top4) {
                index4 = l;
                continue;
            }
            if(logits[l] == top5) {
                index5 = l;
                continue;
            }
        }
        if(labels[index1] == 1 || labels[index2] == 1 || labels[index3] == 1 || labels[index4] == 1 || labels[index5] ==1)
            return true;
        else
            return false;

    }

    private boolean TimelyStop(float [] logits) {
        float [] array = logits.clone();
        Arrays.sort(array);
        float top1=array[99], top2=array[98], top3=array[97],top4=array[96],top5=array[95];
        float sum = top1 + top2 + top3 + top4 + top5;
        if(sum < threshold)
            return false;
        else
            return true;
    }
}