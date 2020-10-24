package com.example.opencvcamera;

import android.content.ContentResolver;
import android.net.Uri;
import android.os.Environment;
import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.StrictMath.max;
import static java.lang.StrictMath.min;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.LINE_8;
import static org.opencv.imgproc.Imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.resize;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    public CameraBridgeViewBase RGBCamera;
    BaseLoaderCallback baseLoaderCallback;

    Bitmap ThermalView;
    int  Bitmap_x ,Bitmap_y;
    public static DecimalFormat df = new DecimalFormat("0.00");
    public int  screen_h, screen_w;


//    String source_path = getFilesDir().toString()+"/src/main/res";

    Mat frame, raw;
    long  ti, tc;
    double fps, fps1, fps2, fps3;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        test_open_file();

        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int screen_h = displayMetrics.heightPixels;
        int screen_w = displayMetrics.widthPixels;

        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{Manifest.permission.CAMERA},
                1);

        RGBCamera = (JavaCameraView)findViewById(R.id.CameraView);
        RGBCamera.setVisibility(SurfaceView.VISIBLE);
        RGBCamera.setCvCameraViewListener(this);
//        RGBCamera.enableFpsMeter();

        ImageView imageView = (ImageView) findViewById(R.id.imageView);
        imageView.setImageBitmap(BitmapFactory.decodeFile("/raw/land.bmp"));

        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        RGBCamera.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };


    }

    public File file2;
    public File file3;
    public void test_open_file(){
        String path = Environment.getExternalStorageDirectory().getAbsolutePath().toString();
        File file = new File(path,"test.txt");
        Log.e("test", path);
        Log.e("test", file.toString());

        if(file.exists()) Log.e("test", "exist");
        else Log.e("test", "load file fail");

        String file2_dir ="/mnt/sdcard/dnn_model/yolov3-tiny.cfg";
        file2 = new File(file2_dir);
        if(file2.exists()) Log.e("test", "file2 exist");
        else Log.e("test", "load file2 fail");

        String file3_dir ="/mnt/sdcard/dnn_model/yolov3-tiny.weights";
        file3 = new File(file3_dir);
        if(file3.exists()) Log.e("test", "file3 exist");
        else Log.e("test", "load file3 fail");

    }

    boolean startCanny = false;
    public void Canny(View Button){
        if (startCanny == false){
            startCanny = true;
        }else {
            startCanny = false;
        }
    }

    boolean startYolo = false;
    boolean firstTimeYolo = false;
    Net tinyYolo;

    String source_path = "/mnt/sdcard/dnn_model";

    String base_path = Environment.getExternalStorageDirectory().toString();

    String tinyYoloCfg ="/mnt/sdcard/dnn_model/yolov3-tiny.cfg";
    String tinyYoloWeights =  "/mnt/sdcard/dnn_model/yolov3-tiny.weights";
///mnt/sdcard/dnn_model/yolov3-tiny.cfg
    public void Yolo(View Button){
        if (startYolo == false){
            startYolo = true;
            if (firstTimeYolo == false){
                firstTimeYolo = true;

                try{
                    tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
                    Log.i("yolo3", "Yolo3 loaded successfully");
                } catch (Exception e) {
                    Log.i("Yolo", "Yolo  loaded FAILLLLLLLLLLLLLLLLLLLLLLLLLLL");
                    e.printStackTrace();
                    Log.e("Yolo", tinyYoloCfg);
                    Log.e("Yolo", tinyYoloWeights);
                }


            }
        }
        else{
            startYolo = false;
        }
    }

//    public Uri getRawUri(String filename) {
//        return Uri.parse(ContentResolver.SCHEME_ANDROID_RESOURCE + File.pathSeparator + File.separator + getPackageName() + "/raw/" + filename);
//    }

    boolean startModel = false;
    boolean firstTimeModel = false;
    Net face_dt_net;

//    Path path = FileSystems.getDefault().getPath("logs", "access.log");
//    String base_path = Environment.getExternalStorageDirectory().toString();
//    String


    String prototxtPath = base_path+"/dnn_model/deploy.prototxt";
    String weightsPath  = base_path+"/dnn_model/res10_300x300_ssd_iter_140000.caffemodel";

//    Path path = FileSystems.getDefault().getPath("logs", "access.log");

    public void Model(View Button){
        if (startModel == false){
            startModel = true;
            if (firstTimeModel == false){
                firstTimeModel = true;
                try{
//                    face_dt_net = Dnn.readNetFromCaffe(prototxtPath, weightsPath);
                    Uri.parse("android.resource://"+getPackageName()+"/raw/deploy.prototxt");
                    getPackageResourcePath();
//                    prototxt_sur = getResources().openRawResource();

                    face_dt_net = Dnn.readNetFromCaffe(prototxtPath, );
                } catch (Exception e) {
                    Log.i("face_detection", "face detection model loaded FAILLLLLLLLLLLLLLLLLLLLLLLLLLL");
                    e.printStackTrace();
                    Log.e("face_detection", prototxtPath);
                }

//                String prototxtPath = FileSystems.getDefault().getPath("prototxt", "deploy.prototxt").toString();
//                String weightsPath = FileSystems.getDefault().getPath("caffemodel", "res10_300x300_ssd_iter_140000.caffemodel").toString();

//                Log.i("face_detection", "face detection model loaded successfully");
            }
        }
        else{
            startModel = false;
        }
    }


//deformed attempt: rotated image @ JavaCameraView.java line 229-230

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        ti = System.nanoTime();
        raw = inputFrame.rgba();
        Mat gray = raw;

        if (startCanny){
            Imgproc.cvtColor(gray, raw, COLOR_RGB2GRAY);
            Imgproc.Canny(gray, gray, 100, 80);
        }
        else if (startYolo){
            Mat imageBlob = Dnn.blobFromImage(raw, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
            tinyYolo.setInput(imageBlob);
            tinyYolo.forward();

//            raw = Yolo_face_detect(raw);
        }
        else if (startModel){
            Mat resize_frame = raw;
            resize(resize_frame, frame, raw.size());
            raw = face_dt_func(raw, resize_frame);
        }


        tc = System.nanoTime()-ti;
        int orientation = getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            // In landscape

            // flipped not fixed
            raw = AddWord(raw);
            return raw;
        } else {
            // In portrait
            // rotate and expand view into portrait
            Mat frame = raw.t();
            Core.flip(raw.t(), frame, 1);
            resize(frame, frame, raw.size());
            frame = AddWord(frame);
            return frame;
        }

//        Core.transpose(frame, frame);        // rotates Mat to portrait for opencv
//        Core.flip(frame, frame, 1);
//        Mat result = frame;                 // maintain frame as original
//
////        frame = opencv_face_detect.main(frame, result);
//
//
//        Core.transpose(frame, frame);
//        Core.flip(frame, frame, 0); // rotates Mat to portrait
//
////        Core.transpose(result, result);
////        Core.flip(result, result, 0); // rotates Mat to portrait
////        return result;
//        return frame;


//-----------------------------------------------------------------------------------------------------------

//        Mat grayMat = new Mat();
//        cvtColor(frame,grayMat,COLOR_RGB2GRAY);
//        final byte [] grayData = new byte[grayMat.cols()*grayMat.rows()];
//        grayMat.get(0,0,grayData);

//        class ImageAnalyzer implements ImageAnalysis.Analyzer {
//
//            @Override
//            public void analyze(ImageProxy imageProxy) {
//                @SuppressLint("UnsafeExperimentalUsageError") Image mediaImage = imageProxy.getImage();
//                if (mediaImage != null) {
//                    InputImage image = InputImage.fromByteArray(
//                            grayData,
//                            /* image width */480,
//                            /* image height */360,
//                            0,
//                            InputImage.IMAGE_FORMAT_NV21 // or IMAGE_FORMAT_YV12
//                    // Pass image to an ML Kit Vision API
//                            FaceDetector detector = FaceDetection.getClient();
//                    );
//                }
//            }
//        }


    }

    public Mat AddWord(Mat process){
        fps3 = fps2;
        fps2= fps1;
        fps1 = (double)  1_000_000_000/tc ;
        fps = (fps1+fps2+fps3)/3;

        String fps_output = "fps: "+df.format(fps/10);

        double processing_time = (double)  tc/1_000_000 ;
        String output = "process T: "+df.format(processing_time)+"ms";

        Imgproc.putText(process, fps_output,
                new Point(10,25),
                FONT_HERSHEY_SIMPLEX ,      // front face
                1,                               // front scale
                new Scalar(0, 200, 255),             // Scalar object for color
                4);
        Imgproc.putText(process, output,
                new Point(10,25+30),
                FONT_HERSHEY_SIMPLEX ,      // front face
                1,                               // front scale
                new Scalar(0, 200, 255),             // Scalar object for color
                4);

        return process;
    }

    public Mat Yolo_face_detect(Mat frame) {
        Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416, 416), new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
        tinyYolo.setInput(imageBlob);
        java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

        List<String> outBlobNames = new java.util.ArrayList<>();
        outBlobNames.add(0, "yolo_16");
        outBlobNames.add(1, "yolo_23");

        tinyYolo.forward(result, outBlobNames);

        float confThreshold = 0.3f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();

        for (int i = 0; i < result.size(); ++i) {

            Mat level = result.get(i);

            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());

                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold) {
                    int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                    int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                    int width = (int) (row.get(0, 2)[0] * frame.cols());
                    int height = (int) (row.get(0, 3)[0] * frame.rows());


                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);

                    rects.add(new Rect(left, top, width, height));
                }
            }
        }
        int ArrayLength = confs.size();

        if (ArrayLength >= 1) {
            // Apply non-maximum suppression procedure.
            float nmsThresh = 0.2f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            Rect[] boxesArray = rects.toArray(new Rect[0]);
            MatOfRect boxes = new MatOfRect(boxesArray);
            MatOfInt indices = new MatOfInt();
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

            // Draw result boxes:
            int[] ind = indices.toArray();
            for (int i = 0; i < ind.length; ++i) {
                int idx = ind[i];
                Rect box = boxesArray[idx];
                int idGuy = clsIds.get(idx);
                float conf = confs.get(idx);

                List<String> cocoNames = Arrays.asList("a person", "a bicycle", "a motorbike", "an airplane", "a bus", "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", "a stop sign", "a parking meter", "a car", "a bench", "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", "a kite", "a baseball bat", "a baseball glove", "a skateboard", "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", "a doughnut", "a cake", "a chair", "a sofa", "a potted plant", "a bed", "a dining table", "a toilet", "a TV monitor", "a laptop", "a computer mouse", "a remote control", "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", "a sink", "a refrigerator", "a book", "a clock", "a vase", "a pair of scissors", "a teddy bear", "a hair drier", "a toothbrush");

                int intConf = (int) (conf * 100);
                Imgproc.putText(frame, cocoNames.get(idGuy) + " " + intConf + "%", box.tl(), FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 0), 2);
                rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);
            }
        }
        return frame;
    }

    public Mat face_dt_func(Mat frame,Mat frame_lowered) {
        Mat blob;
        int w = frame.size(0);
        int h = frame.size(1);
                                                //Size(h_func,w_func)
        // construct a blob from the image
        blob = Dnn.blobFromImage(frame_lowered, 1.0,frame_lowered.size() ,new Scalar(104.0, 177.0, 123.0));
        face_dt_net.setInput(blob);
        Mat detections = face_dt_net.forward();

        for (int i= 0; i<201; i++){
            //extract the confidence (i.e., probability) associated with
            //the detection
            float confidence = (float)(detections.get(i, 2)[0]);

            //filter out weak detections by ensuring the confidence is
            //greater than the minimum confidence
            if (confidence > 0.3){
                //compute the (x, y)-coordinates of the bounding box for
                //the object

                int startX   = (int)(detections.get(i, 3)[0] * w);
                int startY    = (int)(detections.get(i, 4)[0] * h);
                int endX  = (int)(detections.get(i, 5)[0] * w);
                int endY = (int)(detections.get(i, 6)[0] * h);

                startX = max(0, startX);
                startY = max(0, startY);
                endX = min(w - 1, endX);
                endY = min(h - 1, endY);

//                (startX, startY) = (max(0, startX), max(0, startY));
//                (endX, endY) = (min(w - 1, endX), min(h - 1, endY));
                Imgproc.rectangle(frame, new Point(startX, startY), new Point(endX, endY), new Scalar(135, 247, 252),10, LINE_8 );
            }
        }
        return frame;
    }



    @Override
    public void onCameraViewStarted(int width, int height) {

    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
        super.onResume();
    }

    @Override
    protected void onPause() {
        if(RGBCamera!=null){
            RGBCamera.disableView();
        }
        super.onPause();
    }


    @Override
    protected void onDestroy() {
        if (RGBCamera!=null){
            RGBCamera.disableView();
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    RGBCamera.setCameraPermissionGranted();  // <------ THIS!!!
                } else {
                    // permission denied
                }
                return;
            }
        }
    }


}