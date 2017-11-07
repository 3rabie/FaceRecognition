package facerecog;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.*;
import org.opencv.objdetect.*;
import org.opencv.face.*;
import org.opencv.utils.*;

import java.util.*;
import java.io.File;
//import java.io.IOException;

import lowgui.*;

class FaceRec {

    FaceRecognizer fr = Face.createLBPHFaceRecognizer();

    //
    // unlike the c++ demo, let's not mess with csv files, but use a folder on disk.
    //    each person should have its own subdir with images (all images the same size, ofc.)
    //   +- persons
    //     +- maria
    //       +- pic1.jpg
    //       +- pic2.jpg
    //     +- john
    //       +- pic1.jpg
    //       +- pic2.jpg
    //
    public Size loadTrainDir(String dir)
    {
        Size s = null;
        int label = 0;
        List<Mat> images = new ArrayList<Mat>();
        List<java.lang.Integer> labels = new ArrayList<java.lang.Integer>();
        File node = new File(dir);
        String[] subNode = node.list();
        for(String p : subNode){
            System.out.println(""+p);
        }
        if ( subNode==null ) return null;

        for(String person : subNode) {
          
            File subDir = new File(node, person);
            if ( ! subDir.isDirectory() ) continue;
            File[] pics = subDir.listFiles();
            for(File f : pics) {
                Mat m = Imgcodecs.imread(f.getAbsolutePath(),0);
                if (! m.empty()) {
                    images.add(m);
                    labels.add(label);
                    fr.setLabelInfo(label,subDir.getName());
                    s = m.size();
                }
            }
            label ++;
        }
        fr.train(images, Converters.vector_int_to_Mat(labels));
        return s;
    }
    public String predict(Mat img) {
        int[] id = {-1};
        double[] dist = {-1};
        fr.predict(img, id, dist);
        if (id[0] == -1) {
            return "";
        }
        double d = ((int) (dist[0] * 100));
        return fr.getLabelInfo(id[0]) + " : " + d / 100;
    }
}

//
// SimpleSample [persons_dir] [path/to/face_cascade]
//
class FaceRecog {
    
    static Size size ;
    
    public static void main(String[] args) {
        // PLEASE ADJUST TO YOUR PERSONAL SETTINGS !
        System.load("/home/rabie/Downloads/opencv-master/build/lib/libopencv_java310.so");
        
        String personsDir = "/home/rabie/Downloads/FaceRecog/src/data/persons/";
        if (args.length > 1) personsDir = args[1];
        
        
       
        String cascadeFile = "/home/rabie/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml";
        if (args.length > 2) cascadeFile = args[2];

        
        CascadeClassifier cascade = new CascadeClassifier(cascadeFile);
        System.out.println("cascade loaded: "+(!cascade.empty())+" !");

        
        FaceRec face = new FaceRec();
        
        Size trainSize = face.loadTrainDir(personsDir);
        System.out.println("facerec trained: " + (trainSize != null) + " !");

        
        NamedWindow frame = new NamedWindow("Face");

        
        VideoCapture cap = new VideoCapture();
        cap.open(0);
        if (cap.isOpened()) {
           // System.out.println("Sorry, we could not open you capture !");
       
        while (true) {
            Mat im = new Mat();
            cap.read(im);
            Mat gray = new Mat();
            Imgproc.cvtColor(im, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(gray, gray);
            
            
                MatOfRect faces = new MatOfRect();
                cascade.detectMultiScale(gray, faces);
                Rect[] facesArray = faces.toArray();
               for (Rect rect : faces.toArray()) {
                
                    //Rect found = facesArray[0];
                    Imgproc.rectangle(im, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 100, 0), 2);
                    size = gray.size();
                    Mat fi = gray.submat(rect);
                    if (fi.size() != trainSize) // not needed for lbph, but for eigen and fisher
                    {
                        Imgproc.resize(fi, fi, trainSize);
                    }
                    if(trainSize != null){
                    String s = face.predict(fi);
                    if (s != "") {
                        Imgproc.putText(im, s, new Point(40, 40), Core.FONT_HERSHEY_PLAIN, 1.3, new Scalar(0, 0, 200), 2);
                    }
                    frame.imshow(im);
                }}
                
            
            
            int k = frame.waitKey(1);
            if (k == 27) // 'esc'
            {
                break;
            }
            Mat dis =new Mat();
            if (k == 's') {
                Imgproc.resize(im, dis, size);
                int i = 0;
                Imgcodecs.imwrite("frame"+i+".png", im);
                i++;
            }
        
        }
        
        System.exit(0); // to break out of the ant shell.
    }}
}