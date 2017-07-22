#include <opencv2/opencv.hpp>
#include<iostream>
#include "lunkuo.cpp"

using namespace cv;
using namespace std;

int doubleCamera()
{
	char leftstr[] = "D:\\pic\\P\\3\\LEFT\\1.jpg";  //32
	char rightstr[] = "D:\\pic\\P\\3\\RIGHT\\1.jpg";  //33
   //initialize and allocate memory to load the video stream from camera
   VideoCapture camera0(1);
    camera0.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    camera0.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    VideoCapture camera1(0);
    camera1.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    camera1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
 
    if( !camera0.isOpened() ) return 1;
    if( !camera1.isOpened() ) return 1;

    while(true) {
        //grab and retrieve each frames of the video sequentially
        Mat3b frame0;
        camera0 >> frame0;
        Mat3b frame1;
        camera1 >> frame1;
 
        imshow("Video0", frame0);
        imshow("Video1", frame1);
//      std::cout << frame1.rows() << std::endl;
        //wait for 40 milliseconds
        int c = waitKey(40);
 
     //keep the frame if user press "space" key (ASCII value 0f "space" is 32)
		if(32 == char(c)){
			cout<<leftstr<<endl<<rightstr<<endl;
			cv::imwrite(leftstr,frame0);
			cv::imwrite(rightstr,frame1);
			leftstr[16]++;
			rightstr[17]++;
			continue;
		}
	//exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
        if(27 == char(c)) break;
    }
 
    return 0;
}
 
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)  
{  
    // accept only char type matrices  
    CV_Assert(I.depth() != sizeof(uchar));  
    int channels = I.channels();  
    int nRows = I.rows ;  
    int nCols = I.cols* channels;  
    if (I.isContinuous())   //ÊÇ·ñÁ¬Ðø
    {  
        nCols *= nRows;  
        nRows = 1;  
    }  
    int i,j;  
    uchar* p;  
    for( i = 0; i < nRows; ++i)  
    {  
        p = I.ptr<uchar>(i);  
        for ( j = 0; j < nCols; ++j)  
        {  
            p[j] = table[p[j]];  
        }  
    }  
    return I;  
}  

Mat lunkuo123(Mat frame){
	Mat mat;
	int iLowH = 0;
	int iHighH = 10;//70

	int iLowS = 0; 
	int iHighS = 70;//70

	int iLowV = 200;
	int iHighV = 255;
		 Mat imgHSV;
		 vector<Mat> hsvSplit;
		cvtColor(frame, imgHSV, COLOR_BGR2HSV); 

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mat); 
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		//morphologyEx(mat, mat, MORPH_OPEN, element);
		morphologyEx(mat, mat, MORPH_CLOSE, element);
		morphologyEx(mat, mat, MORPH_OPEN, element);
		imshow("Thresholded Image", mat); 
		//cvSave("mat.xml",&mat);
		//cout<<mat<<endl;
 		waitKey(0);
		return mat;
}


void hashClose(Mat Src){

	cv::Mat matSrc, matSrc1 ,matDst1;
	cv::resize(Src, matSrc1, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);
	//cv::resize(matSrc1, matDst1, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
	cv::resize(Src, matDst1, cv::Size(16, 16), 0, 0, cv::INTER_CUBIC);
	cout<<matDst1<<endl;

	//cv::cvtColor(matDst1, matDst1, CV_BGR2GRAY);
	//cout<<"matDst1 is "<<endl<<matDst1<<endl;

}
//
//int main(){
//	Mat scr = imread("D:\\pic\\4.png");
//	Mat SCR = lunkuo123(scr);
//	hashClose(SCR);
//	imshow("lunkuo",SCR);
//	waitKey(0);
//}
//
//int main(){
//	doubleCamera();
//}
