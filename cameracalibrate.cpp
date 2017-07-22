
// opencv_test.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include <cxcore.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cxcore.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv/cxcore.h>
#include <io.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <cv.hpp>
#include <iostream>

using namespace std;
using namespace cv;



const string fileaddress = "C:\\Users\\Administrator\\Desktop\\pro\\calibrateCamera\\leftpic\\";
const int boardWidth = 7;								//����Ľǵ���Ŀ
const int boardHeight = 7;								//����Ľǵ�����
const int boardCorner = boardWidth * boardHeight;		//�ܵĽǵ�����
const int frameNumber = 12;								//����궨ʱ��Ҫ���õ�ͼ��֡��
const int squareSize = 20;								//�궨��ڰ׸��ӵĴ�С ��λmm
const Size boardSize = Size(boardWidth, boardHeight);	//
	
Mat intrinsic;											//����ڲ���
Mat distortion_coeff;									//����������
vector<Mat> rvecs;									    //��ת����
vector<Mat> tvecs;										//ƽ������
vector<vector<Point2d>> corners;						//����ͼ���ҵ��Ľǵ�ļ��� ��objRealPoint һһ��Ӧ
vector<vector<Point3f>> objRealPoint;					//����ͼ��Ľǵ��ʵ���������꼯��


vector<Point2d> corner;									//ĳһ��ͼ���ҵ��Ľǵ�



	/*����궨����ģ���ʵ����������*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth,int boardheight, int imgNumber, int squaresize)
{
//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
		//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

   /*��������ĳ�ʼ���� Ҳ���Բ�����*/
void guessCameraParam(void )
{
	/*�����ڴ�*/
	intrinsic.create(3, 3, CV_64FC1);
	distortion_coeff.create(5, 1, CV_64FC1);

	/*
	fx 0 cx
	0 fy cy
	0 0  1
	*/
	intrinsic.at<double>(0,0) = 256.8093262;   //fx		
	intrinsic.at<double>(0, 2) = 160.2826538;   //cx
	intrinsic.at<double>(1, 1) = 254.7511139;   //fy
	intrinsic.at<double>(1, 2) = 127.6264572;   //cy

	intrinsic.at<double>(0, 1) = 0;
	intrinsic.at<double>(1, 0) = 0;
	intrinsic.at<double>(2, 0) = 0;
	intrinsic.at<double>(2, 1) = 0;
	intrinsic.at<double>(2, 2) = 1;

	/*
	k1 k2 p1 p2 p3
	*/
	distortion_coeff.at<double>(0, 0) = -0.193740;  //k1
	distortion_coeff.at<double>(1, 0) = -0.378588;  //k2
	distortion_coeff.at<double>(2, 0) = 0.028980;   //p1
	distortion_coeff.at<double>(3, 0) = 0.008136;   //p2
	distortion_coeff.at<double>(4, 0) = 0;		  //p3
}

void OutputCameraParam(void )
{
	/*��������*/
	//cvSave("cameraMatrix.xml", &intrinsic);
	//cvSave("cameraDistoration.xml", &distortion_coeff);
	//cvSave("rotatoVector.xml", &rvecs);
	//cvSave("translationVector.xml", &tvecs);
	/*�������*/
	//string s = "C:\\Users\\Administrator\\Desktop\\rightpic\\";
	//string s = "C:\\Users\\Administrator\\Desktop\\pro\\pic\\";
	FileStorage fs(fileaddress+"INTRINSIC_MATRIX"+".xml", FileStorage::WRITE);	
	fs  << "INTRINSIC_MATRIX"<<intrinsic<<"distortion_coeff"<<distortion_coeff;  
	cout<<intrinsic<<endl;
	fs.release();
	cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
	cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;

	cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
	cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
	cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
	cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
	cout << "p3 :" << distortion_coeff.at<double>(4, 0) << endl;
}


bool Cameracalibrate()
{
    int imageHeight;
    int imageWidth;
	int goodFrameCount = 0;

	Mat img,rgbImage;
	string str = fileaddress + "pic01.jpg";
    Mat tImage=imread(str);
	imageHeight = tImage.rows;
	imageWidth = tImage.cols;
	Mat grayImage(imageHeight,imageWidth,CV_8U);
	while (goodFrameCount < frameNumber)
	{
		char filename[100];
		if(goodFrameCount<9){
			sprintf_s(filename, "C:\\Users\\Administrator\\Desktop\\pro\\calibrateCamera\\leftpic\\pic0%d.jpg", goodFrameCount + 1);
		}else{
			sprintf_s(filename,"C:\\Users\\Administrator\\Desktop\\pro\\calibrateCamera\\leftpic\\pic%d.jpg", goodFrameCount + 1);
		}
		rgbImage = imread(filename);
		cvtColor(rgbImage, grayImage, CV_BGR2GRAY);													
		//imshow("Camera", grayImage);
		
		//bool isFind = findChessboardCorners(rgbImage, boardSize, corner,0);
		bool isFind = findChessboardCorners( rgbImage, boardSize, corner,
                    CV_CALIB_CB_ADAPTIVE_THRESH + 
					CV_CALIB_CB_FAST_CHECK 
					+ CV_CALIB_CB_NORMALIZE_IMAGE
					+ CV_CALIB_CB_FILTER_QUADS
					);
               
		if (isFind == true)	//���нǵ㶼���ҵ� ˵�����ͼ���ǿ��е�
		{
			cornerSubPix(grayImage, corner, Size(5,5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			drawChessboardCorners(rgbImage, boardSize, corner, isFind);
			imshow("chessboard", rgbImage);
			corners.push_back(corner);
			goodFrameCount++;
			cout << "The image"<<goodFrameCount<<" is good" << endl;
		}
		else
		{
			cout << "The image is bad please try again" << endl;
			return 0;
		}

	   if (waitKey(10) == 'q')
		{
			break;
		}
	
	}

	/*
	ͼ��ɼ���� ��������ʼ����ͷ��У��
	calibrateCamera()
	������� objectPoints  �ǵ��ʵ����������
			 imagePoints   �ǵ��ͼ������
			 imageSize	   ͼ��Ĵ�С
	�������
			 cameraMatrix  ������ڲξ���
			 distCoeffs	   ����Ļ������
			 rvecs		   ��תʸ��(�����)
			 tvecs		   ƽ��ʸ��(�������
	*/
	
	/*����ʵ�ʳ�ʼ���� ����calibrateCamera�� ���flag = 0 Ҳ���Բ���������*/
	guessCameraParam();			
	cout << "guess successful" << endl;
	/*����ʵ�ʵ�У�������ά����*/
	calRealPoint(objRealPoint, boardWidth, boardHeight,frameNumber, squareSize);
	cout << "cal real successful" << endl;
	/*�궨����ͷ*/
	calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, 0);
	cout << "calibration successful" << endl;
	/*���沢�������*/
	OutputCameraParam();
	cout << "out successful" << endl;
	
	/*��ʾ����У��Ч��*/
	Mat cImage;
	undistort(rgbImage, cImage, intrinsic, distortion_coeff);
	imshow("Corret Image", cImage);
	cout << "Correct Image" << endl;
	cout << "Wait for Key" << endl;
	waitKey();
	
	return 1;
}



//int main()
//{
//	Cameracalibrate();
//	waitKey();
//
//    return 0;
//}