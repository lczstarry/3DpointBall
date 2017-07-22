//
//// opencv_test.cpp : �������̨Ӧ�ó������ڵ㡣
////
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//#include <vector>
//#include <string>
//#include <algorithm>
//#include <iostream>
//#include <iterator>
//#include <stdio.h>
//#include <stdlib.h>
//#include <ctype.h>
//
////#include "stdafx.h"
//#include <opencv2/opencv.hpp>
////#include <opencv2/highgui.hpp>
//#include "cv.h"
//#include <cv.hpp>
//
//using namespace std;
//using namespace cv;
//
//const int imageWidth = 1080;								//����ͷ�ķֱ���
//const int imageHeight = 1902;
//const int boardWidth = 7;								//����Ľǵ���Ŀ
//const int boardHeight = 7;								//����Ľǵ�����
//const int boardCorner = boardWidth * boardHeight;		//�ܵĽǵ�����
//const int frameNumber = 14;								//����궨ʱ��Ҫ���õ�ͼ��֡��
//const int squareSize = 20;								//�궨��ڰ׸��ӵĴ�С ��λmm
//const Size boardSize = Size(boardWidth, boardHeight);	//
//Size imageSize = Size(imageWidth, imageHeight);
//
//Mat R, T, E, F;											//R ��תʸ�� Tƽ��ʸ�� E�������� F��������
//vector<Mat> rvecs;									    //��ת����
//vector<Mat> tvecs;										//ƽ������
//vector<vector<Point2d>> imagePointL;				    //��������������Ƭ�ǵ�����꼯��
//vector<vector<Point2d>> imagePointR;					//�ұ������������Ƭ�ǵ�����꼯��
//vector<vector<Point3f>> objRealPoint;					//����ͼ��Ľǵ��ʵ���������꼯��
//
//
//vector<Point2d> cornerL;								//��������ĳһ��Ƭ�ǵ����꼯��
//vector<Point2d> cornerR;								//�ұ������ĳһ��Ƭ�ǵ����꼯��
//
//Mat rgbImageL, grayImageL;
//Mat rgbImageR, grayImageR;
//
//Mat Rl, Rr, Pl, Pr, Q;									//У����ת����R��ͶӰ����P ��ͶӰ����Q (�����о���ĺ�����ͣ�	
//Mat mapLx, mapLy, mapRx, mapRy;							//ӳ���
//Rect validROIL, validROIR;								//ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������
//
////
////���ȱ궨�õ���������ڲξ���
////fx 0 cx
////0 fy cy
////0 0  1
////
////1.84005261e+003 0. 9.44989868e+002 0. 1.83817017e+003
////    5.89992737e+002 0. 0. 1
//const Mat cameraMatrixL = (Mat_<double>(3, 3) << 1858.0973, 0,       936.9929,
//										  0,       1856.554, 596.4914,
//										  0,       0,       1);
////-1.73999190e-001 9.80499759e-003 -6.29667949e-004 1.45331502e-003
//const Mat distCoeffL = (Mat_<double>(5, 1) << -0.15939, 0.2657, -0.00038347, 0.0011077,1.1874580884533490);
////
////���ȱ궨�õ���������ڲξ���
////fx 0 cx
////0 fy cy
////0 0  1
////
//const Mat cameraMatrixR = (Mat_<double>(3, 3) << 1840.820056206, 0,       889.53751974168119,
//											0,      1839.1639018685730, 467.51654701822054,
//											0,		0,		 1);
//const Mat distCoeffR = (Mat_<double>(5, 1) << -0.19246417816470, -0.0858141464591580, -0.0015809428209646, -0.0037831001410293,-0.0127830046820581);
//
//
///*����궨����ģ���ʵ����������*/
//void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
//{
//	//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
//	vector<Point3f> imgpoint;
//	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
//	{
//		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
//		{
//			//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);
//			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
//		}
//	}
//	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
//	{
//		obj.push_back(imgpoint);
//	}
//}
//
//void outputCameraParam(void)
//{
//	/*��������*/
//	/*�������*/
//	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
//	if (fs.isOpened())
//	{
//		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL <<"cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
//		fs.release();
//		cout << "cameraMatrixL=:" << cameraMatrixL <<endl<< "cameraDistcoeffL=:" << distCoeffL <<endl<<"cameraMatrixR=:" << cameraMatrixR <<endl<< "cameraDistcoeffR=:" << distCoeffR<<endl;
//	}
//	else
//	{
//		cout << "Error: can not save the intrinsics!!!!!" << endl;
//	}
//
//	fs.open("extrinsics.xml", FileStorage::WRITE);
//	if (fs.isOpened())
//	{
//		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
//		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
//		fs.release();
//	}
//	else
//		cout << "Error: can not save the extrinsic parameters\n";
//}
//
//
//int camera_main(int argc)
//{
//	Mat img;
//	int goodFrameCount = 0;
//	while (goodFrameCount < frameNumber)
//	{
//		char filename[100];
//		/*��ȡ��ߵ�ͼ��*/
//		if(goodFrameCount < 9){
//			sprintf_s(filename, "D:\\pic\\leftpic\\pic0%d.jpg", goodFrameCount + 1);
//			rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
//			cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
//	
//			/*��ȡ�ұߵ�ͼ��*/
//			sprintf_s(filename, "D:\\pic\\rightpic\\pic0%d.jpg", goodFrameCount + 1);
//			rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
//			cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
//		}else
//		{
//			sprintf_s(filename, "D:\\pic\\leftpic\\pic%d.jpg", goodFrameCount + 1);
//			rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
//			cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
//	
//			/*��ȡ�ұߵ�ͼ��*/
//			sprintf_s(filename, "D:\\pic\\rightpic\\pic%d.jpg", goodFrameCount + 1);
//			rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
//			cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
//		}
//		bool isFindL, isFindR;
//
//		isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
//		isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
//		if (isFindL == true && isFindR == true)	 //�������ͼ���ҵ������еĽǵ� ��˵��������ͼ���ǿ��е�
//		{
//			/*
//			Size(5,5) �������ڵ�һ���С
//			Size(-1,-1) ������һ��ߴ�
//			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)������ֹ����
//			*/
//			cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
//			drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
//			imshow("chessboardL", rgbImageL);
//			imagePointL.push_back(cornerL);
//
//
//			cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
//			drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
//			imshow("chessboardR", rgbImageR);
//			imagePointR.push_back(cornerR);
//
//			/*
//			����Ӧ���ж�������ͼ���ǲ��Ǻõģ��������ƥ��Ļ��ſ��������궨
//			������������̵��У��õ�ͼ����ϵͳ�Դ���ͼ�񣬶��ǿ���ƥ��ɹ��ġ�
//			���������û���ж�
//			*/
//			//string filename = "res\\image\\calibration";
//			//filename += goodFrameCount + ".jpg";
//			//cvSaveImage(filename.c_str(), &IplImage(rgbImage));		//�Ѻϸ��ͼƬ��������
//			goodFrameCount++;
//			cout << "The image"<<goodFrameCount<<" is good" << endl;
//		}
//		else
//		{
//			cout << "The image is bad please try again" << endl;
//		}
//
//		if (waitKey(10) == 'q')
//		{
//			break;
//		}
//	}
//
//	/*
//	����ʵ�ʵ�У�������ά����
//	����ʵ�ʱ궨���ӵĴ�С������
//	*/
//	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
//	cout << "cal real successful" << endl;
//
//	/*
//		�궨����ͷ
//		��������������ֱ𶼾����˵�Ŀ�궨
//		�����ڴ˴�ѡ��flag = CALIB_USE_INTRINSIC_GUESS
//	*/
//	cout<<"************** 1 *************"<<endl;
//	cout<<cameraMatrixL<<endl<<distCoeffL<<endl;
//	cout<<cameraMatrixR<<endl<<distCoeffR<<endl;
//	double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
//		cameraMatrixL, distCoeffL,
//		cameraMatrixR, distCoeffR,
//		Size(imageWidth, imageHeight), R, T, E, F,
//		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5),
//		CALIB_USE_INTRINSIC_GUESS);
//
//	cout << "Stereo Calibration done with RMS error = " << rms << endl;
//
//	/*
//	����У����ʱ����Ҫ����ͼ���沢���ж�׼ ��ʹ������ƥ����ӵĿɿ�
//	ʹ������ͼ����ķ������ǰ���������ͷ��ͼ��ͶӰ��һ�������������ϣ�����ÿ��ͼ��ӱ�ͼ��ƽ��ͶӰ������ͼ��ƽ�涼��Ҫһ����ת����R
//	stereoRectify �����������ľ��Ǵ�ͼ��ƽ��ͶӰ����������ƽ�����ת����Rl,Rr�� Rl,Rr��Ϊ�������ƽ���ж�׼��У����ת����
//	���������Rl��ת�����������Rr��ת֮������ͼ����Ѿ����沢���ж�׼�ˡ�
//	����Pl,PrΪ���������ͶӰ�����������ǽ�3D�������ת����ͼ���2D�������:P*[X Y Z 1]' =[x y w] 
//	Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ���ʱ��
//	*/
//	cout<<"************** 2 *************"<<endl;
//	cout<<cameraMatrixL<<endl<<distCoeffL<<endl;
//	cout<<cameraMatrixR<<endl<<distCoeffR<<endl;
//	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
//				  CALIB_ZERO_DISPARITY,-1,imageSize,&validROIL,&validROIR);
//	/*
//	����stereoRectify ���������R �� P ������ͼ���ӳ��� mapx,mapy
//	mapx,mapy������ӳ�����������Ը�remap()�������ã���У��ͼ��ʹ������ͼ���沢���ж�׼
//	ininUndistortRectifyMap()�Ĳ���newCameraMatrix����У����������������openCV���棬У����ļ��������Mrect�Ǹ�ͶӰ����Pһ�𷵻صġ�
//	�������������ﴫ��ͶӰ����P���˺������Դ�ͶӰ����P�ж���У��������������
//	*/
//	cout<<"************** 3 *************"<<endl;
//	cout<<cameraMatrixL<<endl<<distCoeffL<<endl;
//	cout<<cameraMatrixR<<endl<<distCoeffR<<endl;
//	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
//	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
//
//
//	Mat rectifyImageL, rectifyImageR;
//	cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
//	cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
//
//	imshow("Rectify Before", rectifyImageL);
//	cout << "��Q1�˳� ..." << endl;
//
//	/*
//	����remap֮�����������ͼ���Ѿ����沢���ж�׼��
//	*/
//	Mat rectifyImageL2,rectifyImageR2;
//	remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
//	remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);
//	cout << "��Q2�˳� ..." << endl;
//
//	imshow("ImageL", rectifyImageL2);
//	imshow("ImageR", rectifyImageR2);
//
//	cout<<"************** 4 *************"<<endl;
//	cout<<cameraMatrixL<<endl<<distCoeffL<<endl;
//	cout<<cameraMatrixR<<endl<<distCoeffR<<endl;
//	/*���沢�������*/
//	outputCameraParam();
//
//	 /*
//		��У�������ʾ����
//		����������ͼ����ʾ��ͬһ��������
//		����ֻ��ʾ�����һ��ͼ���У���������û�а����е�ͼ����ʾ����
//	*/
//	Mat canvas;
//	double sf;
//	int w, h;
//	sf = 600. / MAX(imageSize.width, imageSize.height);
//	w = cvRound(imageSize.width * sf);
//	h = cvRound(imageSize.height * sf);
//	canvas.create(h, w * 2, CV_8UC3);
//
//	/*��ͼ�񻭵�������*/
//	Mat canvasPart = canvas(Rect(w*0, 0, w, h));								//�õ�������һ����
//	resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//��ͼ�����ŵ���canvasPartһ����С
//	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),				//��ñ���ȡ������	
//		      cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
//	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//����һ������
//
//	cout << "Painted ImageL" << endl;
//
//	/*��ͼ�񻭵�������*/
//	canvasPart = canvas(Rect(w, 0, w, h));										//��û�������һ����
//	resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
//	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),			
//			  cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
//	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);
//
//	cout << "Painted ImageR" << endl;
//
//	/*���϶�Ӧ������*/
//	for (int i = 0; i < canvas.rows;i+=16)
//		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
//
//	imshow("rectified", canvas);
//
//	cout << "wait key" << endl;
//	waitKey(0);
//	//system("pause");
//	return 0;
//}
////
////int main(){
////	camera_main(0);
////}
////
