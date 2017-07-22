#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <highgui.h>  
#include <cv.h>
#include <vector>  
#include <opencv\cxcore.hpp>  
#include <iostream>
#include <opencv.hpp>
//#include "lunkuo.cpp" 
  
using namespace cv;  
using namespace std;  
  

 /*
Orange  0-22
Yellow 22- 38
Green 38-75
Blue 75-130
Violet 130-160
Red 160-179
 */

/*	从特征点KeyPoint形式，返回二维点Point2d形式
/*		输入：vector<KeyPoint> kpts
/*		输出：vector<Point2d> &pts
*/
void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2d> &pts) {  
    for (int i = 0; i < kpts.size(); i++) {  
        pts.push_back(kpts[i].pt);  
    }  
}

/*	对传入的图片Src计算哈希矩阵matDst1
/*		输入：Mat Src 
/*		返回：matDst1
*/
Mat hashClose(Mat Src){

	cv::Mat matSrc, matSrc1 ,matDst1;
//	图片Src压缩成固定比例，利于后面矩阵的对应
	cv::resize(Src, matSrc1, cv::Size(800, 600), 0, 0, cv::INTER_NEAREST);
	cv::resize(Src, matDst1, cv::Size(64,36), 0, 0, cv::INTER_CUBIC);					//64*36矩阵
	//cout<<matDst1<<endl;
	return matDst1;
}
/*	
/*	输入图片根据HSV值获得轮廓图
/*		输入：Mat frame
/*		输出：Mat &mat
/*
*/
void getOutline(Mat frame , Mat &mat){

//	HSV设置，最小最大范围
//	H值代表颜色，色调
	int iLowH =100;
	int iHighH = 120;//70

//	S值代表饱和度
	int iLowS = 100; 
	int iHighS = 200;//70

//	V值代表亮度，明度
	int iLowV = 100;
	int iHighV = 255;

	Mat imgHSV;
	 vector<Mat> hsvSplit;
//	cvtColor将原frame从RGB空间转换到HSV空间
	cvtColor(frame, imgHSV, COLOR_BGR2HSV); 
//	inRange函数将范围不在 lowHSV - highHSV范围内的颜色去除
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mat); 
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//	图像的开闭操作去除部分细小噪点
	//morphologyEx(mat, mat, MORPH_OPEN, element);
	morphologyEx(mat, mat, MORPH_CLOSE, element);
	morphologyEx(mat, mat, MORPH_OPEN, element);
	imshow("Thresholded Image", mat); 

}

/*
/*	getPointBorder通过传进来的图像的哈希值<T>矩阵，获得特征点的取点范围highRow - lowRow,highCol - lowCol;
/*		输入：Mat T
/*		输出：highRow - lowRow ， highCol - lowCol
/*
*/
void getPointBorder(Mat T,int &highRow, int &lowRow,int &highCol, int &lowCol){										
		bool flag = true;
		int lastRow,lastCol;
		int nRows = T.rows;
		int nCols = T.cols;
		uchar *p,*next;

		for(int i =0;i<nRows;i++){
			p = T.ptr<uchar>(i);
			for (int j = 0; j < nCols; j++)
			{
				if(p[j]){
					if(flag){
						highCol = lowCol = i;
						highRow = lowRow = j;
						flag =false;
					}
					//初次检查LowRow
					while( p[++lowRow] ){}
					next = T.ptr<uchar>(lowCol+1);
					while( next[j] ){
						int C = 0;
						//探测下一轮范围内有无
						for(int y=highRow;y<lowRow+1;y++){
							if(next[y]){C++;}}
						if(!C){break;}							//如果下一轮没有值，直接break
						//探测下一轮的highRow的前方,如果有，则highRow前移--；
						C = 1;
						while(j-C>0){
							if(next[j-C]){highRow--;C++;}
							else {break;}
						}
						//探测下一轮的lowRow的后方，如果有，则low后移++
						while( next[lowRow+1] ){lowRow++;}
						next = T.ptr<uchar>(++lowCol);
						j = (lowRow+highRow)/2;
					}
					j = lowRow;
					//查找完成
					if( lowCol-highCol>=2 && lowRow-highRow>=2){i = nRows;break;}
						else{flag = true;}
				}
			}
		}
		//64 36 等比例放大到 1920 1080
		highCol = (highCol-1)*30;
		lowCol = (lowCol+1)*30;
		highRow = (highRow-1)*30;
		lowRow = (lowRow+1)*30;
}

/*
/*	取个平均。。。
/*		输入：vector<Point2d> Points
/*		返回：Point2d
/*
*/
Point2d PointAverage(vector<Point2d> Points) {											
	int X = 0;
	int Y = 0;
	int SIZE = Points.size();
	for (int i = 0; i < SIZE; i++)
	{
		X += Points[i].x;
		Y += Points[i].y;
	}
	return  Point2d(X/SIZE,Y/SIZE);
}

/*
/*	取个中心点，按取圆心的方式算
/*		输入：Point2d pt1、pt2、pt3
/*		返回：Point2d
*/
Point2d tcircle(Point2d pt1, Point2d pt2, Point2d pt3)
    {
        double x1 = pt1.x, x2 = pt2.x, x3 = pt3.x;
        double y1 = pt1.y, y2 = pt2.y, y3 = pt3.y;
        double a = x1 - x2;
        double b = y1 - y2;
        double c = x1 - x3;
        double d = y1 - y3;
        double e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0;
        double f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0;
        double det = b * c - a * d;
        if( fabs(det) < 1e-5)
        {
			return NULL;
        }
        double x0 = -(d * e - b * f) / det;
        double y0 = -(a * f - c * e) / det;
		return Point2d(x0, y0);
    }

/*
/*	获得中心点，输入图像获得图像的中心Point2d点；
/*		输入：frame
/*		输出：vector<Point2d>&pts		返回一个中心点集，在这个程序里最后只取了一个
/*
/*/
void getCenterPiont(vector<Point2d>&pts,Mat3b frame){
	
		vector<KeyPoint> keypoint;//检测img1中的SIFT特征点，存储到keypoints中
		FastFeatureDetector fast(100);	// 检测的阈值为50 

		Mat mat;
		//获得轮廓
		getOutline(frame,mat);

		fast.detect(mat, keypoint);

		int KeypiontSize = keypoint.size();
		//获得哈希值矩阵
		Mat T =hashClose(mat);
		//特征点取点范围，白色区域
		int highRow=0,lowRow=0,highCol=0,lowCol=0; 
		//获得轮廓关于像素1920*1080的取点范围，	highRow - lowRow
		//										highCol - lowCol
		getPointBorder(T,highRow,lowRow,highCol,lowCol);
		//去除取点范围点之外的点，符合条件的点加入vectorKP里
		vector<KeyPoint> KP;
		for (int i = 0; i < KeypiontSize; i++){
			if( keypoint.at(i).pt.x>highRow && keypoint.at(i).pt.x<lowRow && keypoint.at(i).pt.y>highCol && keypoint.at(i).pt.y<lowCol)
			{  KP.push_back(keypoint.at(i));  }
		}

		//drawKeypoints(mat,KP,mat);
		//imshow("after",mat);
	
	vector<Point2d>CenterPoint1;

	int size = keypoint.size();
	if(size<3){
		cout<<"keypoint < 5, cannot calculate!!"<<endl;									//点小于5个数量太少，不通过
		return ;
	}

	//从KeyPoint类型转换成Point2d
	vector<Point2d> toPoint2d;
	KeyPointsToPoints(keypoint,toPoint2d);

	//随机取三点取中心点，调用tcircle函数
	vector<Point2d> TT;
	int count = 0;
	while(count<size/3){
		Point2d Point1 = toPoint2d[rand()%size];
		Point2d Point2 = toPoint2d[rand()%size];
		Point2d Point3 = toPoint2d[rand()%size];
		Point2d P = tcircle(Point1,Point2,Point3);
		if(P.x!=0&&P.y!=0 ){
			if(P.x>highRow && P.x<lowRow && P.y>highCol && P.y<lowCol){
				TT.push_back(P);}
			}
		count++;
	}
	if(TT.size()==0){
		//TT可能取不到点，size为空时直接返回，并且输出提醒
		cout<<"keypoint size is "<<size<<endl;
		cout<<"TT is empty!"<<endl;
		return ;}
	//取平均
	Point2d P = PointAverage(TT);
	pts.push_back(P);

}

/*
/*	三角测量函数
/*	输入：u 、u1		两个三维点，只要是二维点，第三个数值是	-1
/*	输入：P 、p1		两个摄像头的R、T信息，其中一个矩阵是单位矩阵
/*/
Mat_<double> LinearLSTriangulation(Point3d u,Matx34d P,Point3d u1,Matx34d P1){

	Matx43d A(u.x*P(2,0)-P(0,0),u.x*P(2,1)-P(0,1),u.x*P(2,2)-P(0,2),
			  u.x*P(2,0)-P(1,0),u.x*P(2,1)-P(1,1),u.x*P(2,2)-P(1,2),
			  u.x*P1(2,0)-P1(1,0),u.x*P1(2,1)-P1(0,1),u.x*P1(2,2)-P1(0,2),
			  u.x*P1(2,0)-P1(1,0),u.x*P1(2,1)-P1(1,1),u.x*P(12,2)-P1(1,2));

	Matx41d B(-(u.x*P(2,3)-P(0,3)),
			  -(u.x*P(2,3)-P(1,3))
			  -(u.x*P1(2,3)-P1(0,3))
			  -(u.x*P1(2,3)-P1(1,3)));

	Mat_<double>X;
	//通过SVD，计算三维点
	solve(A,B,X,DECOMP_SVD);
	return X;
}
/*
/*	计算三维点函数，主要是一些数据的处理，处理完数据之后调用LinearLSTriangulation函数计算
/*	输入数据依次是：第一个数据二维点 、 第二个数据二维点 
/*					第一个相机内参逆矩阵、第二个相机内参逆矩阵
/*					R、T信息矩阵P、P1
/*	返回：vector<Mat>Point3D		矩阵形式的三维点，三个数据
/*/
vector<Mat> TriangulatePoints(vector<Point2d>& pt_set1, vector<Point2d>& pt_set2, Mat& Kinv1, Mat& Kinv2, Matx34d& P, Matx34d& P1)
{
	//求出来两个点集的点不是对应的，所以两个点集的数量不统一，取少的那个
	int pt_size = pt_set1.size() < pt_set2.size() ? pt_set1.size() :pt_set2.size();

	vector<Mat>Point3D;
	for (int i=0; i<pt_size; i++) {
		Point2d kp = pt_set1[i]; 
		Point3d u(kp.x,kp.y,1.0);
		Matx33d U(kp.x,kp.y,1.0,
				0.0,0.0,0.0,
				0.0,0.0,0.0);

		Mat um = (Mat)U.mul(Kinv1);
		u.x = um.at<double>(0,0); u.y = um.at<double>(0,1); u.z = um.at<double>(0,2);

		Point2d kp1 = pt_set2[i]; 
		Point3d u1(kp1.x,kp1.y,1.0);
		Matx33d U1(kp1.x,kp1.y,1.0,
				0.0,0.0,0.0,
				0.0,0.0,0.0);
		Mat um1 = (Mat)U1.mul(Kinv1);
		u1.x = um1.at<double>(0,0); u1.y = um1.at<double>(0,1); u1.z = um1.at<double>(0,2);
		
		Mat_<double> X = LinearLSTriangulation(u,P,u1,P1);
		Point3D.push_back(X);
	}
	return Point3D;
}

/*
/*	主函数
/*/
int main_Pointdetect()
{
	//	RT信息
	 FileStorage fs("D:\\Documents\\Visual Studio 2012\\Projects\\xiaoqiu_move\\xiaoqiu_move\\extrinsics.xml", FileStorage::READ);  
     Mat_<double> R,T;  
     fs["R"] >> R;  
	 fs["T"] >> T;
	 //	相机内参
	 FileStorage fs2("D:\\Documents\\Visual Studio 2012\\Projects\\xiaoqiu_move\\LEFT_INTRINSIC_MATRIX.xml",FileStorage::READ);
	 Mat LeftK ;
	 fs2["INTRINSIC_MATRIX"] >>LeftK;
	 //	内参逆
	 Mat LeftKinv = LeftK.inv();
	 cout<<LeftKinv<<endl;
	 FileStorage fs3("D:\\Documents\\Visual Studio 2012\\Projects\\xiaoqiu_move\\RIGHT_INTRINSIC_MATRIX.xml",FileStorage::READ);
	 Mat RightK;
	 fs3["INTRINSIC_MATRIX"] >>RightK;
	 Mat RightKinv = RightK.inv();
	 cout<<RightKinv<<endl;
	 

	 Matx34d P(1,0,0,0,
			   0,1,0,0,
			   0,0,1,0);

	 Matx34d P1(R(0,0),R(0,1),R(0,2),T(0),
				R(1,0),R(1,1),R(1,2),T(1),
				R(2,0),R(2,1),R(2,2),T(2));

	VideoCapture camera0(1);
    camera0.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    camera0.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    VideoCapture camera1(0);
    camera1.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    camera1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
 
    if( !camera0.isOpened() ) return 1;
    if( !camera1.isOpened() ) return 1;
	
while(true) 
{
		time_t clockBegin, clockEnd;    
    	clockBegin = clock();        
		   
		//	从摄像头读图像
        Mat3b Left_frame;
        camera0 >> Left_frame;

        Mat3b Right_frame;
        camera1 >> Right_frame;

		vector<Point2d>CenterPoint1;
		vector<Point2d>CenterPoint2;

		getCenterPiont(CenterPoint1,Left_frame);
		getCenterPiont(CenterPoint2,Right_frame);

		//如果没有取到点，跳过
		if(CenterPoint1.empty()){cout<<"NO CenterPoint1 "<<endl;continue;}
		if(CenterPoint2.empty()){cout<<"NO CenterPoint2 "<<endl;continue;}

		vector<KeyPoint> M1;
		vector<KeyPoint> M2;
		for (int i = 0; i < CenterPoint1.size(); i++)
		{
			KeyPoint k =KeyPoint(CenterPoint1[i],-1);
			KeyPoint k2 =KeyPoint(CenterPoint2[i],-1);
			M1.push_back(k);
			M2.push_back(k2);
		}


		drawKeypoints(Right_frame,M2,Right_frame);
		drawKeypoints(Left_frame,M1,Left_frame);
		imshow("centerPointLeft",Left_frame);
		imshow("centerPointRight",Right_frame);
		//waitKey(0);

		//计算三维点
		vector<Mat> Point3D;
		Point3D =  TriangulatePoints(CenterPoint1,CenterPoint2,LeftKinv,RightKinv,P,P1);
		cout<<Point3D.size()<<endl;
		for (int i = 0; i < Point3D.size(); i++)
		{
			//输出结果
			cout<<"*****************"<< i <<"****************"<<endl;
			cout<<Point3D[i]<<endl;			
		}
    
		char c=cvWaitKey(33);
		if(c==27) {waitKey();break;}

		clockEnd = clock();   
		cout<<"-------------- FPS :_____"<< (double)6000/(clockEnd - clockBegin)<<endl;    
		}
    return 0;
}

int main(){
	main_Pointdetect();
	return 0;
}