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

/*	��������KeyPoint��ʽ�����ض�ά��Point2d��ʽ
/*		���룺vector<KeyPoint> kpts
/*		�����vector<Point2d> &pts
*/
void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2d> &pts) {  
    for (int i = 0; i < kpts.size(); i++) {  
        pts.push_back(kpts[i].pt);  
    }  
}

/*	�Դ����ͼƬSrc�����ϣ����matDst1
/*		���룺Mat Src 
/*		���أ�matDst1
*/
Mat hashClose(Mat Src){

	cv::Mat matSrc, matSrc1 ,matDst1;
//	ͼƬSrcѹ���ɹ̶����������ں������Ķ�Ӧ
	cv::resize(Src, matSrc1, cv::Size(800, 600), 0, 0, cv::INTER_NEAREST);
	cv::resize(Src, matDst1, cv::Size(64,36), 0, 0, cv::INTER_CUBIC);					//64*36����
	//cout<<matDst1<<endl;
	return matDst1;
}
/*	
/*	����ͼƬ����HSVֵ�������ͼ
/*		���룺Mat frame
/*		�����Mat &mat
/*
*/
void getOutline(Mat frame , Mat &mat){

//	HSV���ã���С���Χ
//	Hֵ������ɫ��ɫ��
	int iLowH =100;
	int iHighH = 120;//70

//	Sֵ�����Ͷ�
	int iLowS = 100; 
	int iHighS = 200;//70

//	Vֵ�������ȣ�����
	int iLowV = 100;
	int iHighV = 255;

	Mat imgHSV;
	 vector<Mat> hsvSplit;
//	cvtColor��ԭframe��RGB�ռ�ת����HSV�ռ�
	cvtColor(frame, imgHSV, COLOR_BGR2HSV); 
//	inRange��������Χ���� lowHSV - highHSV��Χ�ڵ���ɫȥ��
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), mat); 
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//	ͼ��Ŀ��ղ���ȥ������ϸС���
	//morphologyEx(mat, mat, MORPH_OPEN, element);
	morphologyEx(mat, mat, MORPH_CLOSE, element);
	morphologyEx(mat, mat, MORPH_OPEN, element);
	imshow("Thresholded Image", mat); 

}

/*
/*	getPointBorderͨ����������ͼ��Ĺ�ϣֵ<T>���󣬻���������ȡ�㷶ΧhighRow - lowRow,highCol - lowCol;
/*		���룺Mat T
/*		�����highRow - lowRow �� highCol - lowCol
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
					//���μ��LowRow
					while( p[++lowRow] ){}
					next = T.ptr<uchar>(lowCol+1);
					while( next[j] ){
						int C = 0;
						//̽����һ�ַ�Χ������
						for(int y=highRow;y<lowRow+1;y++){
							if(next[y]){C++;}}
						if(!C){break;}							//�����һ��û��ֵ��ֱ��break
						//̽����һ�ֵ�highRow��ǰ��,����У���highRowǰ��--��
						C = 1;
						while(j-C>0){
							if(next[j-C]){highRow--;C++;}
							else {break;}
						}
						//̽����һ�ֵ�lowRow�ĺ󷽣�����У���low����++
						while( next[lowRow+1] ){lowRow++;}
						next = T.ptr<uchar>(++lowCol);
						j = (lowRow+highRow)/2;
					}
					j = lowRow;
					//�������
					if( lowCol-highCol>=2 && lowRow-highRow>=2){i = nRows;break;}
						else{flag = true;}
				}
			}
		}
		//64 36 �ȱ����Ŵ� 1920 1080
		highCol = (highCol-1)*30;
		lowCol = (lowCol+1)*30;
		highRow = (highRow-1)*30;
		lowRow = (lowRow+1)*30;
}

/*
/*	ȡ��ƽ��������
/*		���룺vector<Point2d> Points
/*		���أ�Point2d
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
/*	ȡ�����ĵ㣬��ȡԲ�ĵķ�ʽ��
/*		���룺Point2d pt1��pt2��pt3
/*		���أ�Point2d
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
/*	������ĵ㣬����ͼ����ͼ�������Point2d�㣻
/*		���룺frame
/*		�����vector<Point2d>&pts		����һ�����ĵ㼯����������������ֻȡ��һ��
/*
/*/
void getCenterPiont(vector<Point2d>&pts,Mat3b frame){
	
		vector<KeyPoint> keypoint;//���img1�е�SIFT�����㣬�洢��keypoints��
		FastFeatureDetector fast(100);	// ������ֵΪ50 

		Mat mat;
		//�������
		getOutline(frame,mat);

		fast.detect(mat, keypoint);

		int KeypiontSize = keypoint.size();
		//��ù�ϣֵ����
		Mat T =hashClose(mat);
		//������ȡ�㷶Χ����ɫ����
		int highRow=0,lowRow=0,highCol=0,lowCol=0; 
		//���������������1920*1080��ȡ�㷶Χ��	highRow - lowRow
		//										highCol - lowCol
		getPointBorder(T,highRow,lowRow,highCol,lowCol);
		//ȥ��ȡ�㷶Χ��֮��ĵ㣬���������ĵ����vectorKP��
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
		cout<<"keypoint < 5, cannot calculate!!"<<endl;									//��С��5������̫�٣���ͨ��
		return ;
	}

	//��KeyPoint����ת����Point2d
	vector<Point2d> toPoint2d;
	KeyPointsToPoints(keypoint,toPoint2d);

	//���ȡ����ȡ���ĵ㣬����tcircle����
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
		//TT����ȡ�����㣬sizeΪ��ʱֱ�ӷ��أ������������
		cout<<"keypoint size is "<<size<<endl;
		cout<<"TT is empty!"<<endl;
		return ;}
	//ȡƽ��
	Point2d P = PointAverage(TT);
	pts.push_back(P);

}

/*
/*	���ǲ�������
/*	���룺u ��u1		������ά�㣬ֻҪ�Ƕ�ά�㣬��������ֵ��	-1
/*	���룺P ��p1		��������ͷ��R��T��Ϣ������һ�������ǵ�λ����
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
	//ͨ��SVD��������ά��
	solve(A,B,X,DECOMP_SVD);
	return X;
}
/*
/*	������ά�㺯������Ҫ��һЩ���ݵĴ�������������֮�����LinearLSTriangulation��������
/*	�������������ǣ���һ�����ݶ�ά�� �� �ڶ������ݶ�ά�� 
/*					��һ������ڲ�����󡢵ڶ�������ڲ������
/*					R��T��Ϣ����P��P1
/*	���أ�vector<Mat>Point3D		������ʽ����ά�㣬��������
/*/
vector<Mat> TriangulatePoints(vector<Point2d>& pt_set1, vector<Point2d>& pt_set2, Mat& Kinv1, Mat& Kinv2, Matx34d& P, Matx34d& P1)
{
	//����������㼯�ĵ㲻�Ƕ�Ӧ�ģ����������㼯��������ͳһ��ȡ�ٵ��Ǹ�
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
/*	������
/*/
int main_Pointdetect()
{
	//	RT��Ϣ
	 FileStorage fs("D:\\Documents\\Visual Studio 2012\\Projects\\xiaoqiu_move\\xiaoqiu_move\\extrinsics.xml", FileStorage::READ);  
     Mat_<double> R,T;  
     fs["R"] >> R;  
	 fs["T"] >> T;
	 //	����ڲ�
	 FileStorage fs2("D:\\Documents\\Visual Studio 2012\\Projects\\xiaoqiu_move\\LEFT_INTRINSIC_MATRIX.xml",FileStorage::READ);
	 Mat LeftK ;
	 fs2["INTRINSIC_MATRIX"] >>LeftK;
	 //	�ڲ���
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
		   
		//	������ͷ��ͼ��
        Mat3b Left_frame;
        camera0 >> Left_frame;

        Mat3b Right_frame;
        camera1 >> Right_frame;

		vector<Point2d>CenterPoint1;
		vector<Point2d>CenterPoint2;

		getCenterPiont(CenterPoint1,Left_frame);
		getCenterPiont(CenterPoint2,Right_frame);

		//���û��ȡ���㣬����
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

		//������ά��
		vector<Mat> Point3D;
		Point3D =  TriangulatePoints(CenterPoint1,CenterPoint2,LeftKinv,RightKinv,P,P1);
		cout<<Point3D.size()<<endl;
		for (int i = 0; i < Point3D.size(); i++)
		{
			//������
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