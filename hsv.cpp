#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>

//using namespace std;
//
//
//int max(int R,int G,int B){
//	int Max = R;
//	if(G>R){Max = G;}
//	if(B>Max){Max = B;}
//	return Max;
//}
//	
//int min(int R,int G,int B){
//	int Min = R;
//	if(G<R){Min = G;}
//	if(B<Min){Min = B;}
//	return Min;
//}
//
//int main(){
//	int R = 87;
//	int G = 83;
//	int B = 108;
//	int MAX = max(R,G,B);
//	int MIN = min(R,G,B);
//
//	cout<<MAX<<"++++"<<MIN<<endl;
//	int H,S,V;
//	if(MAX == R){ H = (G-B)/(MAX-MIN);}
//	if(MAX == G){ H = 2+ (B-R)/(MAX-MIN);}
//	if(MAX == B){ H = 4+ (R-G)/(MAX-MIN);}
//
//	H = H*60;
//	if(H<0){H=H+360;}
//
//	V = max(R,G,B);
//	S = (MAX-MIN)/MAX;
//
//	cout<<"H:"<<H<<endl;
//	cout<<"S:"<<S<<endl;
//	cout<<"V:"<<V<<endl;
//
//	return 0;
//}


#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "iostream"
using namespace std;

IplImage* pImg,*imgRGB,*imgHSV;
int flags = 0;
CvPoint pt;
CvScalar s = {0.0},ss={0.0};

void on_mouse( int event, int x, int y, int flags, void* param )
{

	if( !imgRGB )
		return;

	switch(event)
	{
	case CV_EVENT_LBUTTONDOWN: 
		{
			s=cvGet2D(imgRGB,y,x);
			//cout<<"("<<x<<","<<y<<")"<<"="<< s.val[0]<< " ";
			printf("(%d,%d)处的RGB值分别是：B = %f,G = %f, R = %f \n",x,y,s.val[0],s.val[1],s.val[2]);
			ss = cvGet2D(imgHSV,y,x);
			printf("(%d,%d)处的RGB值分别是：H = %f,S = %f, V = %f \n\n",x,y,ss.val[0],ss.val[1],ss.val[2]);
		}
		break;
	}
}
int main_hsv()
{
//"F:\\00 论文\\06 双目视觉\\我的Project\\敬老院图片\\1_强_噪声围绕目标.jpg"
	//imgRGB = cvLoadImage( "D:\\pic\\distansPic\\f\\LEFT\\4\\1.jpg", 1);
	//imgRGB = cvLoadImage( "D:\\pic\\doubleCamera\\2\\LEFT\\1.jpg", 1);
	imgRGB = cvLoadImage( "D:\\pic\\t\\4.jpg", 1);
	imgHSV = cvCreateImage(cvGetSize(imgRGB),8,3);
	cvNamedWindow( "imgRGB", 2);
	cvSetMouseCallback( "imgRGB", on_mouse, 0 );
	cvShowImage( "imgRGB", imgRGB ); //显示图像
	cvCvtColor(imgRGB,imgHSV,CV_RGB2HSV);


	cvWaitKey(); //等待按键
	cvDestroyWindow( "imgRGB" );//销毁窗口
	cvReleaseImage( &imgRGB ); //释放图像
	return 0;
}
//int main(){
//	main_hsv();
//	return 0;
//}
