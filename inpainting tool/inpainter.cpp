#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat image,originalImage,inpaintMask;
Point prevPt(-1,-1);
int thickness=5;

static void onMouse( int event, int x, int y, int flags, void* )
{
    if(event == EVENT_LBUTTONUP||!(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x,y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( inpaintMask, prevPt, pt, Scalar::all(255), thickness, 8, 0 );
        line( image, prevPt, pt, Scalar::all(255), thickness, 8, 0 );
        prevPt = pt;
        imshow("image", image);
    }
}

class GradientCalculator
{
    Mat gradX;
    Mat gradY;
public:
    GradientCalculator();
    void calculateGradient(Mat &src);
    Mat getGradX();
    Mat getGradY();
};

class Inpainter
{
public:

    Inpainter(Mat inputImage,Mat mask,int halfPatchWidth=4,int mode=1);

    Mat inputImage, mask, updatedMask, result, workImage, sourceRegion, targetRegion;
    Mat originalSourceRegion, gradientX, gradientY, confidence, data;
    Mat LAPLACIAN_KERNEL,NORMAL_KERNELX,NORMAL_KERNELY;
    Point2i bestMatchUpperLeft,bestMatchLowerRight;
    std::vector<Point> fillFront;
    std::vector<Point2f> normals;
    int mode;
    int halfPatchWidth;
    int targetIndex;

    void calculateGradients();
    void initializeMats();
    void computeFillFront();
    void computeConfidence();
    void computeData();
    void computeTarget();
    void computeBestPatch();
    void updateMats();
    bool checkEnd();
    void getPatch(Point2i &centerPixel, Point2i &upperLeft, Point2i &lowerRight);
    void inpaint();
};

int main()
{
    originalImage=imread("../images/image.jpg",CV_LOAD_IMAGE_COLOR);
    resize(originalImage, originalImage, Size(960, 480));

    cout << "You can specify the area to be inpainted by mouse and press i to initialise inpainting\n";
    cout << "You can increase thickness of cursor by pressing + or decrease by pressing -\n";
    cout << "after inpainting is done the result will be shown to you and saved in images folder by name result.jpg\n";

    image=originalImage.clone();

    inpaintMask = Mat::zeros(image.size(), CV_8U);
    namedWindow( "image", 1 );
    imshow("image", image);
    setMouseCallback( "image", onMouse, 0 );

    while(true)
    {
        char c = waitKey();

        if( c == 'e' )
            break;

        if( c == 'r' )
        {
            inpaintMask = Scalar::all(0);
            image=originalImage.clone();
            imshow("image", image);
        }

        if( c == 'i' || c == ' ' )
        {
            Inpainter i(originalImage,inpaintMask,4);
            i.inpaint();
            imwrite("../images/result.jpg",i.result);
            inpaintMask = Scalar::all(0);
            namedWindow("result");
            imshow("result",i.result);
        }
        if(c=='+'){
            thickness++;
        }
        if(c=='-'){
            thickness--;
        }
        if(thickness<3)
            thickness=3;
        if(thickness>12)
            thickness=12;
    }

    return 0;
}

void GradientCalculator::calculateGradient(Mat &src)
{
    this->gradX=Mat(src.rows,src.cols,CV_32F,Scalar::all(0));
    this->gradY=gradX.clone();

    Vec3b pixel1;
    Vec3b pixel0;
    Vec3f pixelDiff;


    int x,y;
    if(src.rows>1)
    {
        for( x=0 ; x < src.cols ; x++ )
        {
            pixel1=src.at<Vec3b>(1,x);
            pixel0=src.at<Vec3b>(0,x);
            pixelDiff=pixel1-pixel0;
            gradX.at<float>(0,x)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);

            pixel1=src.at<Vec3b>(src.rows-1,x);
            pixel0=src.at<Vec3b>(src.rows-2,x);
            pixelDiff=pixel1-pixel0;
            gradX.at<float>(src.rows-1,x)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);
        }

    }

    if(src.rows>2)
    {
        for(y=1 ; y < src.rows-1; y++)
        {
             for(x=0 ; x < src.cols ; x++)
             {
                 pixel1=src.at<Vec3b>(y+1,x);
                 pixel0=src.at<Vec3b>(y-1,x);
                 pixelDiff=pixel1-pixel0;
                 gradX.at<float>(y,x)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);
             }
        }

    }

    if(src.cols>1)
    {

        for( y=0 ; y < src.rows ; y++ )
        {
            pixel1=src.at<Vec3b>(y,1);
            pixel0=src.at<Vec3b>(y,0);
            pixelDiff=pixel1-pixel0;
            gradY.at<float>(y,0)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);

            pixel1=src.at<Vec3b>(y,src.cols-1);
            pixel0=src.at<Vec3b>(y,src.cols-2);
            pixelDiff=pixel1-pixel0;
            gradY.at<float>(y,src.cols-1)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);
        }

    }

    if(src.cols>2)
    {
        for(x=1 ; x < src.cols-1; x++)
        {
             for(y=0 ; y < src.rows ; y++)
             {
                 pixel1=src.at<Vec3b>(y,x+1);
                 pixel0=src.at<Vec3b>(y,x-1);
                 pixelDiff=pixel1-pixel0;
                 gradY.at<float>(y,x)=-(pixelDiff[0]+pixelDiff[1]+pixelDiff[2])/(3*255.0);

             }
        }

    }
}

Inpainter::Inpainter(Mat inputImage,Mat mask,int halfPatchWidth,int mode){
    this->inputImage=inputImage.clone();
    this->mask=mask.clone();
    this->updatedMask=mask.clone();
    this->workImage=inputImage.clone();
    this->result.create(inputImage.size(),inputImage.type());
    this->mode=mode;
    this->halfPatchWidth=halfPatchWidth;
}


void Inpainter::inpaint()
{
    initializeMats();
    calculateGradients();
    bool stay=true;
    while(stay){

        computeFillFront();
        computeConfidence();
        computeData();
        computeTarget();
        computeBestPatch();
        updateMats();
        stay=checkEnd();
        waitKey(2);
    }
    result=workImage.clone();
}

void Inpainter::calculateGradients()
{
    Mat srcGray;
    cvtColor(workImage,srcGray,CV_BGR2GRAY);

    Scharr(srcGray,gradientX,CV_16S,1,0);
    convertScaleAbs(gradientX,gradientX);
    gradientX.convertTo(gradientX,CV_32F);


    Scharr(srcGray,gradientY,CV_16S,0,1);
    convertScaleAbs(gradientY,gradientY);
    gradientY.convertTo(gradientY,CV_32F);

    for(int x=0;x<sourceRegion.cols;x++){
        for(int y=0;y<sourceRegion.rows;y++){

            if(sourceRegion.at<uchar>(y,x)==0){
                gradientX.at<float>(y,x)=0;
                gradientY.at<float>(y,x)=0;
            }
        }
    }
    gradientX/=255;
    gradientY/=255;
}

void Inpainter::initializeMats(){
    threshold(this->mask,this->confidence,10,255,CV_THRESH_BINARY);
    threshold(confidence,confidence,2,1,CV_THRESH_BINARY_INV);
    confidence.convertTo(confidence,CV_32F);

    this->sourceRegion=confidence.clone();
    this->sourceRegion.convertTo(sourceRegion,CV_8U);
    this->originalSourceRegion=sourceRegion.clone();

    threshold(mask,this->targetRegion,10,255,CV_THRESH_BINARY);
    threshold(targetRegion,targetRegion,2,1,CV_THRESH_BINARY);
    targetRegion.convertTo(targetRegion,CV_8U);
    data=Mat(inputImage.rows,inputImage.cols,CV_32F,Scalar::all(0));


    LAPLACIAN_KERNEL=Mat::ones(3,3,CV_32F);
    LAPLACIAN_KERNEL.at<float>(1,1)=-8;
    NORMAL_KERNELX=Mat::zeros(3,3,CV_32F);
    NORMAL_KERNELX.at<float>(1,0)=-1;
    NORMAL_KERNELX.at<float>(1,2)=1;
    transpose(NORMAL_KERNELX,NORMAL_KERNELY);
}
void Inpainter::computeFillFront()
{
    Mat sourceGradientX,sourceGradientY,boundryMat;
    filter2D(targetRegion,boundryMat,CV_32F,LAPLACIAN_KERNEL);
    filter2D(sourceRegion,sourceGradientX,CV_32F,NORMAL_KERNELX);
    filter2D(sourceRegion,sourceGradientY,CV_32F,NORMAL_KERNELY);
    fillFront.clear();
    normals.clear();
    for(int x=0;x<boundryMat.cols;x++)
    {
        for(int y=0;y<boundryMat.rows;y++)
        {

            if(boundryMat.at<float>(y,x)>0)
            {
                fillFront.push_back(Point2i(x,y));

                float dx=sourceGradientX.at<float>(y,x);
                float dy=sourceGradientY.at<float>(y,x);
                Point2f normal(dy,-dx);
                float tempF=std::sqrt((normal.x*normal.x)+(normal.y*normal.y));
                if(tempF!=0)
                {

                normal.x=normal.x/tempF;
                normal.y=normal.y/tempF;

                }
                normals.push_back(normal);

            }
        }
    }
}

void Inpainter::computeConfidence(){
    Point2i a,b;
    for(int i=0;i<fillFront.size();i++)
    {
        Point2i currentPoint=fillFront.at(i);
        getPatch(currentPoint,a,b);
        float total=0;
        for(int x=a.x;x<=b.x;x++)
        {
            for(int y=a.y;y<=b.y;y++)
            {
                if(targetRegion.at<uchar>(y,x)==0){
                    total+=confidence.at<float>(y,x);
                }
            }
        }
        confidence.at<float>(currentPoint.y,currentPoint.x)=total/((b.x-a.x+1)*(b.y-a.y+1));
    }
}

void Inpainter::computeData()
{
    for(int i=0;i<fillFront.size();i++)
    {
        Point2i currentPoint=fillFront.at(i);
        Point2i currentNormal=normals.at(i);
        data.at<float>(currentPoint.y,currentPoint.x)=std::fabs(gradientX.at<float>(currentPoint.y,currentPoint.x)*currentNormal.x+gradientY.at<float>(currentPoint.y,currentPoint.x)*currentNormal.y)+.001;
    }
}

void Inpainter::computeTarget()
{
    targetIndex=0;
    float maxPriority=0;
    float priority=0;
    Point2i currentPoint;
    for(int i=0;i<fillFront.size();i++)
    {
        currentPoint=fillFront.at(i);
        priority=data.at<float>(currentPoint.y,currentPoint.x)*confidence.at<float>(currentPoint.y,currentPoint.x);
        if(priority>maxPriority)
        {
            maxPriority=priority;
            targetIndex=i;
        }
    }

}

void Inpainter::computeBestPatch()
{
    double minError=10E15,bestPatchVarience=10E15;
    Point2i a,b;
    Point2i currentPoint=fillFront.at(targetIndex);
    Vec3b sourcePixel,targetPixel;
    double meanR,meanG,meanB;
    double difference,patchError;
    bool skipPatch;
    getPatch(currentPoint,a,b);

    int width=b.x-a.x+1;
    int height=b.y-a.y+1;
    for(int x=0;x<=workImage.cols-width;x++)
    {
        for(int y=0;y<=workImage.rows-height;y++)
        {
            patchError=0;
            meanR=0;meanG=0;meanB=0;
            skipPatch=false;

            for(int x2=0;x2<width;x2++)
            {
                for(int y2=0;y2<height;y2++)
                {
                    if(originalSourceRegion.at<uchar>(y+y2,x+x2)==0)
                    {
                        skipPatch=true;
                        break;
                    }

                    if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0)
                        continue;

                    sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                    targetPixel=workImage.at<Vec3b>(a.y+y2,a.x+x2);

                    for(int i=0;i<3;i++)
                    {
                        difference=sourcePixel[i]-targetPixel[i];
                        patchError+=difference*difference;
                    }
                    meanB+=sourcePixel[0];meanG+=sourcePixel[1];meanR+=sourcePixel[2];


                }
                if(skipPatch)
                    break;
            }

            if(skipPatch)
                continue;
            if(patchError<minError)
            {
                minError=patchError;
                bestMatchUpperLeft=Point2i(x,y);
                bestMatchLowerRight=Point2i(x+width-1,y+height-1);

                double patchVarience=0;
                for(int x2=0;x2<width;x2++)
                {
                    for(int y2=0;y2<height;y2++)
                    {
                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0)
                        {
                            sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                            difference=sourcePixel[0]-meanB;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[1]-meanG;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[2]-meanR;
                            patchVarience+=difference*difference;
                        }
                    }
                }
                bestPatchVarience=patchVarience;

            }else if(patchError==minError)
            {
                double patchVarience=0;
                for(int x2=0;x2<width;x2++)
                {
                    for(int y2=0;y2<height;y2++)
                    {
                        if(sourceRegion.at<uchar>(a.y+y2,a.x+x2)==0)
                        {
                            sourcePixel=workImage.at<Vec3b>(y+y2,x+x2);
                            difference=sourcePixel[0]-meanB;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[1]-meanG;
                            patchVarience+=difference*difference;
                            difference=sourcePixel[2]-meanR;
                            patchVarience+=difference*difference;
                        }

                    }
                }
                if(patchVarience<bestPatchVarience)
                {
                    minError=patchError;
                    bestMatchUpperLeft=Point2i(x,y);
                    bestMatchLowerRight=Point2i(x+width-1,y+height-1);
                    bestPatchVarience=patchVarience;
                }
            }
        }
    }
}



void Inpainter::updateMats()
{
    Point2i targetPoint=fillFront.at(targetIndex);
    Point2i a,b;
    getPatch(targetPoint,a,b);
    int width=b.x-a.x+1;
    int height=b.y-a.y+1;

    for(int x=0;x<width;x++)
    {
        for(int y=0;y<height;y++)
        {
            if(sourceRegion.at<uchar>(a.y+y,a.x+x)==0)
            {
                workImage.at<Vec3b>(a.y+y,a.x+x)=workImage.at<Vec3b>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                gradientX.at<float>(a.y+y,a.x+x)=gradientX.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                gradientY.at<float>(a.y+y,a.x+x)=gradientY.at<float>(bestMatchUpperLeft.y+y,bestMatchUpperLeft.x+x);
                confidence.at<float>(a.y+y,a.x+x)=confidence.at<float>(targetPoint.y,targetPoint.x);
                sourceRegion.at<uchar>(a.y+y,a.x+x)=1;
                targetRegion.at<uchar>(a.y+y,a.x+x)=0;
                updatedMask.at<uchar>(a.y+y,a.x+x)=0;
            }
        }
    }
}

bool Inpainter::checkEnd()
{
    for(int x=0;x<sourceRegion.cols;x++)
    {
        for(int y=0;y<sourceRegion.rows;y++)
        {
            if(sourceRegion.at<uchar>(y,x)==0)
            {
                return true;
            }
        }
    }
    return false;
}
void Inpainter::getPatch(Point2i &centerPixel, Point2i &upperLeft, Point2i &lowerRight)
{
    int x,y;
    x=centerPixel.x;
    y=centerPixel.y;

    int minX=std::max(x-halfPatchWidth,0);
    int maxX=std::min(x+halfPatchWidth,workImage.cols-1);
    int minY=std::max(y-halfPatchWidth,0);
    int maxY=std::min(y+halfPatchWidth,workImage.rows-1);


    upperLeft.x=minX;
    upperLeft.y=minY;

    lowerRight.x=maxX;
    lowerRight.y=maxY;
}