#include <iostream>      //include�͂���Ȃɂ���Ȃ��Ǝv��
#include <vector>
#include <windows.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

//#include <opencv2/features2d/nonfree.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include<opencv2/stitching.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world300d.lib")
#pragma comment(lib, "opencv_ts300d.lib")
#endif


using namespace cv;

int main()
{
  using cv::Mat;
  using std::vector;
  using cv::Size;
  using cv::Vec3b;

  Mat result;

  //���ԕ\��
  const int N = 1000*1000;
  std::vector<int> v;
  auto start = std::chrono::system_clock::now();      

  //�ǂݍ��މ摜�̃p�X
  cv::String scene1_path = "left.jpg";
  cv::String scene2_path = "right.jpg";

  //�����o���摜�̃p�X
  cv::String scene_12_path = "tokuchouryou.jpg";
  cv::String scene_123_path = "gousei_result.jpg";
  cv::String after_homo = "after_homo.jpg";


  //��r�p�摜��ǂݍ���
  cv::Mat scene1 = cv::imread(scene1_path, 1);	
  cv::Mat scene2 = cv::imread(scene2_path, 1);


  //�A���S���Y����AKAZE���g�p����
  auto algorithm = cv::AKAZE::create();

  // �����_���o
  std::vector<cv::KeyPoint> keypoint1, keypoint2;
  algorithm->detect(scene1, keypoint1);
  algorithm->detect(scene2, keypoint2);

  // �����L�q
  cv::Mat descriptor1, descriptor2;
  algorithm->compute(scene1, keypoint1, descriptor1);
  algorithm->compute(scene2, keypoint2, descriptor2);
  
  // �}�b�`���O (�A���S���Y���ɂ�BruteForce���g�p)
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  std::vector<cv::DMatch> match, match12, match21;
  matcher->match(descriptor1, descriptor2, match12);
  matcher->match(descriptor2, descriptor1, match21);

  //�N���X�`�F�b�N(1��2��2��1�̗����Ń}�b�`�������̂������c���Đ��x�����߂�)
  for (size_t i = 0; i < match12.size(); i++)
  {
    cv::DMatch forward = match12[i];
    cv::DMatch backward = match21[forward.trainIdx];
    if (backward.trainIdx == forward.queryIdx)
    {
      match.push_back(forward);
    }
  }

  // �}�b�`���O���ʂ̕`��
  cv::Mat dest;
  cv::drawMatches(scene1, keypoint1, scene2, keypoint2, match, dest);

  //�}�b�`���O���ʂ̏����o��
  cv::imwrite(scene_12_path, dest);
  imshow("matching", dest);
  waitKey(0);

  int num = match.size();
  vector<cv::Vec2f> points1(num);
  vector<cv::Vec2f> points2(num);

  for( size_t i = 0 ; i < match.size() ; ++i )
    {
        points1[i][0] = keypoint1[match[i].queryIdx].pt.x;
        points1[i][1] = keypoint1[match[i].queryIdx].pt.y;

        points2[i][0] = keypoint2[match[i].trainIdx].pt.x;
        points2[i][1] = keypoint2[match[i].trainIdx].pt.y;
    }


  //�z���O���t�B
   Mat homo = cv::findHomography(points1, points2, CV_RANSAC);	//error?
   std::cout << homo << std::endl;
   cv::warpPerspective(scene1, result, homo, Size(static_cast<int>(scene1.cols * 2.0), static_cast<int>(scene1.rows * 1.1)));
  
   /*std::vector<cv::Mat> images;
   images.push_back(result);
   images.push_back(scene2);
   cv::Mat final;
   cv::Stitcher stitcher = cv::Stitcher::createDefault();
   stitcher.stitch(images, final);
   */
   
   cv::imwrite(after_homo,result);
   cv::imshow("arter_homo",result);
   waitKey(0);


   //add
   scene2.copyTo(Mat(result, Rect(0, 0, scene2.cols, scene2.rows)));
   cv::imwrite(scene_123_path, result);
   cv::imshow("win", result);
   
   waitKey(0);

   //���ԕ\��
   auto end = std::chrono::system_clock::now();       // �v���I��������ۑ�
   auto dur = end - start;        // �v�������Ԃ��v�Z
   auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
   // �v�������Ԃ��~���b�i1/1000�b�j�ɕϊ����ĕ\��
   std::cout << msec << " milli sec \n";
   getchar();


  return 0;

}
