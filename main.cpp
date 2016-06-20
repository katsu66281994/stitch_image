#include <iostream>      //includeはこんなにいらないと思う
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

  //時間表示
  const int N = 1000*1000;
  std::vector<int> v;
  auto start = std::chrono::system_clock::now();      

  //読み込む画像のパス
  cv::String scene1_path = "left.jpg";
  cv::String scene2_path = "right.jpg";

  //書き出す画像のパス
  cv::String scene_12_path = "tokuchouryou.jpg";
  cv::String scene_123_path = "gousei_result.jpg";
  cv::String after_homo = "after_homo.jpg";


  //比較用画像を読み込む
  cv::Mat scene1 = cv::imread(scene1_path, 1);	
  cv::Mat scene2 = cv::imread(scene2_path, 1);


  //アルゴリズムにAKAZEを使用する
  auto algorithm = cv::AKAZE::create();

  // 特徴点抽出
  std::vector<cv::KeyPoint> keypoint1, keypoint2;
  algorithm->detect(scene1, keypoint1);
  algorithm->detect(scene2, keypoint2);

  // 特徴記述
  cv::Mat descriptor1, descriptor2;
  algorithm->compute(scene1, keypoint1, descriptor1);
  algorithm->compute(scene2, keypoint2, descriptor2);
  
  // マッチング (アルゴリズムにはBruteForceを使用)
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  std::vector<cv::DMatch> match, match12, match21;
  matcher->match(descriptor1, descriptor2, match12);
  matcher->match(descriptor2, descriptor1, match21);

  //クロスチェック(1→2と2→1の両方でマッチしたものだけを残して精度を高める)
  for (size_t i = 0; i < match12.size(); i++)
  {
    cv::DMatch forward = match12[i];
    cv::DMatch backward = match21[forward.trainIdx];
    if (backward.trainIdx == forward.queryIdx)
    {
      match.push_back(forward);
    }
  }

  // マッチング結果の描画
  cv::Mat dest;
  cv::drawMatches(scene1, keypoint1, scene2, keypoint2, match, dest);

  //マッチング結果の書き出し
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


  //ホモグラフィ
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

   //時間表示
   auto end = std::chrono::system_clock::now();       // 計測終了時刻を保存
   auto dur = end - start;        // 要した時間を計算
   auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
   // 要した時間をミリ秒（1/1000秒）に変換して表示
   std::cout << msec << " milli sec \n";
   getchar();


  return 0;

}
