//#ifndef HAND_DETECTOR_
//#define HAND_DETECTOR_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <stack>
#include <ctype.h>

# define M_PI 3.14159265358979323846 

using namespace cv;
using namespace std;

class HandDetector {
public:

    HandDetector(); 

    void run();

private:

    static String trackbar_window_name_;
    static int low_y;
    static int low_cr;
    static int low_cb;
    static int high_y;
    static int high_cr;
    static int high_cb;

    static int final_cnt;
    static int m;
    static Mat calculate;
    static vector <Mat> pre_calculate;
    static string cal;
    static int result_flag;
    static int flag_;

    const int max_value_H = 360 / 2;
    const int max_value = 255;

    int device_id = 0;
    int api_id = CAP_ANY;

    VideoCapture cap;
    Mat image_;
    Mat left_hand_img_;
    Mat right_hand_img_;

private:
    static void on_mouse(int event, int x, int y, int flag, void* me);
    static void result(string cal);
    static void calc();

    static void ycrcb_on_low_y_thresh_trackbar(int, void*);
    static void ycrcb_on_high_y_thresh_trackbar(int, void*);
    static void ycrcb_on_low_cr_thresh_trackbar(int, void*);
    static void ycrcb_on_high_cr_thresh_trackbar(int, void*);
    static void ycrcb_on_low_cb_thresh_trackbar(int, void*);
    static void ycrcb_on_high_cb_thresh_trackbar(int, void*);

    static void showing(string a, string b);

    inline void set_Roi_Img(Rect rect, Mat& target_image);
    void preprocess_Img(Mat& target_image);
    void find_Contours(const Mat target_image, vector<vector<Point>>& contours);
    void find_Hull(vector<vector<Point>> contours, vector<vector<int>>& hulls, vector<vector<Vec4i>>& defects);
    int count_Finger(vector<vector<Point>> contours, vector<vector<Vec4i>> defects);
    int averageFilter(int cnt);

};
//#endif // HAND_DETECTOR_
