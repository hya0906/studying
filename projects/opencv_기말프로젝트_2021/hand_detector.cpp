#include "hand_detector.h"

//* initialize static variable
String HandDetector::trackbar_window_name_ = "debug image";

int HandDetector::low_y = 80;
int HandDetector::low_cr = 133;
int HandDetector::low_cb = 101;
int HandDetector::high_y = 184;
int HandDetector::high_cr = 159;
int HandDetector::high_cb = 255;

int HandDetector::final_cnt = 0;
string HandDetector::cal;

int HandDetector::result_flag = 0;
int HandDetector::flag_ = 0;

Mat HandDetector::calculate = Mat::ones(500, 500, CV_8UC3);
vector<Mat> HandDetector::pre_calculate = vector<Mat>();
int HandDetector::m = 0;

static struct oper {
    int p; // 연산자 우선순위
    string o; // 연산자
};
static stack<int> num; // 숫자 스택
static stack<oper> op; // 연산자 스택

HandDetector::HandDetector() {
    namedWindow(trackbar_window_name_);
    resizeWindow(trackbar_window_name_, 300, 300);
}

void HandDetector::showing(string a, string b) {
    pre_calculate.pop_back();
    pre_calculate.push_back(calculate.clone());
    putText(calculate, a, Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
    cal += b;
    m += 1;
    pre_calculate.push_back(calculate.clone());
    imshow("calculate", calculate);
}

void HandDetector::calc() {
    int a, b, result;
    b = num.top();
    num.pop();
    a = num.top();
    num.pop();
    string oper = op.top().o;
    op.pop();

    if (oper == "*")
        result = a * b;
    else if (oper == "/")
        result = a / b;
    else if (oper == "+")
        result = a + b;
    else if (oper == "-")
        result = a - b;
    // 결과 값 스택에 다시 저장
    num.push(result);
}

void HandDetector::result(string cal) {
    string input = cal;// "15 + 32 * ( 1 - 8 ) / 2"; // -97
    stringstream ss(input);
    cout << input << endl;
    // 연산자 우선순위에 따라 스택에 push
    // 0 : ( )
    // 1 : + -
    // 2 : * /
    string tok;
    while (ss >> tok) {
        // ( 는 무조건 연산자 스택에 push
        if (tok == "(") {
            op.push({ 0, tok });
        } // ) 가 나오면 ( 가 나올 때 까지 계산
        else if (tok == ")") {
            while (op.top().o != "(")
                calc();
            op.pop();
        }
        else if (tok == "*" || tok == "/" || tok == "+" || tok == "-") {
            int prior; // 연산자 우선순위
            if (tok == "*")
                prior = 2;
            else if (tok == "/")
                prior = 2;
            else if (tok == "+")
                prior = 1;
            else if (tok == "-")
                prior = 1;

            // 연산자 우선 순위 낮은게 top으로 올 때까지 계산
            while (!op.empty() && prior <= op.top().p)
                calc();
            // 스택에 연산자 push
            op.push({ prior, tok });
        }
        else // 숫자일 경우 숫자 스택에 push
            num.push(stoi(tok));
    }
    // 남은 연산자 계산
    while (!op.empty())
        calc();
    putText(calculate, to_string(num.top()), Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
    m += 1;
    cout << num.top() << endl;
}

void HandDetector::on_mouse(int event, int x, int y, int flag, void* me) {
    switch (event) {
    case EVENT_LBUTTONDOWN:
        if (pre_calculate.size() == 0) {
            pre_calculate.push_back(calculate.clone());
            pre_calculate.push_back(calculate.clone());
        }
        if (((50 < x) & (x < 50 + 70)) & ((200 < y) & (y < 270))) //더하기
            showing("+", " + ");

        else if (((50 + 110 < x) & (x < 50 + 110 + 70)) & ((200 < y) & (y < 270))) //빼기
            showing("-", " - ");

        else if (((50 + (110 * 2) < x) & (x < 50 + (110 * 2) + 70)) & ((200 < y) & (y < 270))) //곱하기
            showing("x", " * ");

        else if (((50 + (110 * 3) < x) & (x < 50 + (110 * 3) + 70)) & ((200 < y) & (y < 270))) //나누기
            showing("%", " / ");

        else if (((50 < x) & (x < 50 + 195)) & ((290 < y) & (y < 360))) { // 숫자
            putText(calculate, to_string(final_cnt), Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
            m += 1;
            cal += to_string(final_cnt);
            pre_calculate.push_back(calculate.clone());
            imshow("calculate", calculate);
        }
        else if (((50 + (205 * 1) < x) & (x < 50 + (205 * 1) + 195)) & ((290 < y) & (y < 360))) { //지우기
            if (result_flag == 1) {
                result_flag = 0;
                m = 0;
                cal = "";
                while (pre_calculate.size() > 1)
                    pre_calculate.pop_back();
                pre_calculate.push_back(pre_calculate[0].clone());
                calculate = pre_calculate.back();
            }
            if ((pre_calculate.size() > 0) & (m > 0)) {
                pre_calculate.pop_back();
                calculate = pre_calculate.back();
                m -= 1;
                if (isdigit(cal.back())) { //숫자
                    if (cal.size() <= 1)
                        cal = "";
                    else
                        cal = cal.substr(0, cal.size() - 1);
                }
                if (isspace(int(cal.back())) != 0) {
                    cal.pop_back();
					if (to_string(cal.back()).compare("(")) { //괄호열기
						if (cal.size() == 1)
							cal = "";
						else {
							cal = cal.substr(0, cal.size() - 1);
							flag_ = 0;
						}
					}
                    else if (to_string(cal.back()).compare("+") || to_string(cal.back()).compare("-") || to_string(cal.back()).compare("*") || to_string(cal.back()).compare("/")) {//연산자
                        if (cal.size() == 1)
                            cal = "";
                        else
                            cal = cal.substr(0, cal.size() - 2);
                    }
                }
				else if (to_string(cal.back()).compare(")")) { //괄호닫기
					cal = cal.substr(0, cal.size() - 2);
					flag_ = 1;
				}
                imshow("calculate", calculate);
            }
            if (m == 0) {
                cout << "지울것이 없습니다." << endl;
                while (pre_calculate.size() > 1)
                    pre_calculate.pop_back();
				flag_ = 0;
                pre_calculate.push_back(pre_calculate[0].clone());
                cout << "cal" << cal << "-> " << cal.size() << endl;
            }
        }
        else if (((50 < x) & (x < 50 + 70)) & ((380 < y) & (y < 450))) { //괄호
            pre_calculate.pop_back();
            pre_calculate.push_back(calculate.clone());
            if (flag_ == 0) {
                putText(calculate, "(", Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
                cal += "( ";
                flag_ = 1;
            }
            else if (flag_ == 1) {
                putText(calculate, ")", Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
                cal += " )";
                flag_ = 0;
            }
            pre_calculate.push_back(calculate.clone());
            m += 1;
            imshow("calculate", calculate);
        }
        else if ((50 + 110 < x) & (x < 50 + 110 + 70) & ((380 < y) & (y < 450))) { // 결론 = 
            calculate = pre_calculate.back();
            putText(calculate, "=", Point(80 + (m * 20), 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
            pre_calculate.push_back(calculate.clone());
            m += 1;
            result(cal);
            result_flag = 1;
            imshow("calculate", calculate);
        }
        break;
    default:
        break;
    }
}

void HandDetector::ycrcb_on_low_y_thresh_trackbar(int, void*)
{
    low_y = min(high_y - 1, low_y);
    setTrackbarPos("Low y", trackbar_window_name_, low_y);
}

void HandDetector::ycrcb_on_high_y_thresh_trackbar(int, void*)
{
    high_y = max(high_y, low_y + 1);
    setTrackbarPos("High y", trackbar_window_name_, high_y);
}

void HandDetector::ycrcb_on_low_cr_thresh_trackbar(int, void*)
{
    low_cr = min(high_cr - 1, low_cr);
    setTrackbarPos("Low cr", trackbar_window_name_, low_cr);
}

void HandDetector::ycrcb_on_high_cr_thresh_trackbar(int, void*)
{
    high_cr = max(high_cr, low_cr + 1);
    setTrackbarPos("High cr", trackbar_window_name_, high_cr);
}

void HandDetector::ycrcb_on_low_cb_thresh_trackbar(int, void*)
{
    low_cb = min(high_cb - 1, low_cb);
    setTrackbarPos("Low cb", trackbar_window_name_, low_cb);
}

void HandDetector::ycrcb_on_high_cb_thresh_trackbar(int, void*)
{
    high_cb = max(high_cb, low_cb + 1);
    setTrackbarPos("High cb", trackbar_window_name_, high_cb);
}

int HandDetector::averageFilter(int value) {
    static vector<int> value_list;
    static int average_config = 20;
    int result = 0;

    value_list.push_back(value);

    if (value_list.size() > average_config) {
        value_list.erase(value_list.begin());
    }

    for (int i = 0; i < value_list.size(); i++)
    {
        result += value_list[i];
    }

    return int(result / value_list.size());
}

inline void HandDetector::set_Roi_Img(Rect rect, Mat& target_image){
    target_image = image_(rect);
}

void HandDetector::preprocess_Img(Mat& target_image) {
    cvtColor(target_image, target_image, COLOR_BGR2YCrCb);

    inRange(target_image, Scalar(low_y, low_cr, low_cb), Scalar(high_y, high_cr, high_cb), target_image);

    GaussianBlur(target_image, target_image, Point(3, 3), 0);

    threshold(target_image, target_image, 0, 255, THRESH_BINARY);
}

void HandDetector::find_Contours(const Mat target_image, vector<vector<Point>>& contours) {
    vector<vector<Point>> contours_sample;
    vector<Vec4i> hierarchy;
    findContours(target_image, contours_sample, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours_sample.size(); i++)
    {
        if (contourArea(contours_sample[i]) >= 2000)
            contours.push_back(contours_sample[i]);
    }
}

void HandDetector::find_Hull(vector<vector<Point>> contours, vector<vector<int>>& hulls, vector<vector<Vec4i>>& defects){
    
    hulls.resize(contours.size());
    defects.resize(contours.size());

    for (int i = 0; i < contours.size(); i++) {
        //convexHull(Mat(contours[i]), hull[i], false);
        convexHull(Mat(contours[i]), hulls[i], true);
        convexityDefects(contours[i], hulls[i], defects[i]);
    }

}

int HandDetector::count_Finger(vector<vector<Point>> contours, vector<vector<Vec4i>> defects) {
    int s, e, f, d;// start[2];
    int cnt = 0;
    
    for (int idx = 0; idx < defects.size(); idx++) {
        if (defects[idx].empty() && contourArea(contours[idx]) <= 2000)
            continue;

        for (int i = 0; i < defects[idx].size(); i++) {
            s = defects[idx][i][0];
            e = defects[idx][i][1];
            f = defects[idx][i][2];
            d = defects[idx][i][3];

            Point2i start = contours[idx][s];
            Point2i end = contours[idx][e];
            Point2i far = contours[idx][f];

            double a = sqrt(pow(end.x - start.x, 2) + pow(end.y - start.y, 2));
            double b = sqrt(pow(far.x - start.x, 2) + pow(far.y - start.y, 2));
            double c = sqrt(pow(end.x - far.x, 2) + pow(end.y - far.y, 2));
            double angle = acos((b * b + c * c - a * a) / (2 * b * c));

            if (angle <= M_PI / 3) {
                if (cnt < 5)
                    cnt += 1;
            }
        }
    }

    if (defects.empty()) {
        cnt = 0;
    }
    else {
        cnt++;
    }

    return cnt;
}

void HandDetector::run() {

    createTrackbar("Low y", trackbar_window_name_, &low_y, max_value, &HandDetector::ycrcb_on_low_y_thresh_trackbar, this);
    createTrackbar("High y", trackbar_window_name_, &high_y, max_value, &HandDetector::ycrcb_on_high_y_thresh_trackbar, this);
    createTrackbar("Low cr", trackbar_window_name_, &low_cr, max_value, &HandDetector::ycrcb_on_low_cr_thresh_trackbar, this);
    createTrackbar("High cr", trackbar_window_name_, &high_cr, max_value, &HandDetector::ycrcb_on_high_cr_thresh_trackbar, this);
    createTrackbar("Low cb", trackbar_window_name_, &low_cb, max_value, &HandDetector::ycrcb_on_low_cb_thresh_trackbar, this);
    createTrackbar("High cb", trackbar_window_name_, &high_cb, max_value, &HandDetector::ycrcb_on_high_cb_thresh_trackbar, this);
   
    cap.open(device_id + api_id);

    if (!cap.isOpened()) {
        cout << "unable to open camera!\n";
        return;
    }

	calculate = Mat::ones(500, 500, CV_8UC3);
	calculate.setTo(256);
	line(calculate, Point(0, 180), Point(500, 180), Scalar(0, 0, 255), 3, LINE_AA);
	
	for (int i = 0; i <= 7; i++) {
		if (i < 4)
			rectangle(calculate, Rect(50 + (110 * i), 200, 70, 70), Scalar(0, 0, 0), 2);
		else if (i < 6) {
			rectangle(calculate, Rect(50 + (205 * (i - 4)), 290, 195, 70), Scalar(0, 0, 0), 2);
		}
		else if (i < 8) {
			rectangle(calculate, Rect(50 + (110 * (i - 6)), 380, 70, 70), Scalar(0, 0, 0), 2);
		}
		if (i == 0)drawMarker(calculate, Point(85 + (110 * i), 235), Scalar(0, 0, 0), MARKER_CROSS, 50);
		else if (i == 1)putText(calculate, "-", Point(55 + (110 * i), 260), FONT_HERSHEY_PLAIN, 5, Scalar(0, 0, 0), 1, LINE_AA);
		else if (i == 2)drawMarker(calculate, Point(85 + (110 * i), 235), Scalar(0, 0, 0), MARKER_TILTED_CROSS, 30);
		else if (i == 3)putText(calculate, "%", Point(55 + (110 * i), 260), FONT_HERSHEY_PLAIN, 5, Scalar(0, 0, 0), 1, LINE_AA);
		else if (i == 4)putText(calculate, "append", Point(80 + (205 * (i - 4)), 330), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
		else if (i == 5)putText(calculate, "remove", Point(80 + (205 * (i - 4)), 330), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
		else if (i == 6)putText(calculate, "()", Point(70 + (110 * (i - 6)), 425), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
		else if (i == 7)putText(calculate, "=", Point(70 + (110 * (i - 6)), 425), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 1, LINE_AA);
	}

    while (1) {
        Mat image;
        cap.read(image);

        if (image.empty()) {
            cout << "Image load failed!" << endl;
            return;
        }

        resize(image, image, Size(500, 500));
        image_ = image.clone();

        //* set roi image
        set_Roi_Img(Rect(25 , 100, 150, 300), left_hand_img_);
        set_Roi_Img(Rect(320, 100, 150, 300), right_hand_img_);

        preprocess_Img(left_hand_img_);
        preprocess_Img(right_hand_img_);

        Mat debug;
        hconcat(left_hand_img_, right_hand_img_, debug);
        imshow(trackbar_window_name_, debug);

        //* find contour
        vector<vector<Point>> left_hand_contours;
        vector<vector<Point>> right_hand_contours;
        find_Contours(left_hand_img_, left_hand_contours);
        find_Contours(right_hand_img_, right_hand_contours);

        vector<vector<int>> left_hand_hull;
        vector<vector<Vec4i>> left_hand_defects;
        vector<vector<int>> right_hand_hull;
        vector<vector<Vec4i>> right_hand_defects;
        find_Hull(left_hand_contours, left_hand_hull, left_hand_defects);
        find_Hull(right_hand_contours, right_hand_hull, right_hand_defects);

        int left_finger_cnt = count_Finger(left_hand_contours, left_hand_defects);
        int right_finger_cnt = count_Finger(right_hand_contours, right_hand_defects);
        final_cnt = averageFilter(left_finger_cnt + right_finger_cnt);
        if (final_cnt > 10) final_cnt = 10;
        putText(image, to_string(final_cnt), Point2i(0, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);

        rectangle(image, Rect(25, 100, 150, 300), Scalar(0, 0, 255), 1, LINE_AA, 0);
        rectangle(image, Rect(320, 100, 150, 300), Scalar(0, 0, 255), 1, LINE_AA, 0);
  
        //putText(calculate, to_string(final_cnt), Point2i(400, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);
        namedWindow("calculate");
        setMouseCallback("calculate", on_mouse);
		
		imshow("final_result", image);
        imshow("calculate", calculate);

        if (waitKey(5) == 27)
            break;
    }
}
