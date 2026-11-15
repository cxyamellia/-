#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture cap;
#ifdef _WIN32
    cap.open(0, cv::CAP_DSHOW);
#else
    cap.open(0, cv::CAP_V4L2);
#endif
    if (!cap.isOpened())
    {
        std::cout << "camera open failed" << std::endl;
        return -1;
    }

    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    bool is_tracking = false;
    cv::Scalar lower_hsv = cv::Scalar(0, 0, 180);
    cv::Scalar upper_hsv = cv::Scalar(179, 45, 255);

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Point2f center;
        float radius = 0;
        bool ok = false;

        if (is_tracking)
        {
            cv::Rect2d roi;
            ok = tracker->update(frame, roi);
            if (ok)
            {
                center.x = roi.x + roi.width / 2;
                center.y = roi.y + roi.height / 2;
                radius = std::max(roi.width, roi.height) / 2;
            }
            else
            {
                is_tracking = false;
                tracker = cv::TrackerKCF::create();
            }
        }

        if (!is_tracking)
        {
            cv::Mat hsv, mask;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::inRange(hsv, lower_hsv, upper_hsv, mask);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!contours.empty())
            {
                auto largest = *std::max_element(contours.begin(), contours.end(),
                    [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                    {
                        return cv::contourArea(a) < cv::contourArea(b);
                    });
                double area = cv::contourArea(largest);
                if (area > 300)
                {
                    cv::Rect rect = cv::boundingRect(largest);
                    cv::RotatedRect ellipse = cv::fitEllipse(largest);
                    double circularity = 4 * CV_PI * area / (ellipse.size.width * ellipse.size.height * CV_PI);
                    if (circularity >= 0.7)
                    {
                        center = ellipse.center;
                        radius = ellipse.size.width / 2;
                        tracker->init(frame, rect);
                        is_tracking = true;
                        ok = true;
                    }
                }
            }
        }

        if (ok)
        {
            cv::circle(frame, center, int(radius), cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);
            std::cout << "volley center: " << center.x << " " << center.y << std::endl;
        }

        cv::imshow("volley tracking", frame);
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'r')
        {
            is_tracking = false;
            tracker = cv::TrackerKCF::create();
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
