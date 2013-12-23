cmake_minimum_required(VERSION 2.8)
project( simple_capture )

include_directories(/home/ankush/sources/OpenNI2/Include 
                    /home/ankush/sources/openCV/OpenCV-2.4.2/include/opencv)

link_directories(/home/ankush/sources/OpenNI2/Bin/x64-Release)

add_executable(simple_capture simple_capture.cpp )
target_link_libraries(simple_capture  opencv_core opencv_highgui opencv_imgproc opencv_legacy
                                      OpenNI2)