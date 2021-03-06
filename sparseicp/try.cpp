#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <sstream>
#include <math.h>
#include "ICP.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>


typedef pcl::PointXYZRGB ColorPoint;
typedef pcl::PointCloud<ColorPoint> ColorCloud;
typedef pcl::visualization::PCLVisualizer CloudViz;
typedef boost::shared_ptr<CloudViz> CloudVizPtr;
typedef  pcl::PointCloud<pcl::Normal> NormalCloud;


using namespace std;
using namespace Eigen;

CloudVizPtr create_viewer() {
	CloudVizPtr cviewer(new CloudViz("viewer"));
  cviewer->setBackgroundColor (0, 0, 0);
  cviewer->addCoordinateSystem (1.0);
  cviewer->initCameraParameters ();
  return (cviewer);
}
CloudVizPtr viewer = create_viewer();

void add_point_cloud(ColorCloud::Ptr cloud, std::string cname){
	pcl::visualization::PointCloudColorHandlerRGBField<ColorPoint> rgb(cloud);
  viewer->addPointCloud<ColorPoint> (cloud, rgb, cname);
}

void spin_viewer() {
	while (!viewer->wasStopped ()) {
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds(100000));
	}
	viewer = create_viewer();
}

/** Converts an eigen-matrix to PCL rgb point-cloud : sets the rgb values to 0.*/
ColorCloud::Ptr eigen_to_pcl(MatrixXf &points) {
  ColorCloud::Ptr cloud(new ColorCloud);
  cloud->width  = points.rows();
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size (); ++i) {
    cloud->points[i].x = points(i,0);
    cloud->points[i].y = points(i,1);
    cloud->points[i].z = points(i,2);
    cloud->points[i].r = cloud->points[i].g = cloud->points[i].b =0;
  }
  return cloud;
}

MatrixXf pcl_to_eigen(ColorCloud::Ptr cloud) {
  return cloud->getMatrixXfMap(3,8,0);
}

MatrixXd float_to_double(MatrixXf & mat) {
	return mat.cast<double>();
}

MatrixXf double_to_float(MatrixXd & mat) {
	return mat.cast<float>();
}

NormalCloud::Ptr get_normals(ColorCloud::Ptr cloud) {
  pcl::NormalEstimation<ColorPoint, pcl::Normal> ne;
  ne.setInputCloud (cloud);

  pcl::search::KdTree<ColorPoint>::Ptr tree (new pcl::search::KdTree<ColorPoint> ());
  ne.setSearchMethod(tree);

  NormalCloud::Ptr cloud_normals (new NormalCloud);
  ne.setRadiusSearch (0.03);
  ne.compute (*cloud_normals);

	return cloud_normals;
}

void do_sparseicp(ColorCloud::Ptr c1, ColorCloud::Ptr c2) {
        MatrixXf c1e = pcl_to_eigen(c1);
	MatrixXf c2e = pcl_to_eigen(c2);
	c1e.transposeInPlace();
	c2e.transposeInPlace();

	Vector3f dd(0,0,.01);
	c2e.colwise() += dd;

	MatrixXd tmp = float_to_double(c1e);	
	Matrix3Xd c1ed = tmp;
	tmp = float_to_double(c2e);
	Matrix3Xd c2ed = tmp;

	// get cloud normals:
	NormalCloud::Ptr	cloud_normals = get_normals(c1);
        MatrixXf eig_norm = cloud_normals->getMatrixXfMap(3,8,0);
	eig_norm.transposeInPlace();
	tmp = float_to_double(eig_norm);
	Matrix3Xd c1ed_normals = tmp;

	// do sparse icp
	cout <<"doing sparse icp"<<endl;
	SICP::Parameters params = SICP::Parameters();
	params.use_penalty = true;
	params.p = 0.5;
	params.stop = 1e-2;
	params.max_icp = 10;
	params.max_inner = 1;
	params.max_outer = 10;

	SICP::point_to_plane(c2ed, c1ed, c1ed_normals, params);
	//SICP::point_to_point(c2ed, c1ed, params);
	
	cout <<"sparse icp done"<<endl;

	// convert to float and transpose
	MatrixXd tmpf;
	tmpf = c1ed.transpose();
	c1e = double_to_float(tmpf);
	tmpf = c2ed.transpose();
	c2e = double_to_float(tmpf); 

	ColorCloud::Ptr c1new = eigen_to_pcl(c1e);
	ColorCloud::Ptr c2new = eigen_to_pcl(c2e);

	for (size_t i = 0; i < c1new->points.size (); ++i) {
    c1new->points[i].r = c1->points[i].r;
    c1new->points[i].g = c1->points[i].g;
    c1new->points[i].b = c1->points[i].b;
  }
	for (size_t i = 0; i < c2new->points.size (); ++i) {
    c2new->points[i].r = c2->points[i].r;
    c2new->points[i].g = c2->points[i].g;
    c2new->points[i].b = c2->points[i].b;
  }
	add_point_cloud(c1new, "c1");
	add_point_cloud(c2new, "c2");
	spin_viewer();
}

ColorCloud::Ptr downsampleCloud(const ColorCloud::Ptr in, float sz) {
  pcl::PointCloud<ColorPoint>::Ptr out(new pcl::PointCloud<ColorPoint>());
  pcl::VoxelGrid<ColorPoint> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(sz,sz,sz);
  vg.filter(*out);
  return out;
}



int main (int argc, char** argv) {
  ColorCloud::Ptr cloud1(new ColorCloud), cloud2(new ColorCloud);
  
  // Get input object and scene
  if (argc != 3)  {
    pcl::console::print_error ("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
    return (1);
  }
   // Load object and scene

  if (pcl::io::loadPCDFile<ColorPoint>(argv[1], *cloud1) < 0 ||
      pcl::io::loadPCDFile<ColorPoint>(argv[2], *cloud2) < 0) {
    
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
    }
  
  //cloud1 = downsampleCloud(cloud1, 0.01);
  //cloud2 = downsampleCloud(cloud2, 0.01);
  cout << "c1 num points : "<<cloud1->points.size()<<endl;
  cout << "c2 num points : "<<cloud2->points.size()<<endl;
  do_sparseicp(cloud1, cloud2);
  return 0;
}

