#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

typedef pcl::PointXYZRGB PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<PointNT>::ConstPtr cloud) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointNT> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->initCameraParameters ();
  return (viewer);
}


int main(int argc, char** argv) {
  PointCloudT::Ptr cloud (new PointCloudT);
  PointCloudT::Ptr final (new PointCloudT);
  
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointNT> (argv[1], *cloud) < 0)  {
    pcl::console::print_error ("Error loading object file!\n");
    return (1);
  }
  
  std::vector<int> inlier_indices;

  // created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelPlane<PointNT>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<PointNT> (cloud));

  if(pcl::console::find_argument (argc, argv, "-f") >= 0) {
    pcl::RandomSampleConsensus<PointNT> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inlier_indices);
  }

  // get the points which do not lie on a plane:
  pcl::ExtractIndices<PointNT> extract;
  extract.setInputCloud (cloud);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  inliers->indices = inlier_indices;
  extract.setIndices (inliers);
  extract.setNegative (true);
  extract.filter (*final);

  std::cout << "saving out-of-plane points"<<std::endl;
  pcl::io::savePCDFileASCII("out_plane.pcd", *final);
  
  // creates the visualization object and adds either our orignial cloud or all of the inliers
  // depending on the command line arguments specified.
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
    viewer = simpleVis(final);
  else viewer = simpleVis(cloud);

 
  while (!viewer->wasStopped ())  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  } 

  return 0;
 }
