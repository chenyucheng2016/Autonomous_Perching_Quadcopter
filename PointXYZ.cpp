#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <vision_proj/Pts.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"
#include <math.h>
ros::Publisher pub;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
void callback(const PointCloud::ConstPtr& msg)
{

int j = 0;
int step = 5;
double dist = 0;
vision_proj::Pts p;
pcl::PointXYZ pt;
for (int i = 0; i < 307200;i=i+step) {
    pt = msg->points[i];
    if (!isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z)){
        double length = 0;
        if ((length = sqrt(pt.x*pt.x+pt.y*pt.y+pt.z*pt.z)) < 1) {
           dist = dist + length;
           p.X[j] = pt.x;
           p.Y[j] = pt.y;
           p.Z[j] = pt.z;
           j = j + 1;
        }
    }
}
p.size = j;
p.dist = dist/p.size;
pub.publish(p);
printf("Data sent\n");

}
int main(int argc, char** argv)
{
   ros::init(argc, argv, "sub_pcl"); 
   ros::NodeHandle nh;
   while (ros::ok()) {
      pub = nh.advertise<vision_proj::Pts>("chatter", 1000);
      ros::Subscriber sub = nh.subscribe<PointCloud>("/camera/depth/points", 1, callback);
      ros::spin();
   }
   //ros::spin();
   
}
