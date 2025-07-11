❯ ros2 topic echo /ublox_gps_node/fix --once
header:
  stamp:
    sec: 1749792711
    nanosec: 499524112
  frame_id: gps
status:
  status: 2
  service: 15
latitude: 37.541383599999996
longitude: 127.0777631
altitude: 39.475
position_covariance:
- 0.00019600000000000002
- 0.0
- 0.0
- 0.0
- 0.00019600000000000002
- 0.0
- 0.0
- 0.0
- 0.000225
position_covariance_type: 2
---


❯ ros2 topic echo /ublox_gps_node/fix_velocity --once
header:
  stamp:
    sec: 1749792687
    nanosec: 499529980
  frame_id: gps
twist:
  twist:
    linear:
      x: -0.34800000000000003
      y: 0.053
      z: -0.03
    angular:
      x: 0.0
      y: 0.0
      z: 0.0
  covariance:
  - 0.007225000000000001
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.007225000000000001
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.007225000000000001
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - -1.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
---


❯ ros2 topic echo /ouster/imu --once
header:
  stamp:
    sec: 1749792713
    nanosec: 728664968
  frame_id: os_imu
orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
orientation_covariance:
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
- -1.0
angular_velocity:
  x: -0.020506340393610132
  y: -0.0660463950339651
  z: -0.027963191445832
angular_velocity_covariance:
- 0.0006
- 0.0
- 0.0
- 0.0
- 0.0006
- 0.0
- 0.0
- 0.0
- 0.0006
linear_acceleration:
  x: 1.6112977172851561
  y: 0.63206923828125
  z: 10.034099157714843
linear_acceleration_covariance:
- 0.01
- 0.0
- 0.0
- 0.0
- 0.01
- 0.0
- 0.0
- 0.0
- 0.01
---

❯ ros2 topic echo /sorted_cones_time --once
header:
  stamp:
    sec: 1749793057
    nanosec: 98647977
  frame_id: os_sensor
class_names:
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
- Unknown
layout:
  dim:
  - label: ''
    size: 7
    stride: 21
  - label: ''
    size: 3
    stride: 3
  data_offset: 0
data:
- 0.1343604028224945
- -1.6442407369613647
- -0.5042098760604858
- 1.0872896909713745
- 3.113274574279785
- -0.7936870455741882
- 1.293522834777832
- -2.201655864715576
- -0.7627513408660889
- 2.700814962387085
- -2.583913564682007
- -0.8693170547485352
- 2.8622167110443115
- 2.4892373085021973
- -0.8418356776237488
- 4.014394283294678
- -2.9940550327301025
- -0.9254414439201355
- 5.456058502197266
- -3.5071258544921875
- -0.8748795390129089
---

❯ ros2 topic echo /fused_sorted_cones --once
header:
  stamp:
    sec: 1749793042
    nanosec: 98935151
  frame_id: os_sensor
class_names:
- Unknown
- yellow cone
- blue cone
- yellow cone
- blue cone
- yellow cone
- blue cone
- blue cone
layout:
  dim:
  - label: cones
    size: 8
    stride: 24
  - label: coords
    size: 3
    stride: 3
  data_offset: 0
data:
- 0.022902514785528183
- 2.8655753135681152
- -0.6460021138191223
- 2.4186251163482666
- -2.4913899898529053
- -0.887528121471405
- 2.8509600162506104
- 2.5904531478881836
- -0.9206307530403137
- 4.058539867401123
- -2.639845132827759
- -0.9830846786499023
- 4.4517316818237305
- 2.4836792945861816
- -1.0379422903060913
- 5.40134859085083
- -2.757723331451416
- -1.0379133224487305
- 5.832907676696777
- 2.3523387908935547
- -0.962537407875061
- 8.812877655029297
- 4.273287296295166
- -1.099255919456482
---

❯ ros2 topic echo /fused_sorted_cones_ukf --once
header:
  stamp:
    sec: 1749793057
    nanosec: 898841259
  frame_id: os_sensor
cones:
- track_id: 45
  position:
    x: 0.5895797268029904
    y: -1.960857692140977
    z: -0.6543798363278905
  color: Yellow cone
- track_id: 67
  position:
    x: 1.9670123567357982
    y: -2.0700727409414337
    z: -0.8354409504780164
  color: Yellow cone
- track_id: 69
  position:
    x: 1.181838693791955
    y: 2.934937572564673
    z: -0.5748203551469969
  color: Blue cone
- track_id: 70
  position:
    x: 3.339280553132416
    y: -2.2343041174638625
    z: -0.915167876698713
  color: Yellow cone
- track_id: 71
  position:
    x: 3.195377304150685
    y: 2.7272257558552586
    z: -0.6442705513336777
  color: Blue cone
- track_id: 76
  position:
    x: 4.8308417127925924
    y: -2.4477710907040917
    z: -0.9561084594162758
  color: Yellow cone
- track_id: 78
  position:
    x: 4.874017310128618
    y: 2.3607512002747786
    z: -0.6651842580401713
  color: Blue cone
- track_id: 80
  position:
    x: 6.28381173791468
    y: -2.7271992458810725
    z: -0.9651274457516422
  color: Yellow cone
- track_id: 81
  position:
    x: 6.351610313736045
    y: 2.2037821487065883
    z: -0.7619725583147273
  color: Blue cone
- track_id: 82
  position:
    x: 8.168710215988067
    y: 1.8017112841397718
    z: -0.7681410928152635
  color: Blue cone
- track_id: 83
  position:
    x: 9.61678280987158
    y: 2.4187664937218973
    z: -0.8012927092851959
  color: Blue cone
- track_id: 84
  position:
    x: 9.118144120158387
    y: -3.1151920203765893
    z: -1.0683895946922062
  color: Yellow cone
- track_id: 85
  position:
    x: 10.828036204327475
    y: 3.142399157072255
    z: -0.9263806538691863
  color: Blue cone
- track_id: 86
  position:
    x: 7.890953551778425
    y: -2.852146509225767
    z: -0.9335141045171924
  color: Yellow cone
---

❯ ros2 interface show custom_interface/msg/ModifiedFloat32MultiArray
std_msgs/Header header
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
string[] class_names
std_msgs/MultiArrayLayout layout
	#
	#
	#
	#
	#
	MultiArrayDimension[] dim #
		string label   #
		uint32 size    #
		uint32 stride  #
	uint32 data_offset        #
float32[] data


❯ ros2 interface show custom_interface/msg/TrackedConeArray
# custom_interface/msg/TrackedConeArray.msg
std_msgs/Header header     # Timestamp and frame_id (os_sensor)
	builtin_interfaces/Time stamp
		int32 sec
		uint32 nanosec
	string frame_id
TrackedCone[] cones # Array of tracked cones
	int32 track_id          #
	geometry_msgs/Point position #
		float64 x
		float64 y
		float64 z
	string color            #
