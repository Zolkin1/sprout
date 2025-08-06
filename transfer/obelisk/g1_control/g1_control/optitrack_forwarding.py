import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MocapRelayNode(Node):
    def __init__(self):
        super().__init__('mocap_relay_node')

        # Subscription to OptiTrack mocap data
        self.subscription = self.create_subscription(
            PoseStamped,
            '/g1/pose',  # <-- change this to your actual input topic
            self.listener_callback,
            10)

        # Publisher to relay the data to the robot
        self.publisher = self.create_publisher(
            PoseStamped,
            '/robot/pose',  # <-- output topic
            10)

        self.get_logger().info('Mocap Relay Node has been started.')

    def listener_callback(self, msg):
        # Simply republish the received message
        self.publisher.publish(msg)
        self.get_logger().debug('Relayed PoseStamped message.')

def main(args=None):
    rclpy.init(args=args)
    node = MocapRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
