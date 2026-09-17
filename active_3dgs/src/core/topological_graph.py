#!/usr/bin/env python3

import rospy
import geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
import networkx as nx
# import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
import tf2_ros
# from std_msgs.msg import String
import json
from networkx.readwrite import json_graph
import pickle


class TopoTree:
    def __init__(self):
        # Initialize graph
        self.graph = nx.Graph()
        self.odom = np.eye(4, 4)
        self.budget = 5000.
        self.path = []
        self.goal_node = None
        self.num_levels = 2
        self.planning_graph = [nx.Graph() for _ in range(self.num_levels)]
        self.plans = [[] for _ in range(self.num_levels)]
        # (x,y) bounds
        self.bounds = [(-4.35, 5.7), (-5.0, 12.0)]

        # Publishers
        self.marker_pub = rospy.Publisher('/high_level_planner',
                                          MarkerArray,
                                          queue_size=10)
        self.sim_ = rospy.get_param("~gs_sim", False)
        if self.sim_:  # for sim
            self.odom_frame_id = rospy.get_param("~odom_frame_id", "odom")
            self.world_frame_id = rospy.get_param("~world_frame_id", "world")
        else:
            self.odom_frame_id = rospy.get_param("~robot_odom_frame_id",
                                                 "odom")
            self.world_frame_id = rospy.get_param("~robot_world_frame_id",
                                                  "world")
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("High Level Planner Node Started")
        # self.graph_listener = rospy.Subscriber("/nx_graph", String, self.graph_sub)
        self.odom_history = None
        self.relevancy_suppression = 3 
        self.planning_freq = 1.0
        self.replan_ctr = 11
        self.max_replan_steps = 5
        self.wire_dist = 10.0
       
        # file = open("/bags/graph.pkl",'rb')
        # self.graph = pickle.load(file)
        # print(self.graph)
        # file.close() 

    def graph_sub(self, json_data):
        # json_data = json.loads(msg.data)
        self.graph = json_graph.adjacency_graph(json_data)
        self.planning_graph = [nx.Graph() for _ in range(self.num_levels)]
        self.plans = [[] for _ in range(self.num_levels)]
        for node, data in self.graph.nodes(data=True):
            for k in data.keys():
                if isinstance(data[k], str) and k != 'submap_id':
                    data[k] = float(data[k])
                if k == 'pos':
                    data[k] = np.array(data[k]).astype(np.float32)
                    # data[k][0] = float(data[k][0])
                    # data[k][1] = float(data[k][1])

            if node[0] == 'r':
                for n, d in self.graph.nodes(data=True):
                    if n == data['submap_id']:
                        d['relevancy'] += data['relevancy']
                        break

    def lookup_odom(self, level):
        # Lookup transform
        try:
            # Lookup the static transform
            source_frame = self.world_frame_id
            target_frame = self.odom_frame_id
            print(source_frame, target_frame)
            transform = self.tf_buffer.lookup_transform(source_frame,
                                                        target_frame,
                                                        rospy.Time(0))
        except tf2_ros.LookupException as e:
            print(e)
            rospy.logerr(f"Transform lookup failed: {e}")
            return -1
        except tf2_ros.ConnectivityException as e:
            print(e)
            rospy.logerr(f"Transform connectivity issue: {e}")
            return -1
        except tf2_ros.ExtrapolationException as e:
            print(e)
            rospy.logerr(f"Transform extrapolation issue: {e}")
            return -1
        self.odom_pos = np.array([transform.transform.translation.x,
                                  transform.transform.translation.y,
                                  transform.transform.translation.z])
        self.odom_yaw = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_euler("xyz")[2]
        rospy.loginfo(f"Odom pos: {self.odom_pos}, Odom yaw: {self.odom_yaw}")
        self.odom[0, 3] = transform.transform.translation.x
        self.odom[1, 3] = transform.transform.translation.y
        self.odom[2, 3] = transform.transform.translation.z
        self.odom[:3, :3] = R.from_quat([transform.transform.rotation.x,
                                         transform.transform.rotation.y,
                                         transform.transform.rotation.z,
                                         transform.transform.rotation.w]).as_matrix()
        self.planning_graph[level].add_node('x0',
                            pos=(transform.transform.translation.x,
                                 transform.transform.translation.y),
                            predicted=False,
                            utility=0.,
                            frontier=False,
                            relevancy=0.,
                            submap_id='x0')
        if level == 0:
            if self.odom_history is None:
                self.odom_history = self.odom[:2, 3]
            else:
                self.odom_history = np.vstack((self.odom_history, self.odom[:2, 3]))
                #print("History: ", self.odom_history)

    def cost(self, i, j, level):
        return np.linalg.norm(np.asarray(self.planning_graph[level].nodes[i]['pos'])
                              - np.asarray(self.planning_graph[level].nodes[j]['pos']))

    def make_graphs(self):
        for i in range(self.num_levels):
            self.planning_graph[i] = nx.Graph()
            l_counter = 0
            nodes = [node for node in self.graph if node[0] == 's']
            last = len(nodes)
            for node, data in self.graph.nodes(data=True):
                if self.odom_history is not None:
                    _dist = np.linalg.norm(np.array(data['pos']).reshape(2,) - self.odom_history.reshape((-1, 2)), axis=1)
                    if np.min(_dist) < self.relevancy_suppression:
                        data['relevancy'] = 0.
                        continue
                if node[0] == 's' and i == 0:
                    if node == 's'+str(last):
                        self.planning_graph[i].add_node(node, pos=data['pos'],
                                                    utility=data['relevancy'])
                    else:
                        self.planning_graph[i].add_node(node, pos=data['pos'],
                                                    utility=data['relevancy'])


                    #print(l_counter)
                    l_counter += 1
                if (node[0] == 'r') and i == 1:

                    self.planning_graph[i].add_node(node, pos=data['pos'],
                                                    utility=data['relevancy'],
                                                    submap_id=data['submap_id'])
                if (node[0] == 'l') and i == 1:
                    self.planning_graph[i].add_node(node, pos=data['pos'],
                                                    utility=data['relevancy'],
                                                    submap_id='x0')
                    l_counter +=1
                # if node[0] == 'p' and l_counter == 0:
                #     self.planning_graph[i].add_node(node, pos=data['pos'],
                #                                     utility=0.1,
                #                                     submap_id=node)
            self.wire(i)
            ret = self.lookup_odom(i)
            if ret == -1:
                return -1
            self.wire_closest(i)

    def plan_level(self, level, valid):
        # print("Level: ", level)
        # print("Graph: ")
        # for n, d in self.planning_graph[level].nodes(data=True):
        #     print(n, d)
        # for e in self.planning_graph[level].edges:
        #     print("Edge: ", level, e)
        _, u, p, _ = self.dijkstra('x0', level, valid)
        ix = max(u, key=u.get)
        # print(u, p)
        valid = p[ix]
        return valid

    def plan(self):
        self.path = []
        path = None
        for i in range(self.num_levels):
            if len(self.planning_graph[i].nodes) < 1:
                break
            path = self.plan_level(i, path)
            self.path = []
            for n in path:
                pos = self.planning_graph[i].nodes[n]['pos']
                self.path.append([ pos[0], pos[1] ])
            # self.path = np.array(self.path)
            self.plans[i] = path
        print("Plans: ", self.plans)
        print(self.path)
        with open('graph.pkl', 'wb') as f:
            pickle.dump(self.planning_graph[0], f)
        return path

    def spin(self, json_data):
        # print("Spinning.....")
        ret = self.lookup_odom(0)
        if ret == -1:
            return np.array(self.path)
        # print(self.odom, self.path)
        if self.replan_ctr <= self.max_replan_steps:
            self.replan_ctr += 1
            # print("Skipping replan due to iterations")
            return np.array(self.path)
        else:
            # if len(self.path) > 0:
                # print(np.linalg.norm(self.odom[:2, 3] - np.array(self.path[-1])))
                # if np.linalg.norm(self.odom[:2, 3] - np.array(self.path[-1])) > 1.0:
                #     print("Skipping replan due to goal tol")
                #     self.replan_ctr += 1
                #     return self.path
            self.graph_sub(json_data)
            ret = self.make_graphs()
            if ret == -1:
                return np.array(self.path)
            # print("Planning.....")
            self.plan()
            # print("Graph ----- ")
            # for node, data in self.planning_graph[0].nodes(data=True):
            #     print(node, data)
            self.publish_graph()
            self.replan_ctr = 0
            print("Returned path: ",self.path)
            return np.array(self.path)

    def wire_closest(self, level):
        nodes = list(self.planning_graph[level].nodes)
        # if len(nodes) == 1:
        #    return
        cost = np.inf
        closest = nodes[0]
        # print("Wiring closest at ", level, " cost: ", nodes)
        for n in nodes:
            if n == 'x0':
                continue
            _cost = self.cost('x0', n, level)
            if cost > _cost:
                cost = _cost
                closest = n
        # print("--------------------- Wiring x0 to ", closest)
        self.planning_graph[level].add_edge('x0', closest)

    def wire(self, level):
        if level == 0:
            for d1 in self.planning_graph[level].nodes:
                for d2 in self.planning_graph[level].nodes:
                    if d1 != d2:
                        c = self.cost(d1, d2, 0)
                        if c <= self.wire_dist:# or d1[0] != 'p' or d2[0] != 'p':
                            self.planning_graph[level].add_edge(d1, d2)
        else:
            cluster_ids = []
            for _, d in self.planning_graph[level].nodes(data=True):
                cluster_ids.append(d['submap_id'])
            cluster_ids = list(set(cluster_ids))
            closest_ids = {}
            for i in cluster_ids:
                closest_ids[i] = {} #[j] = [0, 0, np.inf]
            
            for j in cluster_ids:
                for i in cluster_ids:
                    closest_ids[i][j] = [0, 0, np.inf]
            # print( closest_ids)
            for _ in range(len(closest_ids)):
                for n1, d1 in self.planning_graph[level].nodes(data=True):
                    for n2, d2 in self.planning_graph[level].nodes(data=True):
                        if n1 == n2:
                            continue
                        # print(n1, n2, d1, d2)
                        sub_id1 = d1['submap_id']
                        sub_id2 = d2['submap_id']
                        if sub_id1 != sub_id2 and sub_id1 != 'x0' and sub_id2 != 'x0':
                            c = self.cost(n1,
                                      n2,
                                      level)
                            # print(c, closest_ids)
                            if c < closest_ids[sub_id1][sub_id2][2]:
                                closest_ids[sub_id1][sub_id2][0] = n1
                                closest_ids[sub_id1][sub_id2][1] = n2
                                closest_ids[sub_id1][sub_id2][2] = c

            for i in cluster_ids:
                for j in cluster_ids:
                    if i != j and i != 'x0' and j != 'x0':
                        # print(cluster_ids, i, j, closest_ids)
                        ids = closest_ids[i][j]
                        self.planning_graph[level].add_edge(ids[0], ids[1])
            
            for d1 in self.planning_graph[level].nodes:
                for d2 in self.planning_graph[level].nodes:
                    if d1 == "x0" or d2 == 'x0':
                        continue
                    if d1 != d2:
                        c = self.cost(d1, d2, level)
                        if c <= self.wire_dist:# or d1[0] != 'p' or d2[0] != 'p':
                            self.planning_graph[level].add_edge(d1, d2)

    def dijkstra(self, start, level, valid):
        distances = {node: 0. for node in self.planning_graph[level]}
        utilities = {node: 0. for node in self.planning_graph[level]}
        cost_benefit = {node: 0 for node in self.planning_graph[level]}
        paths = {node: [start] for node in self.planning_graph[level]}
        distances[start] = 0.
        queue = [(0, 0, start)]
        while queue:
            current_utility, current_distance, current_node = heapq.heappop(queue)
            if valid is not None:
                # print(self.planning_graph[level].nodes[current_node], current_node)
                if self.planning_graph[level].nodes[current_node]['submap_id'] not in valid:
                    # print("Rejecting: ", current_node)
                    continue
            for neighbor, attr in self.planning_graph[level][current_node].items():
                weight = self.cost(current_node, neighbor, level)
                distance = current_distance + weight
                if neighbor in paths[current_node]:
                    continue
                else:
                    utility = -current_utility + \
                        self.planning_graph[level].nodes[neighbor]['utility']
                if distance >= self.budget:
                    continue
                if utility >= utilities[neighbor]:
                    utilities[neighbor] = utility
                    distances[neighbor] = distance
                    if distance > 0.:
                        cost_benefit[neighbor] = utility / np.exp(distance)
                    paths[neighbor] = paths[current_node] + [neighbor]
                    heapq.heappush(queue, (-utility, distance, neighbor))
        return distances, utilities, paths, cost_benefit

    def safeget(self, d, k, n):
        if k in d.keys():
            return d[k]
        else:
            # print(k, " not in ", d.keys(), "for node ", n)
            return False

    def safedist(self, d, k, n):
        if k in d.keys():
            return d[k]
        else:
            return np.asarray([np.inf, np.inf])

    def print_graph(self):
        print("---------------------------------------------")
        for node, data in self.graph.nodes(data=True):
            print(node, data)

        print(self.graph.edges.data())
        print("||---------------------------------------------||")

    def publish_graph(self):
        marker_array = MarkerArray()
        # print("world frame id is: ", self.world_frame_id)
        if self.world_frame_id is None:
            return
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.ns = 'graph_nodes'
        marker.header.frame_id = self.world_frame_id
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        self.marker_pub.publish(marker_array_msg)
        # Add nodes as spheres
        for i in range(self.num_levels):
            for node, data in self.planning_graph[i].nodes(data=True):
                try:
                    marker = Marker()
                    marker.header.frame_id = self.world_frame_id
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "graph_nodes"
                    marker.id = int(node[1:])
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = float(data['pos'][0])
                    marker.pose.position.y = float(data['pos'][1])
                    marker.pose.position.z = self.num_levels - 1 - i
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    marker.color.a = 1.0
                    if node[0] == 's':
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                    elif node[0] == 'l':
                        marker.color.r = 0.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                    elif node[0] == 'r':
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    elif node[0] == 'x':
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0
                    else:
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    marker_array.markers.append(marker)
                except Exception as e:
                    rospy.loginfo("Exception {} \
                            in publishing the markers".format(e))

            for edge in self.planning_graph[i].edges:
                try:
                    marker = Marker()
                    marker.header.frame_id = self.world_frame_id
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "graph_edges"
                    marker.id = len(self.planning_graph[i].nodes) + \
                        list(self.planning_graph[i].edges).index(edge)
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.scale.x = 0.05

                    marker.color.a = 0.1 if (edge[0] in self.plans[i] and edge[1] in self.plans[i]) else 1.0
                    marker.color.r = 1.0 if i == 0 else 0
                    marker.color.g = 0.0
                    marker.color.b = 1.0

                    p1 = self.planning_graph[i].nodes[edge[0]]['pos']
                    p2 = self.planning_graph[i].nodes[edge[1]]['pos']
                    marker.points = [self.create_point(p1, self.num_levels - i - 1),
                                     self.create_point(p2, self.num_levels - i - 1)]
                    marker_array.markers.append(marker)
                except Exception as e:
                    rospy.loginfo("Exception {} \
                            in publishing the markers".format(e))

        self.marker_pub.publish(marker_array)

    def create_point(self, pos, level):
        point = geometry_msgs.msg.Point()
        point.x = pos[0]
        point.y = pos[1]
        point.z = level
        return point


if __name__ == '__main__':
    try:
        TopoTree()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
