
import numpy as np
from sklearn.neighbors import BallTree
import os.path
import pickle
import struct
from typing import Dict, List, Annotated
import shutil
from memory_profiler import memory_usage

MAX_CLUSTER = 200
CLUSTER_DIMENSION = 70
MAX_CLUSTER_SECOND = 200
MAX_CLUSTER_THIRD = 500

############################################  final 12/10 : def cluste()    #############################################
def store_disk(data,path):
  #NOTE: data is list of lists not only a list
  # with open(path, "ab") as fout:  # Use "ab" for appending in binary mode
  with open(path, "ab") as fout:  # Use "ab" for appending in binary mode
    for stored_list in data:
      # Convert the ID and embedding to binary format
      binary_data = struct.pack(f'{len(stored_list)}f', *stored_list)  # 'f' represents a float (4 bytes)
      fout.write(binary_data)

def store_disk_id(int_list , file_name):
  # Convert the list of integers to a binary string
  binary_data = struct.pack(f'{len(int_list)}i', *int_list)
  # Write the binary data to the file
  with open(file_name, "wb") as fout:  # Use "wb" for writing in binary mode
    fout.write(binary_data)

# initial is in global class, or saved in file
# initialClusters is a dict, key is all first level clusters, each value is the second level cluster for each first level cluster
# isInitialled is true if the first level clusters is build.
# secondClusters is a dictionary with key = (first_level,second_level) and value = list of all ACTUAL IDS of the MAX_CLUSTER_THIRD inside this cluster
# secondClusters size has to be = MAX_CLUSTER * MAX_CLUSTER_SECOND (for ex:200*200)
# firstClustersDistances is first_level as the key and (70*values)-centroid- as the valule
# secondClustersDistances is a dict of (first_level,second_level) as the key and (70*values)-the centroid- as the valule
def cluster_data(data, initialClusters, secondClusters, firstClustersDistances, secondClustersDistances, thirdDistances, pre_path):
  lastData = 0
  dropped_data = []
  i = 0
  if len(initialClusters) <  MAX_CLUSTER:
    for i in range(len(initialClusters), len(data)):
      if lastData == MAX_CLUSTER: 
        break
      if i > len(data) - 1:
        # ball_tree = BallTree(initialClusters)
        initialClusters_dictances =list( firstClustersDistances.values() )
        ball_tree = BallTree(initialClusters_dictances)
        with open(os.path.join(pre_path,'ball_tree_primary.pkl') , 'wb') as f:
          pickle.dump(ball_tree, f)
          # remove it from the RAM
          del ball_tree

        return 

      else:
        if i > 0:
          initialClusters_dictances =list( firstClustersDistances.values() )
          ball_tree_tempPP = BallTree(initialClusters_dictances)
          query_point_PP = np.reshape(data[i][1:],(1,CLUSTER_DIMENSION))
          distances, indices = ball_tree_tempPP.query(query_point_PP, k = 1)
          # distPP = calc_distPP(data[i][1:] , firstClustersDistances)
          distPP = distances[0][0]
          # print("distPP",distPP)
          if  distPP < 2.8 : #3.05: #3 : #5.12 : #(2**3): #(2**32): #(2**63)*(0.01):
            dropped_data.append(data[i])
            continue
        initialClusters[lastData] = [data[i][0]]
        firstClustersDistances[lastData] = data[i][1:]
        #######
        secondClustersDistances[(lastData,0)] = firstClustersDistances[lastData]
        secondClusters[(lastData,0)]  = [data[i][0]] # <= m.s of that
        thirdDistances[(lastData,0)]  = [data[i][1:]] # <= m.s of that
        #####
        lastData += 1

    initialClusters_dictances =list( firstClustersDistances.values() )
    ball_tree = BallTree(initialClusters_dictances)
    with open(os.path.join(pre_path,'ball_tree_primary.pkl' ), 'wb') as f:
      pickle.dump(ball_tree, f)
      # remove it from the RAM
      # del ball_tree
  else:
    with open(os.path.join(pre_path,'ball_tree_primary.pkl'), 'rb') as f:
      ball_tree = pickle.load(f)


  # later and important what is the range
  all_drop_other = dropped_data + data[i:]
  for data_item_index in range(len(all_drop_other)):
    # should we change it to the avg or take for example the best 10 only
    with open(os.path.join(pre_path,'ball_tree_primary.pkl'), 'rb') as f:
      ball_tree = pickle.load(f)
    query_point = np.reshape(all_drop_other[data_item_index][1:],(1,CLUSTER_DIMENSION))
    distances, indices = ball_tree.query(query_point, k = MAX_CLUSTER)

    for first in range(0, MAX_CLUSTER):
      # current_nn = indices[first][first]
      current_nn = indices[0][first]
      list1 = initialClusters[current_nn]
      # this is "if" not "else if"
      if len(list1) <  MAX_CLUSTER_SECOND:
        # initially the id of the first level cluster and second level will be the actuall , till we change them
        list1.append(all_drop_other[data_item_index][0])

        secondClustersDistances[(current_nn,len(list1)-1)] = all_drop_other[data_item_index][1:] #NOTE: len(list1)-1 not len(list1):as we did it after list1.append -m.s
        # secondClusters[(current_nn,len(list1)-1)]  = []
        secondClusters[(current_nn,len(list1)-1)]  = [all_drop_other[data_item_index][0]]
        thirdDistances[(current_nn,len(list1)-1)]  = [all_drop_other[data_item_index][1:]]
        #later + important : change the value of the item of firstClustersDistances[current_nn] to be "new_clustered"(new mean)
        # firstClustersDistances[current_nn] = all_drop_other[data_item_index][1:]
        old_first_center = firstClustersDistances[current_nn]
        # old_first_center_numerator = old_first_center*(len(list1)-1)
        old_first_center_numerator = [i *(len(list1)-1) for i in old_first_center]
        new_first_center = [sum(x) for x in zip(old_first_center_numerator, all_drop_other[data_item_index][1:])]
        firstClustersDistances[current_nn] = [i /(len(list1)) for i in new_first_center] #NOTE: len(list1) not len(list1)-1
        # rebuild using the updated distance
        initialClusters_dictances =list( firstClustersDistances.values() )
        ball_tree = BallTree(initialClusters_dictances)
        with open(os.path.join(pre_path,'ball_tree_primary.pkl') , 'wb') as f:
          pickle.dump(ball_tree, f)


        if len(list1) == MAX_CLUSTER_SECOND:
          # indices_distances = [np.reshape(value,(1,CLUSTER_DIMENSION)) for key, value in secondClustersDistances.items() if key[0] == int(current_nn)]
          indices_distances = [value for key, value in secondClustersDistances.items() if key[0] == int(current_nn)]
          ball_tree_second = BallTree(indices_distances)
          # path_second = 'ball_tree'+ str(current_nn) +'.pkl'
          path_second = os.path.join(pre_path,('ball_tree'+ str(current_nn) +'.pkl'))
          with open(path_second, 'wb') as f:
            pickle.dump(ball_tree_second, f)
            # remove it from the RAM
            # del ball_tree_second
        break


      else:
        path_second = os.path.join(pre_path, ('ball_tree'+ str(current_nn) +'.pkl'))
        with open(path_second, 'rb') as f:
          ball_tree_second = pickle.load(f)
        distances_second, indices_second = ball_tree_second.query(query_point, k = MAX_CLUSTER_SECOND)

        breking = False
        for i in range(0, MAX_CLUSTER_SECOND): # <= wla a5leha range(0, len(indices_second[0]))
          list2 = secondClusters[(current_nn,indices_second[0][i])]
          listDist = thirdDistances[(current_nn,indices_second[0][i])]

          # this is "if" not "else if"
          if len(list2) <  MAX_CLUSTER_THIRD:
            #list2 will have the ACTUAL IDS f3ln
            list2.append(all_drop_other[data_item_index][0])
            listDist.append(all_drop_other[data_item_index][1:])
            breking = True

            # #########   update secondClustersDistances   ##########
            # #later + important : change the value of the item of secondClustersDistances[(current_nn,first)] to be "new_clustered"(new mean)
            # # econdClustersDistances[(current_nn,i)] = all_drop_other[data_item_index][1:]
            # # secondClustersDistances[(current_nn,indices_second[0][i])] = all_drop_other[data_item_index][1:]
            # old_center = secondClustersDistances[(current_nn,indices_second[0][i])]
            # # old_center_numerator = old_center*(len(list2)-1)
            # old_center_numerator = [j *(len(list2)-1) for j in old_center]
            # new_center = [sum(x) for x in zip(old_center_numerator, all_drop_other[data_item_index][1:] )]
            # secondClustersDistances[(current_nn,indices_second[0][i])] = [iii /(len(list2)) for iii in new_center] #NOTE: len(list2) not len(list2)-1
            # # rebuild using the updated distance
            # indices_distances = [value for key, value in secondClustersDistances.items() if key[0] == int(current_nn)]
            # ball_tree_second = BallTree(indices_distances)
            # # path_second = 'ball_tree'+ str(current_nn) +'.pkl'
            # path_second = os.path.join(pre_path, ('ball_tree'+ str(current_nn) +'.pkl'))
            # with open(path_second, 'wb') as f:
            #   pickle.dump(ball_tree_second, f)


            # #########   update firstClustersDistances   ##########
            # # old_primary_center = firstClustersDistances[current_nn]*len(list1) #later: m.s:len(list1)
            # old_primary_center = firstClustersDistances[current_nn]
            # old_primary_center_num = [jj *len(list1) for jj in old_primary_center]
            # # old_primary_center_num = [sub(x) for x in zip(old_primary_center_num, old_center)]
            # zipped = [*zip(old_primary_center_num, old_center)]
            # old_primary_center_num = [(x-y) for (x,y) in zipped]
            # old_primary_center_num = [sum(x) for x in zip(secondClustersDistances[(current_nn,indices_second[0][i])], old_primary_center_num)] #add the new centroid
            # firstClustersDistances[current_nn] = [ll /len(list1) for ll in old_primary_center_num]
            # # rebuild using the updated distance
            # initialClusters_dictances =list( firstClustersDistances.values() )
            # ball_tree = BallTree(initialClusters_dictances)
            # with open(os.path.join(pre_path, ('ball_tree_primary.pkl' )), 'wb') as f:
            #   pickle.dump(ball_tree, f)

            break

        if breking == True:
          break


  kk = -1
  for initialClusters_item in initialClusters.items():
    kk += 1
    key_id , value_list = initialClusters_item
    # if(len(value_list) < MAX_CLUSTER_SECOND )
    if(len(value_list) > 0 and len(value_list) < MAX_CLUSTER_SECOND ):
      indices_distances = [value for key, value in secondClustersDistances.items() if key[0] == kk ]
      ball_tree_second = BallTree(indices_distances)
      # path_second = 'ball_tree'+ str(kk) +'.pkl'
      path_second = os.path.join(pre_path, ('ball_tree'+ str(kk) +'.pkl'))
      with open(path_second, 'wb') as f:
        pickle.dump(ball_tree_second, f)
        # remove it from the RAM
        del ball_tree_second

  for secondClusters_item in secondClusters.items():
    # list2 = secondClusters[(current_nn,indices_second[0][i])]
    key , val = secondClusters_item
    #if there is any item in the cluster
    if len(val) > 0 :
      list_id = val
      primary_level = key[0]
      secondry_level = key[1]
      # path_second_ids = 'ID' + str(primary_level) + '_' + str(secondry_level)+'.bin'
      path_second_ids = os.path.join(pre_path, ('ID' + str(primary_level) + '_' + str(secondry_level)+'.bin'))
      store_disk_id(list_id,path_second_ids)


  # allIDs_Third = 0
  for thirdDistances_item in thirdDistances.items():
    key , val = thirdDistances_item
    #if there is any item in the cluster
    # allIDs_Third += len(val)
    if len(val) > 0 :
      list_dist = val
      primary_level = key[0]
      secondry_level = key[1]
      path_second = os.path.join(pre_path, (str(primary_level) + '_' + str(secondry_level)+'.bin'))
      store_disk(list_dist,path_second)

  return 

##########################################################
def calculate_distances(records, given_record):
    distances = []

    for record in records:
        # Calculate Euclidean distance between each record and the given record
        distance = np.linalg.norm(np.array(record) - np.array(given_record))
        distances.append(distance)
    return distances

def calc_2(records, given_record):
    all_cal_dist = records.dot(given_record.T).T / (np.linalg.norm(records, axis=1) * np.linalg.norm(given_record))
    return all_cal_dist

def Leaf_Brute_Force(records, given_record):
    all_dist = calculate_distances(records, given_record)
    dist_dict = {i: row for i, row in enumerate(all_dist)}
    # temp_dict = {k: v for k, v in zip(flatten_ids,flatten_dist)}
    sorty = dict(sorted(dist_dict.items(), key=lambda x: x[1]))
    ret_indices = list(sorty.keys())
    ret_distances = list(sorty.values())
    return ret_distances,ret_indices

def _cal_score( vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
  return cosine_similarity

def retrive22(dataset , query , top_k = 5):
  scores = []
  # with open(file_path, "r") as fin:
  i = 0
  for row in dataset:
    score = _cal_score(query, row)
    scores.append((score, i))
    i+= 1
  # here we assume that if two rows have the same score, return the lowest ID
  scores = sorted(scores, reverse=True)[:top_k]
  allIds = [s[1] for s in scores]
  allDist = [s[0][0] for s in scores]
  return allDist,allIds


def read_bin(path):
  lists = []
  with open(path, "rb") as fin:  # Use "rb" for reading in binary mode
    binary_data = fin.read()

    # Unpack the binary data into a list of floats
    num_elements = len(binary_data) // struct.calcsize('f')
    unpacked_data = struct.unpack(f'{num_elements}f', binary_data)

  unpacked_data = list(unpacked_data)
  num_of_lists = int(len(unpacked_data)/CLUSTER_DIMENSION)
  lists = np.array(unpacked_data).reshape((num_of_lists,CLUSTER_DIMENSION))
  return lists

def read_bin_id(file_name):
  ## Read the binary data from the file
  with open(file_name, "rb") as fin:  # Use "rb" for reading in binary mode
    binary_data = fin.read()
  # Unpack the binary data into a list of integers
  num_elements = len(binary_data) // struct.calcsize('i')
  unpacked_data = struct.unpack(f'{num_elements}i', binary_data)
  return list(unpacked_data)

def append_or_extend(container, item, req3):
  if hasattr(item, '__iter__'):  # Check if item is iterable
      container.extend(item)
      req3 += len(item)
  else:
      container.append([item])
      req3 += 1

def team_search(query_node,TOP_K , initialClusters , pre_path):
  # print("pre_path for seach ", pre_path)
  ef = 25 #10 # 10
  # path = './ball_tree_primary.pkl'
  path = os.path.join(pre_path, './ball_tree_primary.pkl')
  check_file = os.path.isfile(path)
  # allNodes is the array of actuall ids
  allNodesActIds = []
  allNodesdist = []
  if check_file:
    with open(path, 'rb') as f:
      ball_tree = pickle.load(f)
    tree_array = ball_tree.get_arrays()
    primary_len = len(tree_array[1])

    if primary_len <= TOP_K :
      # allNodes = [i for i in range(0,primary_len)]
      allNodesActIds.append( [i for i in range(0,primary_len)] )
      return allNodesActIds #//uncommet it later

    # req_nodes =min(primary_len,TOP_K .. later
    distances_111 , indices_111  = ball_tree.query(query_node, k = min(ef**3,primary_len))

    req2 = 0
    for nnIndex_1 in range(len(indices_111[0])):
      if req2 >= ef**3 : # m.s later
        break
      current_nn_111 = indices_111[0][nnIndex_1]
      # path_second = 'ball_tree'+ str(current_nn_111) +'.pkl'
      path_second = os.path.join(pre_path, ('ball_tree'+ str(current_nn_111) +'.pkl'))
      check_file_second = os.path.isfile(path_second)
      if check_file_second:
        with open(path_second, 'rb') as f:
          ball_tree_second = pickle.load(f)
        list222 = initialClusters[current_nn_111]
        distances_222 , indices_222  = ball_tree_second.query(query_node, k = min((ef**2), len(list222)))
        req3 = 0
        for nnIndex_2 in range(len(indices_222[0])):
          if req3 >= ef**2 : # m.s later
            break
          current_nn_222 = indices_222[0][nnIndex_2]
          # path_third = str(current_nn_111) + '_' + str(current_nn_222)+'.bin'
          # path_third_ids = 'ID' + str(current_nn_111) + '_' + str(current_nn_222)+'.bin'
          path_third = os.path.join(pre_path, (str(current_nn_111) + '_' + str(current_nn_222)+'.bin'))
          path_third_ids = os.path.join(pre_path, ('ID' + str(current_nn_111) + '_' + str(current_nn_222)+'.bin'))
          check_file_third = os.path.isfile(path_third)
          check_file_third_ids = os.path.isfile(path_third_ids)

          if check_file_third and check_file_third_ids:
            ThirdDist = read_bin(path_third)
            list_actual = read_bin_id(path_third_ids)
            # print('ThirdDist in serach ',ThirdDist , 'for key = ',(current_nn_111,current_nn_222))
            #to deal with if we want ef nodes but the leaf has less than ef nodes
            take3 = min(ef,len(list_actual))
            dist_third_sort , ind_third_sort = retrive22(ThirdDist, query_node, top_k= take3)
            dist_third_sort = dist_third_sort[:take3]
            ind_third_sort = ind_third_sort[:take3]
            final_actual = [list_actual[least] for least in ind_third_sort]
            allNodesActIds.append(final_actual)
            allNodesdist.append(dist_third_sort)
            req3 += take3
          # if there is no tree, then this cluster "(current_nn_111,current_nn_222)" of the secondry level has only one node
          # then return it
          else:
            print("a primary_secondry not exist ----------")
            temp_list = initialClusters[current_nn_111]
            append_or_extend(allNodesActIds, temp_list[current_nn_222],req3)
            # break

        req2 += req3 # add "req3" (which the nodes collected from this cluster "current_nn_111")
      # if there is no tree, then this cluster "current_nn_111" of the primary level has only one node
      # then return it
      else:
        allNodesActIds.append(initialClusters[current_nn_111])
        req2 += len(initialClusters[current_nn_111])
  else:
    print(' "ball_tree_primary.pkl" is not exist' )

  flatten_ids = sum(allNodesActIds, [])
  flatten_dist = sum(allNodesdist, [])
  temp_dict = {k: v for k, v in zip(flatten_ids,flatten_dist)}
  sorty = dict(sorted(temp_dict.items(), key=lambda x: x[1], reverse=True))
  sorty_keys = list(sorty.keys())[:TOP_K]
  # print('my ids = ' , sorty_keys )
  return sorty_keys

#############################################
class VecDB:
  # def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
  def __init__(self, file_path = '0'  , new_db = True) -> None:
    self.last_folder = file_path
    self.file_initialClusters = 'initialClusters.bin'
    self.file_secondClusters = 'secondClusters.bin'
    self.file_firstClustersDistances = 'firstClustersDistances.bin'
    self.file_secondClustersDistances = 'secondClustersDistances.bin'
    self.file_thirdDistances = 'thirdDistances.bin'
    # self.last_len = 0
    self.DBfile_path = "saved_db.csv"

    if new_db: # <= later: remove this condition ?
      self.create_folder()
      self.create_empty_file(os.path.join(self.last_folder, self.file_initialClusters))
      self.create_empty_file(os.path.join(self.last_folder, self.file_secondClusters))
      self.create_empty_file(os.path.join(self.last_folder, self.file_firstClustersDistances))
      self.create_empty_file(os.path.join(self.last_folder, self.file_secondClustersDistances))
      self.create_empty_file(os.path.join(self.last_folder, self.file_thirdDistances))

      # storing the original database
      self.create_empty_file(os.path.join(self.last_folder, self.DBfile_path))




  def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
    # records_np_recovery = np.array([list(rec.values())[1] for rec in rows])
    records_np_recovery = [[list(rec.values())[0],*list(rec.values())[1]] for rec in rows]
    # make a copy from last folder
    new_folder = str( len(records_np_recovery) + int(self.last_folder) )
    self.copy_folder(new_folder)
    # print('after copyyyyy',self.last_folder)

    # ab+: Opens a file for both appending and reading in binary mode.
    with open(os.path.join(self.last_folder, self.DBfile_path), "ab+") as fout:  # Use "ab" for appending in binary mode
      for row in rows:
        id, embed = row["id"], row["embed"]
        # Convert the ID and embedding to binary format
        id_binary = struct.pack('I', id)  # Assuming 'I' represents an unsigned int (4 bytes)
        embed_binary = struct.pack(f'{len(embed)}f', *embed)  # 'f' represents a float (4 bytes)
        # Concatenate the binary data and write it to the file
        fout.write(id_binary + embed_binary)
    # self._build_index()
    del rows
    self._build_index(records_np_recovery)

  def _build_index(self,data):

    # read the old info
    initialClusters = self.read_dict(os.path.join(self.last_folder, self.file_initialClusters))
    secondClusters = self.read_dict(os.path.join(self.last_folder, self.file_secondClusters))
    firstClustersDistances = self.read_dict(os.path.join(self.last_folder, self.file_firstClustersDistances))
    secondClustersDistances  = self.read_dict(os.path.join(self.last_folder, self.file_secondClustersDistances))
    thirdDistances = self.read_dict(os.path.join(self.last_folder, self.file_thirdDistances))

    pre_path = self.last_folder
    cluster_data(data, initialClusters, secondClusters, firstClustersDistances, secondClustersDistances,thirdDistances,pre_path)

    del data
    #### write the new info
    self.store_dict(os.path.join(self.last_folder, self.file_initialClusters) , initialClusters)
    self.store_dict(os.path.join(self.last_folder, self.file_secondClusters) , secondClusters)
    self.store_dict(os.path.join(self.last_folder, self.file_firstClustersDistances) , firstClustersDistances)
    self.store_dict(os.path.join(self.last_folder, self.file_secondClustersDistances) , secondClustersDistances)
    # self.store_dict(os.path.join(self.last_folder, self.file_thirdDistances) , thirdDistances)
    thirdDistances = {key: [] for key in thirdDistances}
    self.store_dict(os.path.join(self.last_folder, self.file_thirdDistances) , thirdDistances)

    return

  def retrive(self, query , top_k):
    pre_path = self.last_folder
    initialClusters = self.read_dict(os.path.join(pre_path , self.file_initialClusters))
    query_trancate = query.astype(np.float32)
    ids_result = team_search(query_trancate ,top_k , initialClusters, pre_path)
    return ids_result

#   def debug_results(self,actual_ids):
#     for item in self.secondClusters.items():
#       key , val = item
#       for oneVal in val:
#         if oneVal in actual_ids:
#           print(key)

#   def print_all(self):
#     print(len(self.secondClusters))
#     print('data' , self.data)

#################### file system folders functions ##################
  def create_empty_file(self,file_path):
    with open(file_path, 'w'):
        pass

  def create_folder(self):
    # os.makedirs(self.last_folder)
    os.makedirs(self.last_folder, exist_ok=True) #<= later: check "exist_ok"

  def copy_folder(self,new_folder_path):
    if os.path.exists(self.last_folder): # and !os.path.exists(new_folder_path):
        shutil.copytree(self.last_folder, new_folder_path)
        # print('copy : ',self.last_folder,'to ', new_folder_path)
        self.last_folder = new_folder_path
        return new_folder_path
    # else:
    #     os.makedirs(folder_path, exist_ok=True)
    #     return folder_path

  # def store_file(self,data, file_path):
  #     file_path = os.path.join(self.last_folder, file_path)
  #     print('now storing to ',file_path)
  #     with open(file_path, 'w') as file:
  #         file.write(data)

  def read_dict(self,file_path):
    # with open(file_path, 'r') as file:
    #   data = json.load(file)
    loaded_data = {}
    if os.path.getsize(file_path) > 0:
      with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

  def store_dict(self,file_path,data):
    # NOTE: the mode here is 'w' as we want to override the old one
    # Store the dictionary in binary format
    with open(file_path, 'wb') as file:
      pickle.dump(data, file)

  # def store_dict_Third(self,file_path,data):
  #   # NOTE: the mode here is 'w' as we want to override the old one
  #   # Store the dictionary in binary format
  #   with open(file_path, 'wb') as file:
  #     pickle.dump(data, file)

# ###############################  main #######################
# if __name__ == "__main__":
#     db = VecDB()
#     rng = np.random.default_rng(50)
#     ########################### 10k  ###########################
#     records_np = rng.random((10000, 70), dtype=np.float32)
#     records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
#     _len = len(records_np)
#     del records_np
#     db.insert_records(records_dict)
#     # res = run_queries_old(db, records_np, 5, 10)
#     # print(evaluate_result(res))

#     ########################### 100k  ###########################
#     records_np_new = rng.random((90000, 70), dtype=np.float32)
#     # records_np = np.concatenate([records_np, records_np_new])
#     # del records_np_new
#     # rng = np.random.default_rng(50)
#     # records_np = rng.random((100000, 70), dtype=np.float32)
#     records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np_new)]
#     _len += len(records_np_new)
#     del records_np_new
#     db.insert_records(records_dict)
#     # res = run_queries_old(db, records_np, 5, 10)
#     # print(evaluate_result(res))

#     # ########################### 1M  ###########################
#     # records_np_new = rng.random((900000, 70), dtype=np.float32)
#     # # records_np = np.concatenate([records_np, records_np_new])
#     # # del records_np_new
#     # # rng = np.random.default_rng(50)
#     # # records_np = rng.random((1000000, 70), dtype=np.float32)
#     # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np_new)]
#     # _len += len(records_np_new)
#     # del records_np_new
#     # db.insert_records(records_dict)
#     # # res = run_queries_old(db, records_np, 5, 10)
#     # # print(evaluate_result(res))

#     # ########################### 5M + 10M + 15M + 20M  ###########################
#     # for _ in range(19):
#     #     records_np_new = rng.random((1000000, 70), dtype=np.float32)
#     #     # records_np = np.concatenate([records_np, records_np_new])
#     #     # del records_np_new
#     #     # rng = np.random.default_rng(50)
#     #     # records_np = rng.random((3000000, 70), dtype=np.float32)
#     #     records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np_new)]
#     #     _len += len(records_np_new)
#     #     del records_np_new
#     #     db.insert_records(records_dict)
#     #     # res = run_queries_old(db, records_np, 5, 10)
#     #     # print(evaluate_result(res))

# ###############################  main #######################
# if __name__ == "__main__":
#     # db = VecDB()
#     rng = np.random.default_rng(50)
    
#     records_np = rng.random((3000000, 70), dtype=np.float32)
#     _len = len(records_np)
#     del records_np

#     records_np = rng.random((3000000, 70), dtype=np.float32)
#     _len += len(records_np)
#     del records_np

#     db = VecDB(file_path = '6000000' , new_db = False)

#     # ########################### 10M + 15M + 20M  ###########################
#     for i in range(14):
#         print('we are in ', i+1 , 'M')
#         records_np_new = rng.random((1000000, 70), dtype=np.float32)
#         # records_np = np.concatenate([records_np, records_np_new])
#         # del records_np_new
#         # rng = np.random.default_rng(50)
#         # records_np = rng.random((3000000, 70), dtype=np.float32)
#         records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np_new)]
#         _len += len(records_np_new)
#         del records_np_new
#         db.insert_records(records_dict)
#         # res = run_queries_old(db, records_np, 5, 10)
#         # print(evaluate_result(res))


