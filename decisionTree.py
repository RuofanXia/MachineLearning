# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:29:42 2015

@author: Fan
"""
import time
import numpy as np
import collections
from numpy import genfromtxt
import pydot

def entropy(vector):
    k = len(vector)
    #get the unique value
    uv = set(vector)
    uv = list(uv)
    #get the percentage of each value
    counter = collections.Counter(vector)
    #initial a list to store percentage value
    a = []
    for i in range(len(uv)):
        a.append(counter[uv[i]]/float(k))
    #summation
    r = 0
    for j in range(len(a)):
        r += -a[j]*np.log2(a[j])
    return r

def splitinfo(attribute_data):
    attribute_value = list(set(attribute_data))
    attribute_data = attribute_data.tolist()
    v = 0
    for i in range(len(attribute_value)):
        subset = []
        for j in range(len(attribute_data)):
            if attribute_data[j] == attribute_value[i]:
                subset.append(attribute_data[j])
        d = len(subset)/float(len(attribute_data))
        v += -d*np.log2(d)
    return v
        
'''
def px_condi_entropy(attribute_data, class_data):
    attribute_value = list(set(attribute_data))
    attribute_data = attribute_data.tolist()
    r = 0
    for i in range(len(attribute_value)):
        index_array = []
        for j in range(len(attribute_data)):
            if attribute_data[j] == attribute_value[i]:
                index_array.append(j)
        p_value = (len(index_array))/float(len(class_data))
        current_class_data = []
        for k in range(len(index_array)):
            current_class_data.append(class_data[index_array[k]])
        r += p_value * entropy(current_class_data)
    return r

'''

def gainratio(attribute_data, class_data):
    infogain_value = infogain(attribute_data, class_data)
    splitinfo_value = splitinfo(attribute_data)
    return infogain_value/ float(splitinfo_value)
 
def infogain(attribute_data, class_data):
    #get the overall entropy
    t_entropy = entropy(class_data)
    attribute_value = list(set(attribute_data))
    #attribute_data = attribute_data.tolist()
    r = 0
    for i in range(len(attribute_value)):
        index_array = []
        for j in range(len(attribute_data)):
            if attribute_data[j] == attribute_value[i]:
                index_array.append(j)
        p_value = (len(index_array))/float(len(class_data))
        current_class_data = []
        for k in range(len(index_array)):
            current_class_data.append(class_data[index_array[k]])
        r += p_value * entropy(current_class_data)
    infogain_value = t_entropy - r
    return infogain_value

def t_infogain_index(real_attribute_data, class_data):
    #get the overall entropy
    t_entropy = entropy(class_data)
    ig_value_list = []
    candidate_threshold = 0
    for i in range(len(real_attribute_data)):
        candidate_threshold = real_attribute_data[i]
        smaller_data_index = []
        bigger_data_index= []
        for j in range(len(real_attribute_data)):
            if real_attribute_data[j] <= candidate_threshold:
                smaller_data_index.append(j)
            else:                
                bigger_data_index.append(j)
        smaller_class_data = []
        bigger_class_data = []
        for m in range(len(smaller_data_index)):
            smaller_class_data.append(class_data[smaller_data_index[m]])
        for n in range(len(bigger_data_index)):
            bigger_class_data.append(class_data[bigger_data_index[n]])
        smaller_p_value = len(smaller_class_data)/float(len(class_data))
        bigger_p_value = len(bigger_class_data)/float(len(class_data))
        ig_value = t_entropy - entropy(smaller_class_data)*smaller_p_value - entropy(bigger_class_data)*bigger_p_value
        ig_value_list.append(ig_value)
    index = ig_value_list.index(max(ig_value_list))
    return index

def t_infogain(real_attribute_data, class_data):
    #get the overall entropy
    t_entropy = entropy(class_data)
    ig_value_list = []
    candidate_threshold = 0
    for i in range(len(real_attribute_data)):
        candidate_threshold = real_attribute_data[i]
        smaller_data_index = []
        bigger_data_index= []
        for j in range(len(real_attribute_data)):
            if real_attribute_data[j] <= candidate_threshold:
                smaller_data_index.append(j)
            else:                
                bigger_data_index.append(j)
        smaller_class_data = []
        bigger_class_data = []
        for m in range(len(smaller_data_index)):
            smaller_class_data.append(class_data[smaller_data_index[m]])
        for n in range(len(bigger_data_index)):
            bigger_class_data.append(class_data[bigger_data_index[n]])
        smaller_p_value = len(smaller_class_data)/float(len(class_data))
        bigger_p_value = len(bigger_class_data)/float(len(class_data))
        ig_value = t_entropy - entropy(smaller_class_data)*smaller_p_value - entropy(bigger_class_data)*bigger_p_value
        ig_value_list.append(ig_value)
    index = ig_value_list.index(max(ig_value_list))
    print ig_value_list[index]
    return ig_value_list[index]
    
def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def update_continues_index(continues_index, index):
    update_index = []
    for j in range(len(continues_index)):
        if continues_index[j] < index:
            update_index.append(continues_index[j])
        elif continues_index[j] > index:
            update_index.append(continues_index[j]-1)   
    return update_index

def gen_mktree(N, attribute_list, tree, continues_index, threshold_list, threshold):
    #print "input_attribute", attribute_list
    #print N
    N = np.array(N)
    #get output values
    output_values = N[len(N)-1]
    #get input values
    original_inputvalue = []
    for i in range(len(N)-1):
        original_inputvalue.append(N[i])
    #get all input values
    input_values = []
    for i in range(len(N)-1):
        for j in range(len(N[i])):
            input_values.append(N[i][j])       
    if len(set(output_values)) == 1:
        tree.append(int(output_values[0]))
        threshold_list.append(threshold)
        return int(output_values[0])
    elif len(set(input_values)) == 1 :
    #elif len(set(input_values)) == 1:
        r = 0
        uv = set(output_values)
        uv = list(uv)
        counter = collections.Counter(output_values)
        for i in range(len(uv)):
            if counter[uv[i]] > r:
                r = uv[i]
        tree.append(int(r)) 
        threshold_list.append(threshold)
        return r
    else:
        #get the highest ig value
        output_intvlaue = []
        for i in range(len(output_values)):
            output_intvlaue.append(int(output_values[i]))
        
        ig_values = []
        #print "N", len(N)-1
        for i in range(len(N)-1):
            
            #convert string value to int
            digit_value = []
            for j in range(len(N[i])):
                digit_value.append(num(N[i][j]))
            #check if this attribute is continues or discrete
            '''
            if len(str(digit_value[0])) > 1:
                ig_values.append(t_infogain(digit_value, output_intvlaue))
                threshold_index = t_infogain_index(digit_value, output_intvlaue)
                #update matrix
                print digit_value[threshold_index]
                for k in range(len(N[i])):
                    if num(N[i][k]) <  digit_value[threshold_index]:
                        N[i][k] = 0
                    else:
                        N[i][k] = 1
            else:
                ig_values.append(infogain(digit_value, output_intvlaue)) 
            '''
            if i in continues_index:
                #treat it as a continues
                ig_values.append(t_infogain(digit_value, output_intvlaue))
            else:
                ig_values.append(infogain(digit_value, output_intvlaue))
        
        index = ig_values.index(max(ig_values))
        #print "ig: ", ig_values
        #print "length: ", len(ig_values)
        tree.append(int(attribute_list[index]))
        threshold_list.append(threshold)
        #print "attribute: ", attribute_list
        #time.sleep(5)
        update_attribute_list = np.delete(attribute_list, [index])
        #update continues index
        update_index = update_continues_index(continues_index, index)
        if index in continues_index:    
            nlnode = N[index]
            non_leaf_node = []
            for i in range(len(nlnode)):
                non_leaf_node.append(num(nlnode[i]))
            threshold_index = t_infogain_index(non_leaf_node, output_intvlaue)
            #print "threshold_index: ", threshold_index
            #print "non_leaf_node[threshold_index]:", non_leaf_node[threshold_index]
            smaller_index = []
            bigger_index = []
            for j in range(len(non_leaf_node)):
                if non_leaf_node[j] <= non_leaf_node[threshold_index]:
                    #for the value smaller than threshold
                    smaller_index.append(j)
                else:
                    bigger_index.append(j)
                    #for the value bigger than threshold
            total_index = []
            total_index.append(smaller_index)
            total_index.append(bigger_index)
            #print total_index
            for k in range(len(total_index)):
                n_threshold_list = []
                temp_tree=[]
                if len(ig_values) !=1:
                    update_N=[]
                    N_trans = np.transpose(N)
                    for l in range(N_trans.shape[0]):
                        if l in total_index[k]:
                            update_N.append(N_trans[l])
                    update_N = np.transpose(update_N)
                    update_N = np.delete(update_N, index, 0)
                    gen_mktree(update_N, update_attribute_list, temp_tree, update_index, n_threshold_list, non_leaf_node[threshold_index]) 
                    if len(n_threshold_list) == 1:
                        threshold_list.append(n_threshold_list[0])
                    else:
                        threshold_list.append(n_threshold_list)
                    if len(temp_tree) == 1:
                        tree.append(temp_tree[0])
                    else:
                        tree.append(temp_tree)
                else:
                    update_N=[]
                    N_trans = np.transpose(N)
                    for l in range(N_trans.shape[0]):
                        if l in total_index[k]:
                            update_N.append(N_trans[l])
                    update_N = np.transpose(update_N)
                    if k == 0:
                        for j in range(len(update_N[0])):
                            update_N[0][j] = 0
                        gen_mktree(update_N, update_attribute_list, temp_tree, update_index, n_threshold_list, non_leaf_node[threshold_index])     
                        if len(n_threshold_list) == 1:
                            threshold_list.append(n_threshold_list[0])
                        else:
                            threshold_list.append(n_threshold_list)
                        if len(temp_tree) == 1:
                            tree.append(temp_tree[0])
                        else:
                            tree.append(temp_tree)
                    else:
                        for j in range(len(update_N[0])):
                            update_N[0][j] = 1
                        gen_mktree(update_N, update_attribute_list, temp_tree, update_index, n_threshold_list, non_leaf_node[threshold_index])
                        if len(n_threshold_list) == 1:
                            threshold_list.append(n_threshold_list[0])
                        else:
                            threshold_list.append(n_threshold_list)
                        if len(temp_tree) == 1:
                            tree.append(temp_tree[0])
                        else:
                            tree.append(temp_tree)
        else:
            nlnode = N[index]
            non_leaf_node = []
            for i in range(len(nlnode)):
                non_leaf_node.append(int(nlnode[i]))
            children_node = list(set(non_leaf_node))
            #print children_node
            #print "children node: ", children_node
            #time.sleep(10)
            for i in range(len(children_node)):
                n_threshold_list = []
                temp_tree = []
                index_list = []
                for j in range(len(non_leaf_node)):
                    if int(children_node[i]) == non_leaf_node[j]:
                        index_list.append(j)
                #update Matrix
                if len(ig_values) == 1:
                    update_N = []
                    N_trans = np.transpose(N)
                    for k in range(N_trans.shape[0]):
                        if k in index_list:
                            update_N.append(N_trans[k])
                    update_N = np.transpose(update_N)
                    #update_N = np.delete(update_N, index, 0)
                    #gen_mktree(update_N, update_attribute_list, temp_tree, continues_index)
                    #tree.append(temp_tree)
                else:
                    update_N = []
                    N_trans = np.transpose(N)
                    for k in range(N_trans.shape[0]):
                        if k in index_list:
                            update_N.append(N_trans[k])
                    update_N = np.transpose(update_N)
                    update_N = np.delete(update_N, index, 0)
                gen_mktree(update_N, update_attribute_list, temp_tree, update_index, n_threshold_list, children_node[i])
                if len(n_threshold_list) == 1:
                    threshold_list.append(n_threshold_list[0])
                else:
                    threshold_list.append(n_threshold_list)                
                if len(temp_tree) == 1:
                    tree.append(temp_tree[0])
                else:
                    tree.append(temp_tree)
                
            
        '''
        if index in continues_index:
            update that column into binary
        remove index from continues_index
        #print str(N[index][0])
        if len(str(N[index][0])) > 1:
            nlnode = original_inputvalue[index]
            non_leaf_node = []
            for i in range(len(nlnode)):
                non_leaf_node.append(num(nlnode[i]))
            threshold_index = t_infogain_index(non_leaf_node, output_intvlaue)
            for k in range(2):
                if k == 0:
                    #for values less than threshold
                    temp_tree = []
                    index_list = []
                    for i in range(len(non_leaf_node)):
                        if non_leaf_node[i] < non_leaf_node[threshold_index]:
                            index_list.append(i)
                    update_N = []
                    N_trans = np.transpose(N)
                    for j in range(N_trans.shape[0]):
                        if j in index_list:
                            update_N.append(N_trans[j])
                    update_N = np.transpose(update_N)
                    #print update_N
                    update_N = np.delete(update_N, index, 0)
                    gen_mktree(update_N, update_attribute_list, temp_tree, continues_index)
                    tree.append(temp_tree)
                elif k == 1:
                    #for values less than threshold
                    temp_tree = []
                    index_list = []
                    for i in range(len(non_leaf_node)):
                        if non_leaf_node[i] > non_leaf_node[threshold_index]:
                            index_list.append(i)
                    update_N = []
                    N_trans = np.transpose(N)
                    for j in range(N_trans.shape[0]):
                        if j in index_list:
                            update_N.append(N_trans[j])
                    update_N = np.transpose(update_N)
                    update_N = np.delete(update_N, index, 0)
                    gen_mktree(update_N, update_attribute_list, temp_tree, continues_index)
                    tree.append(temp_tree)
        '''
                
                        
def mktree(N, attribute_list, tree):
    N = np.array(N)
    #get output values
    output_values = N[len(N)-1]
    #get input values
    original_inputvalue = []
    for i in range(len(N)-1):
        original_inputvalue.append(N[i])
    #get all input values
    input_values = []
    for i in range(len(N)-1):
        for j in range(len(N[i])):
            input_values.append(N[i][j])
            
    if len(set(output_values)) == 1:
        tree.append(int(output_values[0]))
        return int(output_values[0])
    elif len(set(input_values)) == 1:
        r = 0
        uv = set(output_values)
        counter = collections.Counter(output_values)
        for i in range(len(uv)):
            if counter[uv[i]] > r:
                r = uv[i]
        tree.append(r) 
        return r
    else:
        #get the highest ig value
        output_intvlaue = []
        for i in range(len(output_values)):
            output_intvlaue.append(int(output_values[i]))
        
        ig_values = []
        for i in range(len(N)-1):
            #convert string value to int
            int_value = []
            for j in range(len(N[i])):
                int_value.append(int(N[i][j]))
            ig_values.append(infogain(int_value, output_intvlaue))    
        #get the largest values of ig
        index = ig_values.index(max(ig_values))
        update_attribute_list = np.delete(attribute_list, [index])
        tree.append(attribute_list[index])
        nlnode = original_inputvalue[index]
        non_leaf_node = []
        for i in range(len(nlnode)):
            non_leaf_node.append(int(nlnode[i]))
        children_node = list(set(non_leaf_node))
        for i in range(len(children_node)):
            temp_tree = []
            index_list = []
            for j in range(len(non_leaf_node)):
                if children_node[i] == non_leaf_node[j]:
                    index_list.append(j)
            #update Matrix
            update_N = []
            N_trans = np.transpose(N)
            for k in range(N_trans.shape[0]):
                if k in index_list:
                    update_N.append(N_trans[k])
            update_N = np.transpose(update_N)
            update_N = np.delete(update_N, index, 0)
            mktree(update_N, update_attribute_list, temp_tree)
            tree.append(temp_tree)

def errords(one_record, tree, children_node, attribute_list):  
    #get root node
    root = int(tree[0])
    
    while True:
        value = one_record[root-1]
        node_index = children_node.index[value]
        sub_list = tree[node_index]
        if len(sub_list) == 1:
            if int(sub_list[0]) == one_record[len(one_record)-1]:
                return True
        else:
            root = int(sub_list[0])
            tree = sub_list
        
def errorrate(tree, one_record, threshold_list):
    while True:
        index = tree[0] - 1
        value = one_record[index]
        candidate_label= []
        for i in range(len(threshold_list)):
            if i > 0 and isinstance(threshold_list[i],list):
                candidate_label.append(threshold_list[i][0])
            elif i > 0 :
                candidate_label.append(threshold_list[i])
        #if not candidate_label:
        #    return False
        if len(candidate_label) == 2 and candidate_label[0] == candidate_label[1] :
            if num(value) <= candidate_label[0]:
                value_index = 0
            else:
                value_index = 1
        else:
            value_index = candidate_label.index(int(value))
        tree = tree[value_index+1]
        threshold_list = threshold_list[value_index+1]
        if isinstance(tree,int):
            if tree == int(one_record[len(one_record)-1]):
                return True
            else:
                return False
               
          
def cltree(test_data_set, attribute_list, tree, threshold_list):

    #get one record from test_dataset
    #reverse_ds = np.transpose(test_data_set)
    count = 0
    for i in range(len(np.transpose(test_data_set))):
        testCase = np.transpose(test_data_set)[i]
        #one_record = []
        testCase = np.transpose(test_data_set)[i]
        #print testCase
        f = errorrate(tree, testCase, threshold_list)
        if errorrate(tree, testCase, threshold_list) is False:
            count+=1
    print count
    error_rate = count/float(len(np.transpose(test_data_set)))
    return error_rate
'''      
def infogain(m):
    
    r = m.shape[0]
    c = m.shape[1]
    
    
    #total entropy H(S)
    #class_type = list(set(class_data))
    t_entropy = entropy(class_data)
    #print t_entropy  
    #print class_type
    #get each column
    infogain_result = []
    for i in range(c-1):
        #attribute_value = (m[:,i]).T
        attribute_data = np.squeeze(np.asarray(m[:,i]))
        infogain_result.append(t_entropy - px_condi_entropy(attribute_data, class_data))
    #print infogain_result
    return
'''
def convert2dot(filename,tree,attributes,ctr):
    
    ctr1=ctr+1;
    
    for i in range(len(tree)):

        # label name 
        if (isinstance(tree[i],str) or isinstance(tree[i],int)) and i == 0: 
            f = open(filename + ".dot", "a")
            f.write( "\n" + str(ctr) + " [label = \"" + str(tree[i]) + "\"];" )
            f.close()
            			

        # outcome
        if (isinstance(tree[i],str) or isinstance(tree[i],int)) and i > 0:
            f = open(filename + ".dot", "a")
            f.write( "\n" + str(ctr) + " -> " + str(ctr1) + "[label=\"" + str(attributes[i]) + "\"];" ) 
            f.write( "\n" + str(ctr1) + " [shape=box label = \"" + str(tree[i]) + "\"];" )
            f.close()
            ctr1=ctr1+1;

        # this is a subtree
        if isinstance(tree[i],list):
            f = open(filename + ".dot", "a")
            f.write( "\n" + str(ctr) + " -> " + str(ctr1) + "[label=\"" + str(attributes[i][0]) + "\"];" ) 
            f.close()
            ctr1 = convert2dot(filename,tree[i],attributes[i],ctr1)
    
    return ctr1

def tree2dot(filename,tree,attributes):
    
    # generate dot file
    f = open(filename + ".dot", "w")
    f.write( "digraph G{ \ncenter = 1; \nsize=\"1000,1000\";" )
    f.close()
    convert2dot(filename,tree,attributes,0)
    f = open(filename + ".dot", "a")
    f.write( "\n}" )
    f.close()
    
    # generate pdf file
    graph = pydot.graph_from_dot_file( filename + ".dot" )
    graph.write_pdf( filename + ".pdf" )

def main():
    
    ###################################
    ####problem for 2.e################
    ###################################
    '''
    my_data = genfromtxt('dataset_infoth.csv', delimiter=',', dtype=None)
    
    r = my_data.shape[0]
    c = my_data.shape[1]
    #get attributes
    attributes = my_data[0]
    #get the input data
    matrix = []
    
    #get rid of the first line of data
    my_data = np.delete(my_data, 0, 0)
    #check if the first column is id
    if attributes[0].lower() == "id":
        attributes = attributes[1:]
        for i in range(c-1):
            matrix.append(np.squeeze(np.asarray(my_data[:,i+1])))
    else:
        for i in range(c):
            matrix.append(np.squeeze(np.asarray(my_data[:,i])))
    
    #convert string attribute to int
    for i in range(len(attributes)):
        attributes[i] = i+1

    
    t_infogain(matrix[3], matrix[4])

    '''
    my_data = genfromtxt('mpg_train.csv', delimiter=',', dtype=None)
    
    r = my_data.shape[0]
    c = my_data.shape[1]
    #get attributes
    attributes = my_data[0]
    #get the input data
    matrix = []
    
    #get rid of the first line of data
    my_data = np.delete(my_data, 0, 0)
    #check if the first column is id
    if attributes[0].lower() == "id":
        attributes = attributes[1:]
        for i in range(c-1):
            matrix.append(np.squeeze(np.asarray(my_data[:,i+1])))
    else:
        for i in range(c):
            matrix.append(np.squeeze(np.asarray(my_data[:,i])))
    
    #convert string attribute to int
    for i in range(len(attributes)):
        attributes[i] = i+1
    
    ######################################
    ###calculate error rate of train ds###
    ######################################
    '''
    continues_index = []
    for i in range(c-1):
        #convert string value to int
        digit_value = []
        for j in range(len(matrix[i])):
            digit_value.append(num(matrix[i][j]))
        #check if this attribute is continues or discrete
        if len(set(digit_value))> 13:
            #treat it as a continues
            continues_index.append(i)
    
    tree = []
    threshold_list = []
    gen_mktree(matrix, attributes, tree, continues_index, threshold_list, 0) 
    print cltree(matrix, attributes, tree, threshold_list)
    '''
    ######################################
    ###calculate error rate of test ds####
    ######################################
    '''
    continues_index = []
    for i in range(c-1):
        #convert string value to int
        digit_value = []
        for j in range(len(matrix[i])):
            digit_value.append(num(matrix[i][j]))
        #check if this attribute is continues or discrete
        if len(set(digit_value))> 13:
            #treat it as a continues
            continues_index.append(i)
    
    tree = []
    threshold_list = []
    gen_mktree(matrix, attributes, tree, continues_index, threshold_list, 0)
     # load test dataset
    test_data = genfromtxt('mpg_test.csv', delimiter=',', dtype=None)
    #get attributes
    attris = test_data[0]
    
    col = test_data.shape[1]
    #get the input data
    m = []
    #get rid of the first line of data
    test_data = np.delete(test_data, 0, 0)
    #check if the first column is id
    if attris[0].lower() == "id":
        attris = attris[1:]
        for i in range(col-1):
            m.append(np.squeeze(np.asarray(test_data[:,i+1])))
    else:
        for i in range(col):
            m.append(np.squeeze(np.asarray(test_data[:,i])))
    print tree
    print threshold_list
    #one_record = np.transpose(m)[13]
    #print one_record
    #print errorrate(tree, one_record, threshold_list)
    print cltree(m, attributes, tree, threshold_list)

    '''
    #######################################
    ###problem 5b figure .pdf and .dot#####
    ####################################### 
    
    continues_index = []     
    for i in range(c-1):
        #convert string value to int
        digit_value = []
        for j in range(len(matrix[i])):
            digit_value.append(num(matrix[i][j]))
        #check if this attribute is continues or discrete
        if len(set(digit_value))> 10:
            #treat it as a continues
            continues_index.append(i)
    #print continues_index
    tree = []
    threshold_list =[]
    #cltree(matrix,attributes,tree)
    gen_mktree(matrix, attributes, tree, continues_index, threshold_list, 0) 
    #cltree(matrix,attributes,tree)
    print 'tree is:'
    print tree
    
    print 'threshold:'
    print threshold_list
    
    tree2dot("problem5b",tree, threshold_list)
    

    #######################################
    ###problem 5d figure .pdf and .dot#####
    #######################################
    
    continues_index = []
    for i in range(c-1):
        #convert string value to int
        digit_value = []
        for j in range(len(matrix[i])):
            digit_value.append(num(matrix[i][j]))
        #check if this attribute is continues or discrete
        if len(set(digit_value))> 13:
            #treat it as a continues
            continues_index.append(i)
    #print continues_index
    tree = []
    threshold_list =[]
    gen_mktree(matrix, attributes, tree, continues_index, threshold_list, 0) 
    print 'tree is:'
    print tree
    
    print 'threshold:'
    print threshold_list
    
    tree2dot("problem5d",tree, threshold_list)
    
              
if __name__ == "__main__":
    main()
