# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 00:42:38 2022

@author: zahra
"""


import sys
sys.path.append('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/code VRP/TDVRPTW')

import random
import numpy as np 
import copy
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from random import randrange



#Loading data from file
class Node:
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
  
def create_from_file(file_path):
    # Read location of service point, customer from file
    node_list = []
    with open(file_path, 'rt') as f:
        count = 1
        for line in f:
            if count == 5:
                vehicle_num, vehicle_capacity = line.split()
                vehicle_num = int(vehicle_num)
                vehicle_capacity = int(vehicle_capacity)
            elif count >= 10:
                node_list.append(line.split())
            count += 1
    
    node_list=node_list[:26]
    node_list.append(node_list[0].copy())
    node_list[26][0]='26'
    node_num = len(node_list)
    nodes = list(Node(int(item[0]), np.float32(item[1]), np.float32(item[2]), int(item[3]), np.float32(item[4])*10, np.float32(item[5])*10, np.float32(item[6])*10) for item in node_list)
    
    # Create a distance matrix
    node_dist_mat = np.zeros((node_num, node_num), dtype=np.float32)
    for i in range(node_num):
        node_a = nodes[i]
        node_dist_mat[i][i] = 1e-8
        for j in range(i+1, node_num):
            node_b = nodes[j]
            node_dist_mat[i][j] = round(calculate_dist_coordinate(node_a, node_b),1)
            node_dist_mat[j][i] = node_dist_mat[i][j]

    return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity

def calculate_dist_coordinate(node_a, node_b):
    return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


#%%


class Problem_Genetic(object):
    
    def __init__(self,genes,individuals_length,fitness):
        self.genes= genes
        self.individuals_length= individuals_length
        self.fitness= fitness

    def mutation(self, chromosome0, prob):        
        first_element=chromosome0[0]
        remaining=chromosome0[1:]
        def inversion_mutation(chromosome_aux,l):
                chromosome = chromosome_aux
                index1 = randrange(0,l-1)
                index2 = randrange(index1,l)
                chromosome_mid = chromosome[index1:index2]
                chromosome_mid.reverse()
                chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                return chromosome_result
            
        def swap_mutation(chromosome_aux,l):
                chromosome = chromosome_aux
                index1 = randrange(0,l-1)
                index2 = randrange(index1+1,l)               
                chromosome_result = chromosome.copy()
                chromosome_result[index1]= chromosome[index2]
                chromosome_result[index2]= chromosome[index1]
                return chromosome_result
            
        def scramble_mutation(chromosome_aux,l):
                chromosome = chromosome_aux
                index1 = randrange(0,l-1)
                index2 = randrange(index1+1,l)
                #print(index1, index2)
                chromosome_mid = chromosome[index1:index2]
                random.shuffle(chromosome_mid)
                chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                return chromosome_result
            
        l=len(remaining)
        
        if random.random() < prob :
            mutation_type=random.random()
            
            if mutation_type < (1/3):
                remaining = inversion_mutation(remaining,l)
            elif mutation_type < (2/3):
                remaining = swap_mutation(remaining,l)
            else:
                remaining = scramble_mutation(remaining,l)
        
        chromosome=[first_element]+remaining

        for j in range(l+1):
            if chromosome[j][0]==0:
                # update only 50% of dt in a dt list
                if random.random() <0.5:
                    chromosome[j]=(0,round(random.uniform(depot_ready_time, depot_due_time/2),5))
        return chromosome
                
    def crossover1_chromosome(self,parent):
        up = round(num_customer/7) + random.randint(3,5)
        cut=random.randint(2,up)
        #25 : (2,6)
        #50 : (2,10)
        #100: (2,17)

        List = [i for i in range(cut + 1)]

        shuffle1=List.copy()
        shuffle2=List.copy()

        random.shuffle(shuffle1)
        random.shuffle(shuffle2)

        part=[0]*(cut+1)
        pos = sorted(random.sample(range(1,len(parent)), cut))

        for i in range(cut+1):
            if i==0:                
                part[i]=parent[:pos[i]]
            elif i!=cut:
                part[i]=parent[pos[i-1]:pos[i]]
            else:
                part[i]=parent[pos[i-1]:]
        
        child1=[]
        child2=[]

        for i in range(cut+1):
            child1.append(part[shuffle1[i]])
            child2.append(part[shuffle2[i]])
            
        child1 = [item for sublist in child1 for item in sublist]
        child2 = [item for sublist in child2 for item in sublist]

        #select randomly
        if random.random()<0.5:        
            return child1
        else:
            return child2


    def crossover2_chromosome(self,parent1,parent2):

        point1, point2 = sorted(random.sample(range(min(len(parent1),len(parent2))), 2))
        point1=3
        point2=7
        
        child1=parent1.copy()
        child2=parent2.copy()
        
        child1[point1:point2+1]=parent2[point1:point2+1]
        child2[point1:point2+1]=parent1[point1:point2+1]


        #repair missing and duplicate except depot
        def repair(chromosome, other_chromosome):
            missing = [g for g in other_chromosome if g not in chromosome]
            
            missing = [(x,y) for x,y in missing if x!=0]

            for i, gene in enumerate(other_chromosome):
                if gene in missing:
                    chromosome.insert(i, gene)

            unique_chromosome = [x for i, x in enumerate(chromosome) if x[0]==0 or x not in chromosome[:i]]
            return unique_chromosome
        
        r=repair(child1, parent2)
        s=repair(child2, parent1)
        
        return r,s
    


    def blend_crossover_dt(self,parent1, parent2, alpha):
        n1, n2 = len(parent1), len(parent2)

        child1=[0]*n1
        child2=[0]*n2
            
        n=min(n1,n2)
            
        for i in range(n):
            child1[i]=alpha * parent1[i] + (1 - alpha) * parent2[i] 
            
        for i in range(n):
            child2[i]=alpha * parent2[i] + (1 - alpha) * parent1[i] 
                
        if n1<n2:       
            for i in range(n1,n2):
                child2[i]=parent2[i]
                    
        elif n1>n2:        
            for i in range(n2,n1):
                child1[i]=parent1[i]
                
        return child1,child2
    



#%%
#travel time function considering time dependency and 5 time intervals and speed profile

def Travel_Time(depature_time, dist,i,j):
    # Get the cluster for the given arc index
    cluster_index = data['clusters'][i][j]
    speed = speeds.get(cluster_index)
    #index=1000
    for r in time_intervals:
        if r[0] <= depature_time <= r[1]:
            index=time_intervals.index(r)
            break
    
    d=dist
    st = depature_time
    TT= 0 
    for i in range(index, l_time_intervals):
        ft = st + d / speed[i]
        if ft <= time_intervals[i][1]:
            TT += ft - st
            break
        else:
            TT += time_intervals[i][1] -st
            d -= (time_intervals[i][1] - st)*speed[i]
            st = time_intervals[i][1]
    return TT



#%%

#Total_time+TW+capacity
def fitness(seq, dt,num_v, num_c, p_cap, p_TW):
    lenght = num_c + num_v*2
    dt.append(12300000)
    Total_time = 0
    penalty_TW=0
    coun = 0
    value_penalty = 0
    current_capacity= 0
    
    departure_time = dt[coun]

    for i in range(1, lenght):

        if departure_time<0:
            print('seq,dt',seq,dt,departure_time)
        departure_time = departure_time + Travel_Time(departure_time, node_dist_mat[seq[i-1]][seq[i]],seq[i-1],seq[i])

        LT = nodes[seq[i]].due_time
        #if math.floor(departure_time)>LT:
        #print('TW',departure_time,LT,math.floor(departure_time)>LT )
        penalty_TW += p_TW*(departure_time-LT) if departure_time-LT>1 else 0
    
        current_capacity += nodes[seq[i]].demand

        if seq[i] == num_c+1 :
    
            Total_time += departure_time - dt[coun]
            coun +=1
            departure_time = dt[coun] 
            
            if current_capacity > vehicle_capacity:
                value_penalty += p_cap*(current_capacity-vehicle_capacity)
            current_capacity = 0

        if departure_time < nodes[seq[i]].ready_time:
            departure_time = nodes[seq[i]].ready_time 
        departure_time += nodes[seq[i]].service_time
        
    fitness_value = Total_time + penalty_TW + value_penalty
    dt.pop(-1)

    return fitness_value

def feasibility_check(seq, dt,num_v, num_c, p_cap, p_TW):
    lenght = num_c + num_v*2
    dt.append(12300000)
    Total_time = 0
    penalty_TW=0
    coun = 0
    value_penalty = 0
    current_capacity= 0
    
    departure_time = dt[coun]
    for i in range(1, lenght):
        
        departure_time = departure_time + Travel_Time(departure_time, node_dist_mat[seq[i-1]][seq[i]],seq[i-1],seq[i])

        LT = nodes[seq[i]].due_time
        penalty_TW += p_TW*(departure_time-LT) if  departure_time-LT>1 else 0
    
        current_capacity += nodes[seq[i]].demand

        if seq[i] == num_customer+1 :
            
            Total_time += departure_time - dt[coun]
            #print('TT',departure_time - dt[coun])
            coun +=1
            departure_time = dt[coun] 
            
            if current_capacity > vehicle_capacity:
                value_penalty += p_cap*(current_capacity-vehicle_capacity)
            current_capacity = 0

        if departure_time < nodes[seq[i]].ready_time:
            departure_time = nodes[seq[i]].ready_time 
        departure_time += nodes[seq[i]].service_time
        
    dt.pop(-1)

    feasiblity_TW = 1 if penalty_TW ==0 else 0
    feasiblity_cap= 1 if value_penalty ==0 else 0


    return feasiblity_TW, feasiblity_cap , Total_time 

#%%

def create_routes(S, num_v):
   indices=[i for i, tpl in enumerate(S[:-1]) if tpl== 0] 
   routes=[]
   if num_v==0:
       routes=[S]
   else:
       j=0
       for j in range(1,num_v):
           routes.append(S[indices[j-1]:indices[j]])
       routes.append(S[indices[j]:])
   return routes



def objective(dt, c , num_c, p_cap, p_TW):
    if dt < 0:
        return 123456789000 
    else:
        return fitness(c, [dt] , 1, num_c, p_cap, p_TW)

def new_optimal_dt(c, dt , num_c, p_cap, p_TW):
    #print('c, dt , num_c, p_cap, p_TW',c, dt , num_c, p_cap, p_TW)
    bounds = (0, 10000)  # dt>=0
    result = minimize_scalar(objective, bounds=bounds, args=(c, num_c, p_cap, p_TW,))
    #best_dt = result.x
    best_dt = max(result.x, 0)
    return best_dt 



def optimal_dt_routes(S,dt, num_v, p_cap, p_TW,fit):
    new_dt=[]
    Routes= create_routes(S,num_v)
    for i in range(len(Routes)):
        num_c= len(Routes[i])-2
        temp_dt = new_optimal_dt(Routes[i],dt[i] , num_c, p_cap, p_TW )
        new_dt.append (temp_dt)
    
        
    new_fit = fitness( S, new_dt, num_v, num_customer, p_cap, p_TW)
    
    if new_fit > fit :
        new_dt=dt
        
    return S,new_dt,num_v, p_cap,p_TW
    

#Total_time + TW for one route
def fitness_one(seq, dt, num_c, p_TW):
    lenght = num_c + 2

    #Total_time = 0
    penalty_TW=0
    
    departure_time = dt
    for i in range(1, lenght):
        if departure_time<0:
            print('one,seq,dt',seq,dt,departure_time)
        departure_time = departure_time + Travel_Time(departure_time, node_dist_mat[seq[i-1]][seq[i]],seq[i-1],seq[i])
        
        LT = nodes[seq[i]].due_time
        penalty_TW += p_TW*(departure_time-LT) if departure_time-LT>1  else 0
        

        if departure_time < nodes[seq[i]].ready_time:
            departure_time = nodes[seq[i]].ready_time 
        departure_time += nodes[seq[i]].service_time
        
        if seq[i]==0:
            departure_time = dt
    
    Total_time = departure_time - dt

    fitness_value = Total_time + penalty_TW 
    return fitness_value



#the output is min insertion cost of customer i in route r (fit_best) and the best position result in min cost (i_best) 
def min_reinsertion_cost(customer, route,dt_i,p_TW):
    #print('customer, route,dt_i,p_TW',customer, route,dt_i,p_TW)
    #new_route=route.copy()
    fit_best= float('inf')
    i_best=-10000
    
    l=len(route)
    
    for i in range(1,l):
        if i==1:
            new_route=route.copy()
            new_route.insert(i,customer)
        else:
            new_route[i],new_route[i-1] = new_route[i-1],new_route[i]
        #print(new_route)    
        fit= fitness_one(new_route,dt_i,l-1,p_TW)
        if fit<fit_best:
            fit_best=fit
            i_best=i
    return fit_best,i_best


#the output is k best routes for inserting customer i :[((1591.5701697995146, 0), 0),((6452124.1671962505, 0), 2), ((320452251.8905066, 1), 1)]
#for instance, the above list inclueds 3 best routes, ((min insertioncost,best position in route), which route)

def first_time(customer, routes,dt,k,p_TW,num_v):
    route_cost=[]
    for i,route in enumerate(routes):
        #print('route',route)
        route_cost.append(min_reinsertion_cost(customer, route,dt[i],p_TW))
    return route_cost


def sorted_route_insertion_cost(customer, routes,dt,k,p_TW,num_v,best_r,route_cost,counter):
    if best_r!=None:
        #print('customer',customer,routes[best_r])

        route_cost[best_r] = min_reinsertion_cost(customer, routes[best_r],dt[best_r],p_TW)
    #print('route_cost',route_cost) 
    sorted_indices = sorted(range(num_v), key=lambda i: route_cost[i][0])
    #print('sorted_indices',sorted_indices) 
    return [(route_cost[i], i) for i in sorted_indices[:k+1]],route_cost


def regret_insertion(removed_customers, solution, dt, k, num_v,p_TW):
    unassigned_customers = removed_customers.copy()
    #print('solution_regret',solution)
    #print('------------------unassigned_customers',unassigned_customers)
    routes = create_routes(solution,num_v)
    
    best_r = None   
    route_cost_customer=[]
    counter=0
    while unassigned_customers:
        counter+=1
        regret_values = []
        for i,customer in enumerate(unassigned_customers):
            #print('---------customer',customer)
            if counter==1:
                route_cost_customer.append(first_time(customer, routes,dt,k,p_TW,num_v))
            
            # Calculate the regret value for customer insertion
            best_routes,route_cost_customer[i] = sorted_route_insertion_cost(customer, routes,dt,k,p_TW,num_v,best_r,route_cost_customer[i],counter)
            regret = sum(element[0][0] - best_routes[0][0][0] for element in best_routes[1:]) 
            
            #(customer, regret, best_route, best_position) 
            regret_values.append((customer, regret, best_routes[0][1], best_routes[0][0][1]))
            
        # Sort customers based on regret values in descending order
        regret_values.sort(key=lambda x: x[1], reverse=True)
        #print('regret_values',regret_values[0])

        # Insert customer into the route with the lowest insertion cost
        customer_to_insert, _ ,best_r,best_p= regret_values[0]
        routes[best_r].insert(best_p,customer_to_insert)
        index_c=unassigned_customers.index(customer_to_insert)
        unassigned_customers.pop(index_c)
        route_cost_customer.pop(index_c)
        
    solution =[item for sublist in routes for item in sublist]
    return solution


#%%


def one_empty_route(S,dt,num_v):
    #print('------- s,dt',S,dt, num_v, len(dt))

    counter_r=num_v
    counter_er=0
    for i in range(len(S)-1,0,-1):
        #print('i',i)
        if S[i]==num_customer+1:
            counter_r -= 1
            #print('counter_r',counter_r)
            if S[i-1]==0:
                counter_er+=1
                dt[counter_r]=0

                #print('counter_er',counter_er)

                if counter_er>1:
                    S.pop(i)
                    S.pop(i)
                    dt.pop(counter_r)
                    #print('s,dt',S,dt)
    
    if counter_er==0:
        S.extend([0,num_customer+1])
        dt.append(0)
            
    num_v = num_v-counter_er+1
    
    
    c=S.count(0)
    if c!=num_v or c!=len(dt) or num_v!= len(dt):
        print('c',c)
        print('empty,dt,num_v',S,dt,num_v)
        raise ValueError

    return S,dt,num_v

def Local_Search_individual(S, dt, num_v, num_c, p_cap, p_TW, fit):
    initial_p_cap = p_cap
    initial_p_TW = p_TW
    best_S=S
    best_dt=dt
    best_num_v=num_v
    best_fit=fit
    
    

    counter_f =0
    counter = 0
    while counter < 1000:
        #check num_v and empty > make sure there is only one empty 
        S,dt,num_v = one_empty_route(S,dt,num_v)
            
            
        counter += 1
        #print('---------------- counter', counter)
        
        old_fit = fit
        
        S, fit = inter_relocate(S, dt, num_v, num_c, p_cap, p_TW, fit)
        #print('after inter relocate: fit', fit)

        S, fit = intra_relocate(S, dt, num_v, num_c, p_cap, p_TW, fit)
        #print('after intra relocate: fit', fit)
        
        S, dt, fit = local_search_dt(S, dt, num_v,num_c, p_cap, p_TW, fit)
        #print('after departure time local serach: fit', fit)

        S, fit = inter_swap(S, dt, num_v, num_c, p_cap, p_TW, fit)
        #print('after inter swap: fit', fit)
        
        S, fit = block_insertion_2_3(S, dt, num_v, num_c, p_cap, p_TW, fit)
        #print('after block insertion: fit', fit)
        
        #S, fit  = ALNS_individual(S, dt, num_v, num_c, p_cap, p_TW, fit)
        #print('after ALNS: fit', fit)

        improve = old_fit - fit

        if counter_f >= 1:
            if improve ==0 :
                #print('counter',counter)
                break
        
        if counter > 30 and counter_f==0:
            raise ValueError ('LS did not find feasible solution after 30 run')
        
        feasibility_TW, feasibility_cap, Total_time = feasibility_check(S, dt, num_v, num_c, p_cap, p_TW)
        
        if feasibility_TW+feasibility_cap==2:
            counter_f +=1
            best_S = S.copy()
            best_dt = dt.copy()
            best_num_v = num_v
            best_fit = fit
            #print('best_S',best_S,best_num_v)

        random_TW = random.uniform(1.5, 2.5)
        random_cap = random.uniform(1.5, 2.5)
        
        p_TW = p_TW * random_TW if not feasibility_TW else p_TW / random_TW
        p_cap = p_cap * random_cap if not feasibility_cap else p_cap / random_cap
        

        fit = fitness(S, dt, num_v, num_c, p_cap, p_TW)
    
    S = best_S
    dt = best_dt
    num_v = best_num_v
    fit = best_fit
    
    return S, dt, num_v, num_c, initial_p_cap, initial_p_TW, fit



def local_search_dt(S, dt, num_v, num_c, p_cap, p_TW, fit):
    
    S, dt, num_v, p_cap, p_TW = optimal_dt_routes(S, dt, num_v, p_cap, p_TW, fit)
    new_fit = fitness(S, dt, num_v, num_c, p_cap, p_TW)
    
    if new_fit -fit > 0.0001:
        print('Error')
        print('new_fit',new_fit,'fit',fit)
        raise ValueError("local serach dt create worse solution")
    return S , dt, new_fit 



# it does not need to consider p_cap , p_vehicle because they do not change in intra relocation
def intra_relocate(S, dt, num_v, num_c, p_cap, p_TW, fit):
    #print('dt',dt)
    Updated_S = S
    #print('initial_whole_fit',fit)
    
    routes0 = create_routes(S, num_v)   
    routes = copy.deepcopy(routes0)

    
    for r in range(num_v):
        ##print('----------------------------------- r',r)

        #route = copy.deepcopy(routes[r])
        l = len(routes[r])
        initial_fit = fitness_one(routes[r], dt[r], l - 2, p_TW)
        
        for p1, c in enumerate(routes[r][1:-1], start=1):
            ##print('---------- customer', c)
            old_fit = fit

            #print('route0',routes[r])
            best_route = routes[r]
            best_improve=0
            
            route = copy.deepcopy(routes[r])
            ##print('route:', route, 'initial_fit',initial_fit)
            
            for p2 in range(1, l - 1):
                if p2==1:
                    route.remove(c)  
                    route.insert(p2, c)
                else:
                    temp1 = route[p2]
                    route[p2]= route[p2-1]
                    route[p2-1]= temp1
                                    
                new_fit = fitness_one(route, dt[r], l - 2, p_TW)
                #print('after swap : route:', route,'new_fit:', new_fit)


                improve = initial_fit- new_fit
                #print('improve',improve)
                if improve > best_improve:
                    best_route = route.copy()
                    best_improve = improve
                    best_fit = new_fit
                #print('best_improve',best_improve)

            
            if best_improve!=0:
                initial_fit = best_fit
                routes[r]= best_route
                fit -= best_improve

             
    Updated_S= [item for sublist in routes for item in sublist]
    #print('Updated_S',Updated_S)
    fit_Updated_S = fitness(Updated_S, dt, num_v, num_c, p_cap , p_TW)
    return Updated_S, fit_Updated_S




def Total_demand_route(chromosome):
    demand = 0
    for i in chromosome[1:-1]:
        demand += nodes[i].demand
    return demand

def inter_relocate(S, dt, num_v, num_c, p_cap, p_TW, fit):
    Updated_S = S
    routes0 = create_routes(S, num_v)   
    routes = copy.deepcopy(routes0)

    Demand= [Total_demand_route(i) for i in routes]
    #print('routes',routes)

    #for r1 in range(1):
    for r1 in range(num_v):

        l1=len(routes[r1])
        #print('routes[r1]',routes[r1])
        d1=Demand[r1]
        #print('--------------------------------- r1=',r1)

        fit1 = fitness_one(routes[r1], dt[r1],l1-2, p_TW) + p_cap*(d1>vehicle_capacity)*(d1-vehicle_capacity)
        #print('fit1',fit1)
        
        for c in routes[r1][1:-1]:
            #print('------- customer',c)
            old_fit = fit

            l1=len(routes[r1])
            customer_demand = nodes[c].demand
            updated_d1 = (Demand[r1] - customer_demand)
            p_cap1 = (updated_d1>vehicle_capacity)
            
            route1 = copy.deepcopy(routes[r1])
            route1.remove(c)
            #print('uroute1',route1)
            u_fit1 = fitness_one(route1, dt[r1],l1-3, p_TW) + p_cap*p_cap1*(updated_d1-vehicle_capacity)
            #print('ufit1',u_fit1)

            best_improve = 0
            best_r2 = -1
            
            for r2 in range(num_v):
            #for r2 in range(2,5):
                #print('------- r2=',r2)
                if r1!=r2:
                    l2=len(routes[r2])
                    d2=Demand[r2]
                    fit2 = fitness_one(routes[r2], dt[r2],l2-2, p_TW) + p_cap*(d2>vehicle_capacity)*(d2-vehicle_capacity)
                    #print('fit2',fit2)
                    initial_fit = fit1 + fit2
                    #print('routes[r2]=', routes[r2])
                    #print('initial fit=',fit1,fit2,initial_fit)

                    updated_d2 = ( d2 + customer_demand)
                    p_cap2 = (updated_d2>vehicle_capacity)

                    for p2 in range(1, l2):
                        #print('-- p2=',p2)
                        if p2==1:
                            route2 = copy.deepcopy(routes[r2])
                            route2.insert(1, c)
                        else:
                            temp1 = route2[p2]
                            route2[p2]= route2[p2-1]
                            route2[p2-1]= temp1
                            
                        u_fit2 = fitness_one(route2, dt[r2],l2-1, p_TW) + p_cap*p_cap2*(updated_d2-vehicle_capacity)
                        new_fit = u_fit1 + u_fit2
                        
                        #print('uroute2', route2)
                        #print('new_fit',u_fit1,u_fit2,new_fit)
                        improve = initial_fit - new_fit                         
                        
                        if improve > best_improve:
                            best_improve = improve
                            best_route2 = route2.copy()
                            best_r2=r2
                            best_d2 = updated_d2
                        # print('improve', improve)
                        # print('best_improve', best_improve)
                        # print('best_r2',best_r2)
                    
            if best_improve!=0:            
                routes[r1] = route1
                routes[best_r2] = best_route2
                Demand[r1] = updated_d1
                Demand[best_r2] = best_d2
                fit -= best_improve
                fit1 = u_fit1

            ##
            

        
    Updated_S= [item for sublist in routes for item in sublist]
    fit_Updated_S = fitness(Updated_S, dt, num_v, num_c, p_cap , p_TW)

    return Updated_S, fit_Updated_S


def inter_swap(S, dt, num_v, num_c, p_cap, p_TW, fit):
    Updated_S = S

    routes0 = create_routes(S, num_v)   
    routes = copy.deepcopy(routes0)
    
    
    Demand= [Total_demand_route(i) for i in routes]
    #print('routes',routes)
    
    for r1 in range(num_v):
        
        for p1, c in enumerate(routes[r1][1:-1], start=1):
           
            old_fit = fit

            l1=len(routes[r1])

            customer_demand1 = nodes[c].demand
            

            best_improve = 0
            best_r1 = -1
            best_r2 = -1
            
            for r2 in range(num_v):

                if r1!=r2:
                    l2=len(routes[r2])

                    d1=Demand[r1]
                    d2=Demand[r2]
                    
                    fit1 = fitness_one(routes[r1], dt[r1],l1-2, p_TW) + p_cap*(d1>vehicle_capacity)*(d1-vehicle_capacity)
                    fit2 = fitness_one(routes[r2], dt[r2],l2-2, p_TW) + p_cap*(d2>vehicle_capacity)*(d2-vehicle_capacity)
                    #print('fit2',fit2)
                    initial_fit = fit1 + fit2

                    for p2 in range(1, l2-1):
                        if p2==1:
                            route1 = copy.deepcopy(routes[r1])
                            route2 = copy.deepcopy(routes[r2])
                            route1[p1],route2[p2]=route2[p2],route1[p1]

                            
                        else:
                            temp1 = route1[p1]
                            route1[p1]= route2[p2]
                            route2[p2]=route2[p2-1]
                            route2[p2-1]= temp1
                        #print('route1', route1)
                        #print('route2',route2,'\n')
                        
                        customer_demand2= nodes[route1[p1]].demand
                        updated_d1 = Demand[r1] - customer_demand1+ customer_demand2
                        p_cap1 = (updated_d1>vehicle_capacity)

                        updated_d2 = Demand[r2] + customer_demand1 - customer_demand2
                        p_cap2 = (updated_d2>vehicle_capacity)
    
                        u_fit1 = fitness_one(route1, dt[r1],l1-2, p_TW) + p_cap*p_cap1*(updated_d1-vehicle_capacity)
                        u_fit2 = fitness_one(route2, dt[r2],l2-2, p_TW) + p_cap*p_cap2*(updated_d2-vehicle_capacity)
                        new_fit = u_fit1 + u_fit2
                        
                        #print('uroute2', route2)
                        #print('new_fit',u_fit1,u_fit2,new_fit)
                        improve = initial_fit - new_fit                         
                        
                        if improve > best_improve:
                            best_improve = improve
                            best_route1 = route1.copy()
                            best_route2 = route2.copy()
                            best_r1=r1
                            best_r2=r2
                            best_d1 = updated_d1
                            best_d2 = updated_d2
     
                        
            if best_improve!=0:            
                routes[best_r1] = best_route1
                routes[best_r2] = best_route2
                Demand[best_r1] = best_d1
                Demand[best_r2] = best_d2
                fit -= best_improve

              
        
    Updated_S= [item for sublist in routes for item in sublist]
    fit_Updated_S = fitness(Updated_S, dt, num_v, num_c, p_cap , p_TW)
    return Updated_S, fit_Updated_S
    

def block_insertion(S, dt, num_v, num_c, p_cap, p_TW, fit):
    Updated_S = S
 
    routes0 = create_routes(S, num_v)   
    routes = copy.deepcopy(routes0)

    Demand= [Total_demand_route(i) for i in routes]
  
    for r1 in range(num_v):

        l1=len(routes[r1])
        if l1<4:
            continue
        d1=Demand[r1]
        #print('--------------------------------- r1=',r1)

        fit1 = fitness_one(routes[r1], dt[r1],l1-2, p_TW) + p_cap*(d1>vehicle_capacity)*(d1-vehicle_capacity)
        
        best_improve = 0
        best_r2 = -1
        best_r1 = -1
        
        for p1, c in enumerate(routes[r1][1:-2], start=1):
            c_next= routes[r1][p1+1]

            old_fit = fit
            
            l1=len(routes[r1])
            customer_demand = nodes[c].demand + nodes[c_next].demand
            updated_d1 = (Demand[r1] - customer_demand)
            p_cap1 = (updated_d1>vehicle_capacity)
            
            route1 = copy.deepcopy(routes[r1])
            del route1[p1:p1+2]

            #print('uroute1',route1)
            u_fit1 = fitness_one(route1, dt[r1],l1-4, p_TW) + p_cap*p_cap1*(updated_d1-vehicle_capacity)
            #print('ufit1',u_fit1)


            
            for r2 in range(num_v):
                #print('------- r2=',r2)
                if r1!=r2:
                    l2=len(routes[r2])
                    d2=Demand[r2]
                    fit2 = fitness_one(routes[r2], dt[r2],l2-2, p_TW) + p_cap*(d2>vehicle_capacity)*(d2-vehicle_capacity)
                    #print('fit2',fit2)
                    initial_fit = fit1 + fit2

                    updated_d2 = ( d2 + customer_demand)
                    p_cap2 = (updated_d2>vehicle_capacity)

                    for p2 in range(1, l2):
                        #print('-- p2=',p2)
                        if p2==1:
                            route2 = copy.deepcopy(routes[r2])
                            route2[1:1] = [c,c_next]
                        else:
                            temp1 = route2[p2+1]
                            route2[p2], route2[p2+1]= route2[p2-1], route2[p2]
                            route2[p2-1]= temp1
                            
                        u_fit2 = fitness_one(route2, dt[r2],l2, p_TW) + p_cap*p_cap2*(updated_d2-vehicle_capacity)
                        new_fit = u_fit1 + u_fit2
                        
                        #print('uroute2', route2)
                        #print('new_fit',new_fit)
                        improve = initial_fit - new_fit                         
                        
                        if improve > best_improve:
                            best_improve = improve
                            best_route1 = route1.copy()
                            best_route2 = route2.copy()
                            best_r1=r1
                            best_r2=r2
                            best_d1 = updated_d1
                            best_d2 = updated_d2
                    
                    
        if best_improve!=0:            
            routes[best_r1] = best_route1
            routes[best_r2] = best_route2
            Demand[best_r1] = best_d1
            Demand[best_r2] = best_d2
            fit -= best_improve
            fit1 = u_fit1
            
        
    Updated_S= [item for sublist in routes for item in sublist]
    fit_Updated_S = fitness(Updated_S, dt, num_v, num_c, p_cap , p_TW)
    return Updated_S, fit_Updated_S


def block_insertion_2_3(S, dt, num_v, num_c, p_cap, p_TW, fit):
    Updated_S = S

    routes0 = create_routes(S, num_v)   
    routes = copy.deepcopy(routes0)

    Demand= [Total_demand_route(i) for i in routes]
    #print('routes',routes)
    
    block=[2,3]
    #block=[2]

    for b in block:
    
        for r1 in range(num_v):
    
            l1=len(routes[r1])
            if l1<2+b:
                continue
            d1=Demand[r1]
            #print('--------------------------------- r1=',r1)
    
            fit1 = fitness_one(routes[r1], dt[r1],l1-2, p_TW) + p_cap*(d1>vehicle_capacity)*(d1-vehicle_capacity)
            
            best_improve = 0
            best_r2 = -1
            best_r1 = -1
            
            #for p1, c in enumerate(routes[r1][1:2], start=1):
            for p1, c in enumerate(routes[r1][1:-b], start=1):
                cc_next= routes[r1][p1:p1+b]
                #print('------- customer',cc_next)
    
                old_fit = fit
                
                l1=len(routes[r1])
                customer_demand = Total_demand_route([0,*cc_next,num_customer+1])
                updated_d1 = (Demand[r1] - customer_demand)
                p_cap1 = (updated_d1>vehicle_capacity)
                
                route1 = copy.deepcopy(routes[r1])
                del route1[p1:p1+b]
    
                #print('uroute1',route1)
                u_fit1 = fitness_one(route1, dt[r1],l1-2-b, p_TW) + p_cap*p_cap1*(updated_d1-vehicle_capacity)
                #print('ufit1',u_fit1)
    
    
                
                for r2 in range(num_v):
                #for r2 in range(2,5):
                    #print('------- r2=',r2)
                    if r1!=r2:
                        l2=len(routes[r2])
                        d2=Demand[r2]
                        fit2 = fitness_one(routes[r2], dt[r2],l2-2, p_TW) + p_cap*(d2>vehicle_capacity)*(d2-vehicle_capacity)
                        #print('fit2',fit2)
                        initial_fit = fit1 + fit2
                        #print('routes[r2]=', routes[r2])
                        #print('initial fit=',fit1,fit2,initial_fit)
    
                        updated_d2 = ( d2 + customer_demand)
                        p_cap2 = (updated_d2>vehicle_capacity)
    
                        for p2 in range(1, l2):
                            #print('-- p2=',p2)
                            if p2==1:
                                route2 = copy.deepcopy(routes[r2])
                                route2[1:1] = cc_next
                            else:
                                temp1 = route2[p2+b-1]
                                route2[p2:p2+b] = route2[p2-1:p2+b-1]

                                
                                #route2[p2:p2+2] = route2[p2-1:p2+1]
                                #route2[p2], route2[p2+1]= route2[p2-1], route2[p2]
                                route2[p2-1]= temp1
                                
                            u_fit2 = fitness_one(route2, dt[r2],l2-2+b, p_TW) + p_cap*p_cap2*(updated_d2-vehicle_capacity)
                            new_fit = u_fit1 + u_fit2
                            
                            improve = initial_fit - new_fit                         
                            
                            if improve > best_improve:
                                best_improve = improve
                                best_route1 = route1.copy()
                                best_route2 = route2.copy()
                                best_r1=r1
                                best_r2=r2
                                best_d1 = updated_d1
                                best_d2 = updated_d2

                        
            if best_improve!=0:            
                routes[best_r1] = best_route1
                routes[best_r2] = best_route2
                Demand[best_r1] = best_d1
                Demand[best_r2] = best_d2
                fit -= best_improve
                fit1 = u_fit1
                
          
        
    Updated_S= [item for sublist in routes for item in sublist]
    fit_Updated_S = fitness(Updated_S, dt, num_v, num_c, p_cap , p_TW)

    return Updated_S, fit_Updated_S



#%%
#ALNS_REMOVAL OPERATORS
yrem = 4  # Randomization factor, adjust as needed


def insertion_cost(customer, solution, dt, num_v, num_c, p_cap, p_TW, fit):
    solution2 = solution.copy()
    solution2.remove(customer)
    f2 = fitness(solution2 , dt, num_v, num_c-1, p_cap, p_TW)
    #print(f2,customer)
    return fit-f2


def worst_removal_operator (solution, dt, num_v, num_c, p_cap,p_TW, fit, a):
    customers=[i for i in solution if i!=0 and i!=num_customer+1]
    costs = [insertion_cost(customer, solution, dt, num_v, num_customer, p_cap,p_TW, fit) for customer in customers]
    
    sorted_customers = sorted(customers, key=lambda x: costs[customers.index(x)], reverse=True)
    #print('sorted_customers',sorted_customers)

    customers_to_remove = []
    while len(customers_to_remove) < a:
        nu = random.uniform(0, 1)  
        i = int(nu ** yrem * len(sorted_customers))  # Calculate the index of the customer to select
        customer = sorted_customers[i]  # Select the customer based on the calculated index
        sorted_customers.remove(customer)  # Remove the customer from the ranked list
        customers_to_remove.append(customer)  # Add the selected customer to the removal list
    return customers_to_remove

def average_travel_time(i,j,dist,ei,si,lj):
    if i==j:
        return 0

    # Get the cluster for the given arc index
    cluster_index = data['clusters'][i][j]
    speed = speeds.get(cluster_index)

    index1 = next((i for i, r in enumerate(time_intervals) if r[0] <= ei+si), None)
    index2 = next((i for i, r in enumerate(time_intervals) if lj <= r[1]), None)
        
    T= lj-(ei+si)
    T += 0.000000001 if T <= 0 else 0

    v= speed[index1] *(time_intervals[index1][1]-(ei+si))
    for k in range(index1+1, index2):
       v += speed[k]* (time_intervals[k][1]-time_intervals[k][0])

    v += speed[index2] *(lj-time_intervals[index2][0])
    #average speed
    v = v/T
    t= dist/v
    return t
    
    
def calculate_closeness(i, j):
    cwt = 0.5
    ctw = 0.85
    
    # Average travel time from customer i to j during feasible time window
    li = nodes[i].due_time  
    ei = nodes[i].ready_time 
    si = nodes[i].service_time  
    lj = nodes[j].due_time 
    ej = nodes[j].ready_time  
    dist= node_dist_mat [i][j]
    
    ti_j = average_travel_time(i,j,dist,ei,si,lj)

    closeness = (ti_j + cwt * max(0, ej - li - si - Travel_Time(si+li , dist ,i,j)) * max(0, si - (li + si)) +
                 ctw * max(0, ei + si + Travel_Time(si+ei , dist ,i,j) - lj) * max(0, (ei + si) - lj))
    
    return closeness


def advanced_shaw_removal(solution, dt, num_v, num_c, p_cap, p_TW, fit, a):
    customers=[i for i in solution if i!=0 and i!=num_customer+1]

    L1 = []
    L2 = []
    unique_list=[]
    
    while len(unique_list) < a:
        if len(L1)==0 or len([c for c in L1 if not c[1]]) == 0:
            i = random.randint(0, num_c - 1)
            customer = customers[i]

            if i == 0:
                neighbor = customers[i+1]
            elif i == num_c - 1:
                neighbor = customers[i-1]
            else:
                neighbor = random.choice([customers[i-1], customers[i+1]])
                
            L1.append([customer, False])  
            L2.append([neighbor, False])


        else:
            i0 = next(c for c in L1 if not c[1])
            #print('i0',i0)
            num_customers = random.randint(1, 2)
            d=customers.copy()

            for _ in range(num_customers):
                # Perform Shaw removal to select a customer
                customer = max(d, key=lambda c: calculate_closeness(c, i0[0]))
                #print('c',customer)
                i0[1] = True  # Mark customer as processed in L1
                L1.append([customer, False]) if not any(item[0] == customer for item in L1 + L2) else None 
                d.remove(customer)
            
                i=customers.index(customer)

                if i == 0:
                    neighbor = customers[i+1]
                elif i == len(customers) - 1:
                    neighbor = customers[i-1]
                else:
                    neighbor = random.choice([customers[i-1], customers[i+1]])

                L2.append([neighbor, False]) if not any(item[0] == customer for item in L1 + L2) else None

        combined_list = L1 + L2 
        # Remove duplicates and convert to list
        unique_list = list(set([entry[0] for entry in combined_list]))  
    
    unique_list = unique_list[:a]  
    #print("Unique :", unique_list)    
    return  unique_list

def fitness_TT(seq, dt):
    
    lenght = len(seq)
    Total_time = 0
    departure_time = dt
    
    for i in range(1, lenght):
        departure_time = departure_time + Travel_Time(departure_time, node_dist_mat[seq[i-1]][seq[i]],seq[i-1],seq[i])
                
        if seq[i] == num_customer+1 :
            Total_time += departure_time - dt

        if departure_time < nodes[seq[i]].ready_time:
            departure_time = nodes[seq[i]].ready_time 
        departure_time += nodes[seq[i]].service_time
        
        if seq[i]==0:
            departure_time = dt
    
    return Total_time

def route_based_shaw_removal(solution, dt, num_v, num_c, p_cap, p_TW, fit, a):
    customers = [i for i in solution if i!=0 and i!=num_customer+1]  
    #print('customers',customers, len(customers))
    #print('solution',solution)
    #create routes
    routes = create_routes(solution, num_v)
        
    # Calculate effectiveness for each route
    effectiveness = np.zeros(num_v, dtype=float)
    for i, route in enumerate(routes):
        if len(route) > 2:
            effectiveness[i] = fitness_TT(route, dt[i]) / Total_demand_route(route)
    
    total_effectiveness = np.sum(effectiveness)
    probability = effectiveness / total_effectiveness
    #print('effectiveness',effectiveness)
    L1 = []
    L2 = []
    unique_list=[]
    len_u= len(unique_list)
    
    while len_u < a :
        #print('-----------')
        #print('len_u',len_u)
    #while sum(1 for _, value in L1 if value) < a:
        if len(L1) == 0 or len([c for c in L1 if not c[1]]) == 0:
            # Select a route based on the effectiveness probability
            selected_route = random.choices(routes, weights=probability)[0]
            #print('selected_route',selected_route)
            L1.extend([[c, False] for c in selected_route[1:-1]])
 
        else:
            i0 = next(c for c in L1 if not c[1])
            num_customers = min(random.randint(1, 2),a-len_u)
            d=customers.copy()

            for _ in range(num_customers):
                # Perform Shaw removal to select a customer
                customer = max(d, key=lambda c: calculate_closeness(c, i0[0]))
                #print('customer',customer)

                i0[1] = True  # Mark customer as processed in L1
                #print('i0',i0)
                L1.append([customer, False]) if not any(item[0] == customer for item in L1 + L2) else None
                #L1.append([customer, False]) 
                #print('L1,',L1)

                d.remove(customer)

                # Append neighbors to L2
                i=customers.index(customer)
                #print('i',i)
                if i == 0:
                    neighbor = customers[i+1]
                elif i == num_c - 1:
                    neighbor = customers[i-1]
                else:
                    neighbor = random.choice([customers[i-1], customers[i+1]])

                #L2.append([neighbor, False]) 
                L2.append([neighbor, False]) if not any(item[0] == neighbor for item in L1 + L2) else None

        
        combined_list = L1 + L2  # Combine L1 and L2
        unique_list = list(set([entry[0] for entry in combined_list]))  # Remove duplicates and convert to list
        len_u= len(unique_list)
       
    unique_list = unique_list[:a]  
    return unique_list
                

def mixed_removal(solution, dt, num_v, num_c, p_cap, p_TW, fit, a):
    customers = [i for i in solution if i!=0 and i!=num_customer+1]  
    
    routes = create_routes(solution, num_v)
    # Calculate effectiveness for each route
    effectiveness = np.zeros(num_v, dtype=float)
    #print('num_v',num_v,'effectiveness',effectiveness)
    for i, route in enumerate(routes):
        if len(route) > 2:
            #print('route',route,dt[i],i)
            effectiveness[i] = fitness_TT(route, dt[i]) / Total_demand_route(route)

    total_effectiveness = np.sum(effectiveness)
    probability = effectiveness / total_effectiveness
    
    L1 = []
    L2 = []
    unique_list=[]
    len_u= len(unique_list)
    
    while len_u < a:
        if len(L1) == 0 or len([c for c in L1 if not c[1]]) == 0:
            
            # Select a route based on the effectiveness probability
            selected_route = random.choices(routes, weights=probability)[0]

            L1.extend([[c, False] for c in selected_route[1:-1]])

            remaining_routes = [route for route in routes if route != selected_route]
            remaininig_customers = [customer for route in remaining_routes for customer in route[1:-1]]
            if remaininig_customers!=[]:
                selected_customer = random.choice(remaininig_customers)
                L1.append([selected_customer,False])
            
        else:
            i0 = next(c for c in L1 if not c[1])
            num_customers = min(random.randint(1, 2), a-len_u)
            d=customers.copy()

            for _ in range(num_customers):
                # Perform Shaw removal to select a customer
                customer = max(d, key=lambda c: calculate_closeness(c, i0[0]))
                i0[1] = True  # Mark customer as processed in L1
                L1.append([customer, False]) if not any(item[0] == customer for item in L1 + L2) else None
                d.remove(customer)

                    
                # Append neighbors to L2
                i=customers.index(customer)

                if i == 0:
                    neighbor = customers[i+1]
                elif i == num_c - 1:
                    neighbor = customers[i-1]
                else:
                    neighbor = random.choice([customers[i-1], customers[i+1]])

                L2.append([neighbor, False]) if not any(item[0] == neighbor for item in L1 + L2) else None

        
        combined_list = L1 + L2  # Combine L1 and L2
        unique_list = list(set([entry[0] for entry in combined_list]))  # Remove duplicates and convert to list
        len_u = len(unique_list)
        
    unique_list = unique_list[:a]  
    return  unique_list


def random_removal(solution, dt, num_v, num_c, p_cap, p_TW, fit, a):
    customers = [i for i in solution if i!=0 and i!=num_customer+1]  
    customer_remove = random.sample(customers,a)
    return customer_remove


#%%
def ALNS_individual(solution , dt, num_v, num_c, p_cap,p_TW, fit):    

    amin = 0.12  # Minimum perturbation rate
    amax = 0.6  # Maximum perturbation rate
    arate = 0.05 # Increase rate of perturbation rate  
    counter_noimprove = 0   
    
    #50 times
    max_number=1
    for i in range(max_number):
        #print('-------------- i:',i)
        
        a = round(min(amax, amin + arate * counter_noimprove) * (num_customer))
        #a = 12

        removal_operators = {
        'Mixed': mixed_removal,   
        'Route_Shaw': route_based_shaw_removal,   
        'Advanced_Shaw': advanced_shaw_removal,
        'Worst': worst_removal_operator,
        'Random': random_removal}

        removed_customers = []
        
        for operator_name, operator_func in removal_operators.items():
            #print('operator_name',operator_name)
            removed_customers = operator_func(solution, dt, num_v, num_c, p_cap, p_TW, fit, a)
            #print('removed_customers',removed_customers)
    
            remained_solution=[c for c in solution if c not in removed_customers]
            #print('remained_solution',remained_solution)
    
            for k in range(1,4):
                #print('k',k)
                #print('removed_customers,k',removed_customers,k)
                
                new_solution = regret_insertion(removed_customers, remained_solution, dt, k, num_v,p_TW)
                #new_solution =[item for sublist in new_routes for item in sublist]
                new_fit = fitness (new_solution,dt, num_v, num_c, p_cap, p_TW)
                

                    
                prob = 0.8
                if new_fit <= fit:
                    solution = new_solution
                    fit = new_fit
                    counter_noimprove=0
                else:
                    counter_noimprove +=1
                    if random.random() < prob:
                        solution = new_solution
                        fit = new_fit    
 
    
    return  solution, fit 

def ALNS_individual_LS(solution , dt, num_v, num_c, p_cap,p_TW, fit):  
   
    amin = 0.12  # Minimum perturbation rate
    amax = 0.6  # Maximum perturbation rate
    arate = 0.05 # Increase rate of perturbation rate  
    counter_noimprove = 0   
    
    #50 times
    max_number=5
    for i in range(max_number):
        #print('----- i:',i)
        
        a = round(min(amax, amin + arate * counter_noimprove) * (num_customer))
        
        removal_operators = {
        'Mixed': mixed_removal,   
        'Route_Shaw': route_based_shaw_removal,   
        'Advanced_Shaw': advanced_shaw_removal,
        'Worst': worst_removal_operator,
        'Random': random_removal}


        removed_customers = []
        
        for operator_name, operator_func in removal_operators.items():
            #print('operator_name',operator_name)
            removed_customers = operator_func(solution, dt, num_v, num_c, p_cap, p_TW, fit, a)
            #print('removed_customers',removed_customers)
    
            remained_solution=[c for c in solution if c not in removed_customers]
            #print('remained_solution',remained_solution)
    
            for k in range(1,4):

                new_solution = regret_insertion(removed_customers, remained_solution, dt, k, num_v,p_TW)
                #new_solution =[item for sublist in new_routes for item in sublist]
                new_fit = fitness (new_solution,dt, num_v, num_c, p_cap, p_TW)
                
                    
                prob = 0    
                if new_fit <= fit:
                    solution = new_solution
                    fit = new_fit
                    counter_noimprove=0
                else:
                    counter_noimprove +=1
                    if random.random() < prob:
                        solution = new_solution
                        fit = new_fit    


    return  solution , dt, num_v, num_c, p_cap,p_TW, fit 

#%%

def initial_population(Problem_Genetic,size, v, p_cap, p_TW): 
    def generate_chromosome_random():
        seq=[0]
        for i in Problem_Genetic.genes:
            seq.append(i)
        dt=[]
        
        for j,tpl in enumerate(seq):
            if tpl==0:
                #r=random.randint(5,15)
                r=6
                dt.append(round(random.uniform(depot_ready_time, depot_due_time/r),2))
                                       
        #v=10
        seq = [seq[0]] + random.sample(seq[1:], num_customer+v-1)

        modified_seq = add_26(seq)
        num_v=len(dt)
        #print('len',len(dt),num_v)                

        fit = fitness(modified_seq, dt , num_v , num_customer, p_cap, p_TW)
        return modified_seq, dt , num_v , num_customer, p_cap, p_TW, fit
        
    def generate_chromosome_regret():
        seq=[0]
        for i in Problem_Genetic.genes:
            seq.append(i)
        
        #v= 10   
        customers=seq[1:-v+1]
        
        dt=[]
        for j,tpl in enumerate(seq):
            if tpl==0:
                r=random.randint(5,15)
                dt.append(round(random.uniform(depot_ready_time, depot_due_time/r),2))
        num_v=len(dt)
        #print('len',len(dt),num_v)                
        solution=[0,num_customer+1]*v

        k=random.randint(1,3)
        #k=2
        chromosome = regret_insertion(customers, solution,dt, k , num_v,p_TW) 
        #first_solution =[item for sublist in chromosome for item in sublist]
        
        fit = fitness(chromosome, dt , num_v , num_customer, p_cap, p_TW)

        return chromosome,dt, num_v, num_customer, p_cap, p_TW, fit

    prob_regret = 0.2
    popu = [generate_chromosome_regret() if random.random() < prob_regret else generate_chromosome_random() for _ in range(size)]
    random.shuffle(popu)
    
    return popu



def do_genetic (Problem_Genetic,k,opt,population,ratio_cross, prob_mutate, size):

    def tournament_selection(Problem_Genetic,population,n,k,opt):
        winners=[]
        for _ in range(n):
            elements = random.sample(population,k)
            winners.append(opt(elements,key=lambda pair: pair[1]))
        return winners
    
    def random_selection(population, n):
        return random.sample(population, n)
    
    def proportional_selection_max(Problem_Genetic, population, n , total_fitness):
        selection_probabilities = [individual[1] / total_fitness for individual in population]
        selected_indices = random.choices(range(size), weights=selection_probabilities, k=n)
        selected_individuals = [population[index] for index in selected_indices]
        return selected_individuals
    
    def proportional_selection(Problem_Genetic, population, n, total_fitness):
        inverse_fitness = [1 / individual[1] for individual in population]
        selection_probabilities = [fit / sum(inverse_fitness) for fit in inverse_fitness]
        selected_indices = random.choices(range(size), weights=selection_probabilities, k=n)
        selected_individuals = [population[index] for index in selected_indices]
        return selected_individuals

    def adaptive_prob_m(Problem_Genetic,fit_avg,population, prob_mutate):
        for i in population:
            if i[1] <= fit_avg:
                prob=0.3
            else:
                prob=0.4
        return prob

    
    def cross_parents(Problem_Genetic,parents,which_cross,num_customer, p_cap, p_TW):
            childs=[]

            for i in range(0,len(parents),2):
                selected_parents_scores = random.sample(parents, 2)
                selected_parents=[z[0] for z in selected_parents_scores]
                
                first_elem_p1 = selected_parents[0][0]
                remaining_elems_p1 = selected_parents[0][1:]

                first_elem_p2 = selected_parents[1][0]
                remaining_elems_p2 = selected_parents[1][1:]

                #crossover function with 1 parent
                if which_cross== 1:
                    child1_c= Problem_Genetic.crossover1_chromosome(remaining_elems_p1)
                    child2_c = Problem_Genetic.crossover1_chromosome(remaining_elems_p2)
                
                
                elif which_cross== 2:
                    child1_c,child2_c =Problem_Genetic.crossover2_chromosome(remaining_elems_p1,remaining_elems_p2)
            
                child1_c = [first_elem_p1] + child1_c
                child2_c = [first_elem_p2] + child2_c
                
                ch1_seq_indices=[]
                p1_dt=[]
                ch2_seq_indices=[]
                p2_dt=[]
                
                for i in range(len(child1_c)):
                    if child1_c[i][0]==0:
                        ch1_seq_indices.append(i)
                        p1_dt.append(child1_c[i][1])
                
                        
                for i in range(len(child2_c)):
                    if child2_c[i][0]==0:
                        ch2_seq_indices.append(i)
                        p2_dt.append(child2_c[i][1])
                
                alpha= random.uniform(0.4,0.6)
                child1_dt,child2_dt =Problem_Genetic.blend_crossover_dt(p1_dt,p2_dt,alpha)
                
                counterr1=0
                for i in range(len(child1_c)):
                    if i in ch1_seq_indices:
                        child1_c[i]=(0,child1_dt[counterr1])
                        counterr1+=1
                
                counterr2=0
                
                for i in range(len(child2_c)):
                    if i in ch2_seq_indices:
                        child2_c[i]=(0,child2_dt[counterr2])
                        counterr2+=1       
                
                childs.extend([child1_c,child2_c])

                parents.remove(selected_parents_scores[0])
                parents.remove(selected_parents_scores[1])     
                
            return childs

    def mutate(Problem_Genetic,population,prob):
        for i in population:
            Problem_Genetic.mutation(i,prob)
        return population
    
    
    def do_selection(Problem_Genetic, population, n, k, opt,total_fit):
        prob=random.random()
        if prob<1/3:
            
            selected = tournament_selection(Problem_Genetic, population, n, k, opt)
        elif prob <2/3:
            selected = random_selection(population, n)
        else: 
            selected = proportional_selection(Problem_Genetic, population, n,total_fit)

        #selected_indi=[z[0] for z in selected]
        #print('selected',selected)
        return selected 
        
    def do_cross(Problem_Genetic,selected,num_customer, p_cap, p_TW):
        prob=0.5
        if random.random() < prob:                    
            #crossover using 1 parent
            crosses = cross_parents(Problem_Genetic,selected,1,num_customer, p_cap, p_TW)
        else:
            #crossover using 2 parents
            crosses = cross_parents(Problem_Genetic,selected,2,num_customer, p_cap, p_TW)
        return crosses
    
    population = [seq_to_chromosome(*i) for i in population]

    #only cheomosome and fit
    population_GA = [(i[0],i[4]) for i in population]
    num_customer, p_cap, p_TW=population[0][1],population[0][2],population[0][3]
    
    
    sorted_population = sorted(population_GA, key=lambda pair: pair[1])
    
    
    n_parents = round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents

   
    # Adaptive prob mutation
    fit_list=[indi[1] for indi in population_GA]
    total_fit= sum(fit_list)
    fit_avg = total_fit / size
    
    
    directs= do_selection(Problem_Genetic, population_GA, n_directs-1, k, opt,total_fit)
    directs=[z[0] for z in directs]
    #Elitism     
    directs.append(sorted_population[0][0])
    #print('directs',directs)
    selected_parents = do_selection(Problem_Genetic, population_GA, n_parents, k, opt,total_fit)
    #print('selected_parents',selected_parents)

    crosses = do_cross(Problem_Genetic,selected_parents,num_customer, p_cap, p_TW)     

    prob_m=adaptive_prob_m(Problem_Genetic,fit_avg,population_GA, prob_mutate)
    mutations = mutate(Problem_Genetic, crosses, prob_m)
    
    new_generation = directs + mutations
    
    new_generation= [(*chromosome_to_seq(i),num_customer, p_cap, p_TW,fitness(*chromosome_to_seq(i),num_customer, p_cap, p_TW)) for i in new_generation]
    #print('new_generation',new_generation)
    new_generation = sorted(new_generation, key=lambda pair: pair[6])

    return new_generation




def add_26(seq):
    modified_seq = []

    for num in seq:
        if num == 0:
            modified_seq.extend([num_customer+1, num])
        else:
            modified_seq.append(num)                    
    modified_seq.pop(0)
    modified_seq.append(num_customer+1)
    return modified_seq
    
def seq_to_chromosome(sequence, dt, num_v, num_customer, p_cap,p_TW, fit):
    #print('sequence',sequence, 'num_v',num_v,dt)
    sequence = [x for x in sequence if x != num_customer+1]
    chromosome = []
    dt_index = 0
    for element in sequence:
        if element == 0:
            chromosome.append((0, dt[dt_index]))
            dt_index += 1
        else:
            chromosome.append((element, nodes[element].demand))
    return chromosome,num_customer, p_cap, p_TW, fit


def chromosome_to_seq(chromosome):
    sequence=[i[0] for i in chromosome]
    dt=[i[1] for i in chromosome if i[0]==0]
    modified_seq = add_26(sequence)
    num_v=len(dt)
    return modified_seq, dt , num_v
    
def check_fitness(population, size):
    for indi in range(size):
        fit = fitness(*population[indi][:6])
        #abs
        if fit != population[indi][6]:
            raise ValueError ('check fitness')


def discarding (population, size, v, p_cap, p_TW,Problem_Genetic):
    gap = 100
    pop_fit=[i[6]  for i in population]
    #print('pop_fit',pop_fit)

    for i in range(size-1):
        if pop_fit[i+1] - pop_fit[i] < gap:
            population[i+1] = initial_population(Problem_Genetic, 1, v, p_cap, p_TW)[0]
            #print('pop[i+1]', population[i+1])

    #print('pop', population)
    population = sorted(population, key=lambda pair: pair[6])
    return population

def local_search_population(population,size):
    for indi in range(size):
        #print('indi',indi)
        population[indi] = Local_Search_individual(*population[indi])
        #print('population[indi]',population[indi][2])                
    population = sorted(population, key=lambda pair: pair[6])
    return population 

def ALNS_local_search_population(population,size):
    for indi in range(size):
        #if indi<5:
        #print('--------- indi',indi)
        #print('population[indi][2]',population[indi][2])                

        population[indi] = ALNS_individual_LS(*population[indi])
        #print('population[indi]',population[indi][6])    
    population = sorted(population, key=lambda pair: pair[6])
    return population 

def ALNS_mutation_population (population,size):
    for indi in range(size):
        #if indi % 3 == 0:
            #print('indi',indi)
        S, dt, num_v, num_c, p_cap,p_TW, fit = population[indi]
        new_solution, new_fit = ALNS_individual(S, dt, num_v, num_c, p_cap,p_TW, fit)
        population [indi] = new_solution, dt, num_v, num_c, p_cap,p_TW, new_fit
    population = sorted(population, key=lambda pair: pair[6])
    return population

def ALNS_mutation_population_best_q (population,size):
    for indi in range(size):
        #if indi % 5 == 0:
            #print('indi',indi)
        S, dt, num_v, num_c, p_cap,p_TW, fit = population[indi]            
        new_solution, new_fit = ALNS_individual(S, dt, num_v, num_c, p_cap,p_TW, fit)
        population [indi] = new_solution, dt, num_v, num_c, p_cap,p_TW, new_fit
    population = sorted(population, key=lambda pair: pair[6])
    return population


def count_empty_route(lst):
    #print('lst',lst)
    count = 0
    i = 0
    while i < len(lst) - 1:
        if lst[i] == 0 and lst[i + 1] == num_customer+1:
            count += 1
            i += 2  # Skip the checked pair
        else:
            i += 1
    #print('count',count)
    return count

def max_vehicle_LS_population(population,size):
    for indi in range(size):
        count_all = population[indi][0].count(0)
        count_empty = count_empty_route(population[indi][0])
        if count_all-count_empty> vehicle_num:
            print('num_v is greater than max vehicles')
            raise ValueError
        

def best_q_selection(population, best_q, q):
    pop=copy.deepcopy(population)
    pop_bestq = pop + best_q
    #print('pop_bestq',pop_bestq)
    pop_bestq = sorted(pop_bestq, key=lambda pair: pair[6])

    best_q = [pop_bestq[0]]
    for i,item in enumerate(pop_bestq):
        if abs(item[6]- pop_bestq[i+1][6])>0.5:
            best_q.append(pop_bestq[i+1])
        if len(best_q)==q:
            break
        
    return best_q

def check_dt_v(population):
    for i in range(len(population)):
        S, dt, num_v, num_c, p_cap,p_TW, fit = population[i]

        c=S.count(0)
        if c!=num_v or c!=len(dt) or num_v!= len(dt):
            print('c',c)
            print('S,dt,num_v',S,dt,num_v)
            raise ValueError

    


#%%

def algorithm (Problem_Genetic,v):

    #parameters
    k = 5
    opt = min
    ngen = 50
    size = 10
    ratio_cross = 0.7
    prob_mutate = 0.3
    
    best_fit_list=[]
    p_cap = 10000
    p_TW = 150000
    q=3


    #Initial Population
    population = initial_population(Problem_Genetic, size, v, p_cap, p_TW)

    best_q= population[:q]

       
    for u in range(1,ngen+1):

        print('----------- iteration', u)

        population = do_genetic (Problem_Genetic, k, opt, population, ratio_cross, prob_mutate, size)

        best_q= best_q_selection (population, best_q,q)
        
        population = ALNS_mutation_population (population,size)
     
        population = ALNS_local_search_population(population,size)
       
        population = local_search_population (population,size)
      
        max_vehicle_LS_population(population,size)

        best_q = best_q_selection(population, best_q,q)
        
        population = discarding (population, size,v, p_cap, p_TW,Problem_Genetic)
       
     
        ##

        best_q = ALNS_local_search_population(best_q,q)
        
        best_q = local_search_population (best_q,q)

        best_q_copy = copy.deepcopy(best_q)
        mutated_best_q = ALNS_mutation_population_best_q (best_q_copy,q)
        
        mutated_best_q = ALNS_local_search_population(mutated_best_q,q)
        mutated_best_q = local_search_population (mutated_best_q,q)
        
        best_q = best_q_selection(mutated_best_q, best_q,q)
        print('best_three_q',[round(i[6],2) for i in best_q])

        ##

        #for plotting fitness
        best_fit_list.append(population[0][6])
        
        #best_fit_list.append(min(i[6] for i in population))
        #bestpop = opt(population, key=lambda x: x[6])
        bestpop = population[0]
        print('best_fit', round(bestpop[6],2))
        #print('best_seq', bestpop[0])
        
    print("------------------------------------------------")
    
    
    x=range(1,u+1)
    plt.plot(x, best_fit_list[:])
    best_three_fit = [round(i[6],2) for i in best_q]

    #bestChromosome = opt(population, key = Problem_Genetic.fitness)
    #print_solution(bestChromosome)
    return best_fit_list[-1], best_three_fit


def VRP(k):
    List_nodes=[]
    for i in range(1,num_customer+1):
        #List_nodes.append((nodes[i].id, nodes[i].demand))
        List_nodes.append(nodes[i].id)
        
    #if there are 10 nodes, we need at most 10 vehicles 
    v= min (num_customer+1,vehicle_num)

    for i in range(v-1):
        List_nodes.append(0) 
    VRP_PROBLEM = Problem_Genetic(List_nodes, num_customer, lambda x: fitness(*x))
    
    import time
    def Run_Algorithms(k):
        cont  = 0
        print("")
        #time_inicial = time()
        start_time = time.perf_counter()

        while cont <= k: 
            last_fit, best_three =algorithm (VRP_PROBLEM,v)
            cont+=1
        #time_final = time()
        end_time = time.perf_counter()
        Total_time = end_time - start_time
        #print("\n") 
        #print("Total time: ",(time_final - time_inicial)," secs.\n")
        print("Total time: ",Total_time," secs.\n")
        return last_fit, best_three,Total_time

    last_fit, best_three, Total_time=Run_Algorithms(k)
    print("--------------------------------------------------")
    return last_fit, best_three,Total_time


#%%

if __name__ == "__main__":
    random.seed(23)
    report=[]
    inst=['R102']

    import datetime
    for i in inst:
        print('----------------------', i,datetime.datetime.now())
        node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity = create_from_file ('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/VRPTW-ACO-python-master/VRPTW-ACO-python-master/'
                                                                                          'solomon-100/{}.txt'.format(i))
        
        num_customer= node_num-2
        
        depot_due_time = nodes[0].due_time
        depot_ready_time = nodes[0].ready_time
        
        
        zones=5
        l0=nodes[0].due_time
        time_intervals= [[0,0.2*l0],[0.2*l0,0.3*l0],[0.3*l0,0.7*l0],[0.7*l0,0.8*l0],[0.8*l0,float('inf')]]
        break_points=[0,0.2*l0,0.3*l0,0.7*l0,0.8*l0,l0]
        l_time_intervals=len(time_intervals)
        
        speeds_slow = np.array([1, 0.333333, 0.666667, 0.5, 0.833333])/10
        speeds_normal= np.array([1.16667, 0.666667, 1.33333, 0.833333, 1])/10
        speeds_fast = np.array([1.5, 1, 1.66667, 1.16667, 1.33333])/10
        
        speeds = {
            0: np.array([1, 0.333333, 0.666667, 0.5, 0.833333]) / 10,
            1: np.array([1.16667, 0.666667, 1.33333, 0.833333, 1]) / 10,
            2: np.array([1.5, 1, 1.66667, 1.16667, 1.33333]) / 10
        }
        
        # Open the JSON file in read mode
        with open('C:/Users/zahra/Desktop/Thesis/VRP-IoT-TT/code VRP/DM-TDVRPTW/DM-TDVRPTW/{}_25.json'.format(i), 'r') as f:
            # Load the JSON data into a Python dictionary
            data = json.load(f)
        
        
        
        # Constant that is an instance object 
        genetic_problem_instances = 0
        #print("EXECUTING ", genetic_problem_instances+1, " INSTANCES ")
        last_fit, best_three,Total_time = VRP(genetic_problem_instances)   
        report.append([i,last_fit, best_three,Total_time])


