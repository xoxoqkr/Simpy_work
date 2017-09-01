import random

import simpy

import math

import operator

import scipy.stats

import numpy as np

#define functions

"""
def experience_factor(departure, arrival, driver_object):
	#the experience is reverse proportional with the distance from the driver`s main area centerector
	driving_experience = round(distance(driver_object.mainarea, departure) + distance(driver_object.mainarea, arrival)/max_parameter)*0.1
	return driving_experience

def EmptyPr(arrival, time):
	time_distance = min(abs(180 - time), abs(530 - time)) # among the gap from lunch(180) and dinner(530), chose close one
	center_distance = min(distance(arrival,[12,37]),distance(arrival,[37,37]),distance(arrival,[12,12]),distance(arrival,[37,12]))
	center_distance = round(center_distance,4)
	max_center_distance = distance([0,0],[12,12])
	return time_distance + center_distance
"""

class Counter(object):
	def __init__(self, env, name, start_location, seats, speed):
		#self.env = env
		self.name = name
		self.taxi = simpy.Resource(env, capacity = 1)
		self.location = start_location
		self.speed = int(speed)
		self.customer = 0
		self.idle = True
		self.seat = seats
		self.next = start_location
		self.endtime = 0
		self.mainarea = [random.choice([12,37]),random.choice([12,37])]
		self.Served = []
		self.Endpoint = [25,25]
		self.Advance = 0 # further update
		self.Experience = 1 - round(random.random(),4)
		self.CurrentCharming = 0
		self.L_k = -random.randrange(250,600)
		self.ETE = 0
		self.Queue_Endpoint = []
		self.Queue_ETE = 0

	def drive(self,customer):
		yield self.env.timeout(int(distance)/self.speed)
		self.location = customer.location2
		print "taxi",self.name, "finish its work at", env.now
		self.customer += 1

	def drive_to_center(self, hotplace):
		#interrupt function must be added
		if self.Endpoint != [25,25]:
			self.ilde = False
			with self.taxi.request() as req:
				tib2 = (self.Endpoint, hotplace)/self.speed # time fo go back to garage
				yield env.timeout(tib2)
				print "taxi is go back to garage"
			self.ilde = True

	def CalculateCharming(self, current_time, objective_charm , customer_popup_interval, variance, new_para):
		"""
		calculate current basket`s charming
		if new_para == 1, this basket is new candidate for selecting
		if new_para == 0, this basket is current doing
		"""
		if new_para == 1:
			Charm1 = 0
			if self.CurrentCharming <= objective_charm:
				Charm1 = (objective_charm - self.CurrentCharming)**(0.88)
			else: 
				Charm1 = -2.25*((self.CurrentCharming - objective_charm)**0.88)
			#Charm = round(Charm,4)
			remain_time = self.Queue_ETE  - current_time
			CalEmptyPr = scipy.stats.norm(customer_popup_interval, variance).pdf(remain_time)
			print("pr", CalEmptyPr, Charm1)
			Charm = objective_charm*(1 + Charm1*0.1) - CalEmptyPr*self.L_k # complete work
			print("main value",objective_charm*(1 + Charm1*0.1),"time weight", - CalEmptyPr*self.L_k)
			print (round(self.Queue_ETE,4) , round(current_time,4), objective_charm, round(CalEmptyPr,4), self.L_k, round(Charm,4))
			self.CurrentCharming = Charm
			return Charm1
		else: 
			return CurrentCharming

	def CalculateOpportunityCost():
		None
		"""
		calcualte the OpportunityCost for driver
		"""
		return 400

		
class customer_gen(object):
	#generate information about customer
	def __init__(self, env, number, max_parameter_value):
		self.name = number
		self.EVENT = simpy.events.Event(env)
		self.REQUEST = 0
		self.departure = [int(random.randrange(1,51)),int(random.randrange(1,51))]
		self.arrival = [int(random.randrange(1,51)),int(random.randrange(1,51))]
		self.occurtime = env.now 
		self.departure_time = 0
		self.arrival_time = 0
		self.weight = 0
		self.information = []
		self.payment = round(math.exp(1)*(distance(self.arrival,self.departure)*max_parameter_value),4) # y=e^x when 0 <= x <= 2


def CalculateObjectiveCharm(basket_info):
	Charm = 0
	BasketTimeList = []
	BasketProfitList = []
	for order in basket_info:
		BasketTimeList.append(distance(order.arrival,order.departure))
		BasketProfitList = [order.payment]
	return round(sum(BasketProfitList)/sum(BasketTimeList),4)


def RoulleteWheel(driver_charm_list,cal_method = 1):
	"""
	by the given list, construct roulletewheel 
	return selected object
	"""
	if cal_method == 1:
		return driver_charm_list.index(max(driver_charm_list))
	else:
		total = sum(driver_charm_list)
		revised_Pr = 0
		revised_Pr_list = []
		for each_driver in driver_charm_list:
			revised_Pr += round(each_driver/total,4)
		select_para = random.random()
		select_index = 0
		for each_driver in revised_Pr_list:
			if each_driver <= select_para:
				return select_index
			select_index += 1


def Queue_length_returner(Resource):
	queue_length = len(Resource.taxi.put_queue)
	#print "queue", len(Resource.taxi.put_queue)
	"""
	queue_length = 0
	print "queue len1"
	if len(Resource.Served) > 0:
		for event in Resource.Served:
			if event.succeed == False:
				queue_length += 1
	print "queue len2"
	"""
	return queue_length


def distance(p1,p2):
	return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2),4)


def source2(env, number, customer_occur_pool, counter_pool):
	for i in range(number):
		c = customer_gen(env, 'Customer%02d' % i , max_parameter)
		#c2 = customer2(env,'Customer%02d' % i, counter_pool ,c)
		c2 = customer3(env,'Customer%02d' % i, counter_pool ,c)
		env.process(c2)
		customer_list.append(c)
		#for j in counter_pool:
		#	print i ,"taxi info" , j.Endpoint, j.endtime
		interval = random.choice(customer_occur_pool)
		yield env.timeout(interval) # this function excute after t simulation times


def charming_function(resource, empty_Pr, payment, departure):
	queue_list_len = Queue_length_returner(resource)
	value = round(payment/2,4) + round(empty_Pr/1,4) + queue_list_len
	if len(resource.taxi.users) == 0: #currently this resource is idle.
		value = 100000
	return value

def selection_process(charming_pool):
	#let`s assume that driver has max charming choose fastest way.
	#driver has max charming take the order.
	if max(charming_pool) < failure_threshold:
		return None
	else:
		return charming_pool.index(max(charming_pool))

def agree_function(career, revised_profit, empty_possiblity):
	#revised_profit is the (his profit)/(average profit)
	value = (Na_E**(2*revised_profit*career - 2)/Na_E)*0.7 - empty_possiblity*0.3
	if random.random() < value:
		return 1
	else:
		return 0

def empty_possiblity(departure, time):
	#return the possiblity of empty comeback 
	#when the rider go back from the location
	time_value = min(abs(time - 180), abs(time - 540))/200
	distance_value = distance(departure, [25,25])/ max_parameter
	return distance_value*0.7 + time_value*0.3


def customer2(env, name, counter_pool, customer_class):
	arrive = env.now
	print('%7.4f %s: Here I am' % (arrive, name))
	possible_list = []
	"""
	select_index = random.randrange(0,3)
	print "selected taxi", select_index
	"""
	#revised rider selection
	charming_pool = []
	for resource_index in range(0,len(counter_pool)):
		empty_PR = empty_possiblity(customer_class.departure , env.now)
		charming = charming_function(counter_pool[resource_index], empty_PR, customer_class.payment*counter_pool[resource_index].Experience, customer_class.departure)
		charming_pool.append(round(charming,4))
	select_index = selection_process(charming_pool)
	#print charming_pool, select_index
	if select_index != None:
		with counter_pool[select_index].taxi.request() as req:
			customer_class.REQUEST = req
			patience = random.uniform(120,180)
			result = yield req | env.timeout(patience)
			wait = env.now - arrive
			if req in result:
				counter_pool[select_index].idle = True
				customer_class.EVENT.succeed()
				counter_pool[select_index].Endpoint = customer_class.departure
				counter_pool[select_index].Served.append(req)
				#print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
				tib =  distance(counter_pool[select_index].next, customer_class.departure)/speed + distance(customer_class.departure, customer_class.arrival)/speed
				yield env.timeout(tib)
				counter_pool[select_index].customer += 1
				counter_pool[select_index].next = customer_class.arrival
				duration = env.now - customer_class.occurtime
				print('%7.4f %s: Finished wait for %7.4f patience %7.4f' % (env.now, name, duration,patience))
				counter_pool[select_index].idle = False
				#added queue calculate
				if counter_pool[select_index].taxi.users != None:
					counter_pool[select_index].Queue_Endpoint = customer_class.arrival
					#print"oh", counter_pool[select_index].Queue_ETE, tib
					counter_pool[select_index].Queue_ETE = counter_pool[select_index].Queue_ETE + tib
					#print "self info", counter_pool[select_index].Queue_Endpoint, counter_pool[select_index].Queue_ETE
				if len(counter_pool[select_index].taxi.users) == 0:
					env.process(counter_pool[select_index].drive_to_center([25,25])) # if the rider is idle it relocate to the center of area
			else:
				print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))
	else:
		print('no one want to %s' % (name))
	raw_input("for pass type enter")



def customer3(env, name, counter_pool, customer_class):
	arrive = env.now
	print('%7.4f %s: Here I am' % (arrive, name))
	possible_list = []
	"""
	select_index = random.randrange(0,3)
	print "selected taxi", select_index
	"""
	#revised rider selection
	charming_pool = []
	objectivecharm = CalculateObjectiveCharm([customer_class])
	for resource_index in range(0,len(counter_pool)):
		empty_PR = empty_possiblity(customer_class.departure , env.now)
		charming2 = counter_pool[resource_index].CalculateCharming(env.now, objectivecharm , mu, sigma , 1)
		#charming = charming_function(counter_pool[resource_index], empty_PR, customer_class.payment*counter_pool[resource_index].Experience, customer_class.departure)
		charming_pool.append(round(charming2,4))
	print ("charming pool",charming_pool)
	print ("max", max(charming_pool))
	select_index = selection_process(charming_pool)
	#print charming_pool, select_index
	if select_index != None:
		print ("it assign to", select_index)
		with counter_pool[select_index].taxi.request() as req:
			customer_class.REQUEST = req
			patience = random.uniform(120,180)
			result = yield req | env.timeout(patience)
			wait = env.now - arrive
			if req in result:
				counter_pool[select_index].idle = True
				customer_class.EVENT.succeed()
				counter_pool[select_index].Endpoint = customer_class.departure
				counter_pool[select_index].Served.append(req)
				#print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
				tib =  distance(counter_pool[select_index].next, customer_class.departure)/speed + distance(customer_class.departure, customer_class.arrival)/speed
				counter_pool[select_index].Queue_Endpoint = customer_class.arrival
				counter_pool[select_index].Queue_ETE = counter_pool[select_index].Queue_ETE + tib
				yield env.timeout(tib)
				counter_pool[select_index].customer += 1
				counter_pool[select_index].next = customer_class.arrival
				duration = env.now - customer_class.occurtime
				print('%7.4f %s: Finished wait for %7.4f patience %7.4f' % (env.now, name, duration,patience))
				counter_pool[select_index].idle = False
				"""
				#added queue calculate
				if counter_pool[select_index].taxi.users != None:
					counter_pool[select_index].Queue_Endpoint = customer_class.arrival
					#print"oh", counter_pool[select_index].Queue_ETE, tib
					counter_pool[select_index].Queue_ETE = counter_pool[select_index].Queue_ETE + tib
					#print "self info", counter_pool[select_index].Queue_Endpoint, counter_pool[select_index].Queue_ETE
				if len(counter_pool[select_index].taxi.users) == 0:
					env.process(counter_pool[select_index].drive_to_center([25,25])) # if the rider is idle it relocate to the center of area
				"""
			else:
				print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, wait))
	else:
		print('no one want to %s' % (name))
	raw_input("for pass type enter")

#running part
global failure_threshold
failure_threshold = 300

global speed
speed = 1

env = simpy.Environment()

global customer_list
customer_list = []

env = simpy.Environment()

global Na_E
Na_E = round(math.e,4)

#parameter for generating the customer
mu = 10
sigma = 2

max_parameter = round(distance([0,0],[50,50])/float(2),2)  #for calculating payment for driver

cap1 = 0
cap2 = 0 
cap3 = 0
Cap_list = [cap1,cap2,cap3]

for i in range(0,3):
	Cap_list[i] = Counter(env, 'c'+ str(i), [25,25], random.choice([4,8]), 2)


customer_occur_pool = np.random.normal(mu, sigma, 10000)

env.process(source2(env,30,customer_occur_pool,Cap_list))
env.run(until = 1200)
